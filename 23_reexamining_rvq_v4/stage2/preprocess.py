"""Audio preprocessing pipeline.

Splits source audio files into fixed-length segments, extracts model
features for each segment via `MyFeatures`, and writes out train/val
filelists pointing at the saved feature tensors.

Segments are saved using sequential numeric IDs rather than the source
file's basename, since a single source file can now produce many
segments. A `manifest.tsv` file in the output directory records, for
every saved segment, its numeric ID, speaker ID, and source audio path.
This manifest makes it possible to:
  * resume an interrupted run (--skip_exists)
  * regenerate train/val filelists without re-running extraction
    (--regen_filelist)
"""

import argparse
import os
import random
import re
from typing import Iterable, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import torch
from tqdm import tqdm

from features import MyFeatures

EXPECTED_KEYS: Tuple[str, ...] = (
    'whisper', 'f0', 'f0_confidence', 'f0_subharmonic',
    'f0_inharmonic', 'spec', 'spk', 'wave',
)

MANIFEST_FILENAME = 'manifest.tsv'


def win_longpath(path: str) -> str:
    """Prefix a path with the Windows long-path marker, if needed."""
    if os.name != 'nt':
        return path
    if path.startswith('\\\\?\\'):
        return path
    return '\\\\?\\' + os.path.abspath(path)


def split_into_segments(data: np.ndarray, sample_rate: int,
                         segment_seconds: float) -> List[np.ndarray]:
    """Split a 1-D audio array into consecutive, non-overlapping chunks.

    The final chunk may be shorter than the rest; callers should discard
    chunks that end up too short to be useful (see `min_segment_seconds`
    in `process_filelist`).
    """
    segment_samples = max(1, int(segment_seconds * sample_rate))
    return [
        data[start:start + segment_samples]
        for start in range(0, data.shape[0], segment_samples)
    ]


def segment_savepaths(output_dir: str, segment_id: str) -> dict:
    """Build the {feature_key: filepath} mapping for a given segment ID."""
    return {
        key: win_longpath(os.path.join(output_dir, f'{segment_id}.{key}'))
        for key in EXPECTED_KEYS
    }


def format_line(savepaths: dict, sid) -> str:
    return '|'.join([savepaths[key] for key in EXPECTED_KEYS] + [str(sid)])


def load_filelist(filelist_path: str,
                   shuffle_seed: int,
                   filepath_regex_pattern: Optional[str],
                   filepath_regex_rep: Optional[str]) -> List[str]:
    with open(filelist_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    random.seed(shuffle_seed)
    random.shuffle(lines)

    if filepath_regex_pattern is not None:
        lines = [
            re.sub(filepath_regex_pattern, filepath_regex_rep, line).replace('\\', '/')
            for line in lines
        ]

    # De-dupe (the regex step above can create duplicates) while
    # preserving the shuffled order. The original implementation used
    # set(), which silently discarded the shuffle's ordering.
    return list(dict.fromkeys(lines))


def write_split(output_dir: str, lines: Sequence[str], f0_lines: Sequence[str],
                 val_fraction: float) -> None:
    """Write train/val filelists, splitting `lines` / `f0_lines` by `val_fraction`."""
    if val_fraction > 0 and len(lines) > 1:
        val_size = max(1, int(len(lines) * val_fraction))
        train_lines, val_lines = lines[:-val_size], lines[-val_size:]
        train_f0, val_f0 = f0_lines[:-val_size], f0_lines[-val_size:]
    else:
        # Too few samples for a meaningful split: keep everything in
        # train, and reuse the first item as a placeholder val sample.
        train_lines, val_lines = lines, lines[:1]
        train_f0, val_f0 = f0_lines, f0_lines[:1]

    def _write(name: str, content: Sequence[str]) -> None:
        with open(os.path.join(output_dir, name), 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))

    _write('train.txt', train_lines)
    _write('val.txt', val_lines)
    _write('train_f0.txt', train_f0)
    _write('val_f0.txt', val_f0)


def process_filelist(filelist_path: str,
                      config: str = 'configs/base_linux.yaml',
                      val_fraction: float = 0.05,
                      output_dir: str = 'output',
                      shuffle_seed: int = 42,
                      feats_to_extract: Optional[Iterable[str]] = None,
                      skip_exists: bool = False,
                      filepath_regex_pattern: Optional[str] = None,
                      filepath_regex_rep: Optional[str] = None,
                      do_normalize: bool = True,
                      segment_seconds: float = 8.0,
                      min_segment_seconds: float = 1.0) -> None:
    """Split audio files into segments, extract features, and write filelists.

    Each source file is loaded once and cut into `segment_seconds` chunks
    in memory; every chunk is extracted and saved independently under a
    sequential numeric ID (e.g. `00000123.whisper`).
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, MANIFEST_FILENAME)

    lines = load_filelist(filelist_path, shuffle_seed, filepath_regex_pattern, filepath_regex_rep)

    if feats_to_extract is None:
        # Do not perform normalization inside the feature extraction (i.e. clip by clip basis)
        extractor = MyFeatures(config=config, do_normalize=False)
    else:
        extractor = MyFeatures(config=config, feats_to_extract=feats_to_extract, do_normalize=False)

    # Resume support: skip source files already recorded in the manifest
    # and continue segment numbering where the previous run left off.
    segment_counter = 0
    already_processed = set()
    resumed = False
    new_lines: List[str] = []
    new_f0_lines: List[str] = []
    if skip_exists and os.path.exists(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for row in f:
                row = row.strip()
                if not row:
                    continue
                segment_id, sid, source_path = row.split('\t')
                already_processed.add(source_path)
                segment_counter = max(segment_counter, int(segment_id) + 1)

                savepaths = segment_savepaths(output_dir, segment_id)
                new_lines.append(format_line(savepaths, sid))
                new_f0_lines.append(savepaths['f0'])
        if already_processed:
            resumed = True
            print(f'Resuming: {len(already_processed)} source file(s) already processed; '
                  f'continuing from segment id {segment_counter}.')

    sid_avgs = {}
    sid_sums = {}
    is_multispk = False
    min_segment_samples = max(1, int(min_segment_seconds * extractor.expected_sample_rate))
    segments_written = 0

    manifest_file = open(manifest_path, 'a' if resumed else 'w', encoding='utf-8')
    try:
        for line in tqdm(lines, total=len(lines), desc='Preprocessing'):
            if 'longform' in line:
                continue

            sid = 0
            if '|' in line:
                if not is_multispk:
                    print('=== Multispeaker filelist detected! ===')
                    is_multispk = True
                line, sid = line.split('|', 1)

            audio_path = win_longpath(line)
            if audio_path in already_processed:
                continue
            if not os.path.exists(audio_path):
                print(f'File not found: {audio_path}')
                continue

            try:
                audio_16k, _ = librosa.load(audio_path, sr=16000)
                audio_48k, _ = librosa.load(audio_path, sr=48000)
                if do_normalize:
                    audio_16k = audio_16k / (np.abs(audio_16k).max()) * 0.99
                    audio_48k = audio_48k / (np.abs(audio_48k).max()) * 0.99
            except Exception as e:
                print(f'Error loading {audio_path}: {e}')
                continue

            for segment_16k, segment_48k in zip(
                split_into_segments(audio_16k, 16000, segment_seconds),
                split_into_segments(audio_48k, 48000, segment_seconds)):
                if segment_16k.shape[0] < min_segment_samples:
                    continue
                if segment_16k.sum() == 0:
                    print(f'Empty segment in {audio_path}, skipping')
                    continue

                try:
                    feats = extractor.extract_features_data(segment_16k, segment_48k)
                except ValueError as e:
                    print(f'Error extracting features for segment of {audio_path}: {e}')
                    continue
                if feats['whisper'].shape[0] < 16:
                    continue

                segment_id = f'{segment_counter:08d}'
                segment_counter += 1
                savepaths = segment_savepaths(output_dir, segment_id)

                for key in EXPECTED_KEYS:
                    value = feats.get(key)
                    if value is not None:
                        torch.save(value, savepaths[key])

                new_lines.append(format_line(savepaths, sid))
                new_f0_lines.append(savepaths['f0'])
                manifest_file.write(f'{segment_id}\t{sid}\t{audio_path}\n')
                manifest_file.flush()
                segments_written += 1

                if feats_to_extract is None or 'spk' in feats_to_extract:
                    if sid not in sid_avgs:
                        sid_avgs[sid] = torch.zeros_like(feats['spk'])
                        sid_sums[sid] = 0
                    sid_avgs[sid] += feats['spk']
                    sid_sums[sid] += 1
    finally:
        manifest_file.close()
        del extractor

    print(f'Wrote {segments_written} segment(s) from {len(lines)} source line(s).')

    if resumed:
        print('Skipped saving sid_avgs.pt because this run resumed from an existing '
              'manifest. Run with --regen_spk_index afterwards to rebuild it.')
    elif feats_to_extract is None or 'spk' in feats_to_extract:
        print('Saving sid_avgs...')
        sid_avgs = {sid: total / sid_sums[sid] for sid, total in sid_avgs.items()}
        torch.save(sid_avgs, os.path.join(output_dir, 'sid_avgs.pt'))

    write_split(output_dir, new_lines, new_f0_lines, val_fraction)


def regen_filelist(output_dir: str, val_fraction: float = 0.05, shuffle_seed: int = 42) -> None:
    """Rebuild train/val filelists from an existing manifest, without re-extracting features."""
    manifest_path = os.path.join(output_dir, MANIFEST_FILENAME)
    with open(manifest_path, 'r', encoding='utf-8') as f:
        entries = [row.strip().split('\t') for row in f if row.strip()]

    random.seed(shuffle_seed)
    random.shuffle(entries)

    new_lines = []
    new_f0_lines = []
    for segment_id, sid, _source_path in entries:
        savepaths = segment_savepaths(output_dir, segment_id)
        new_lines.append(format_line(savepaths, sid))
        new_f0_lines.append(savepaths['f0'])

    write_split(output_dir, new_lines, new_f0_lines, val_fraction)


def regen_spk_index(output_dir: str) -> None:
    """Rebuild sid_avgs.pt from the speaker embeddings referenced in train/val filelists."""
    lines = []
    for name in ('train.txt', 'val.txt'):
        with open(os.path.join(output_dir, name), 'r', encoding='utf-8') as f:
            lines.extend(line.strip() for line in f if line.strip())

    sid_avgs = {}
    sid_sums = {}
    for line in tqdm(lines, total=len(lines), desc='Regenerating speaker index'):
        _, _, _, _, _, _, spkpath, _, sid = line.split('|')
        sid = int(sid)
        spk = torch.load(spkpath)
        if sid not in sid_avgs:
            sid_avgs[sid] = torch.zeros_like(spk)
            sid_sums[sid] = 0
        sid_avgs[sid] += spk
        sid_sums[sid] += 1

    sid_avgs = {sid: total / sid_sums[sid] for sid, total in sid_avgs.items()}
    torch.save(sid_avgs, os.path.join(output_dir, 'sid_avgs.pt'))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--filelist', type=str, help='Path to the source filelist')
    parser.add_argument('--config', type=str, default='configs/base_linux.yaml')
    parser.add_argument('--val_fraction', type=float, default=0.05)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--shuffle_seed', type=int, default=42)
    parser.add_argument('--segment_seconds', type=float, default=8.0,
                         help='Length, in seconds, of each audio segment before feature extraction')
    parser.add_argument('--min_segment_seconds', type=float, default=1.0,
                         help='Trailing segments shorter than this (in seconds) are discarded')
    parser.add_argument('--skip_exists', action='store_true',
                         help='Resume a previous run using its manifest.tsv')
    parser.add_argument('--regen_filelist', action='store_true',
                         help='Rebuild train/val filelists from an existing manifest, '
                              'without re-running feature extraction')
    parser.add_argument('--regen_spk_index', action='store_true',
                         help='Rebuild sid_avgs.pt from existing train/val filelists')
    parser.add_argument('--no_normalize', action='store_true')
    parser.add_argument('--filepath_regex_pattern', type=str, default=None,
                         help='Optional regex applied to each filelist path before processing')
    parser.add_argument('--filepath_regex_rep', type=str, default=None,
                         help='Replacement string for --filepath_regex_pattern')
    return parser


if __name__ == '__main__':
    args = build_arg_parser().parse_args()

    if args.regen_spk_index:
        regen_spk_index(args.output_dir)
    elif args.regen_filelist:
        regen_filelist(args.output_dir, val_fraction=args.val_fraction, shuffle_seed=args.shuffle_seed)
    else:
        process_filelist(
            filelist_path=args.filelist,
            config=args.config,
            val_fraction=args.val_fraction,
            output_dir=args.output_dir,
            shuffle_seed=args.shuffle_seed,
            skip_exists=args.skip_exists,
            filepath_regex_pattern=args.filepath_regex_pattern,
            filepath_regex_rep=args.filepath_regex_rep,
            do_normalize=not args.no_normalize,
            segment_seconds=args.segment_seconds,
            min_segment_seconds=args.min_segment_seconds,
        )