import argparse
import os
import random
import re

import torch
from tqdm import tqdm

from features import MyFeatures

EXPECTED_KEYS = [
    'whisper', 'f0',
    'f0_confidence', 'f0_subharmonic', 'f0_inharmonic',
    'spec', 'spk', 'wave',
]


# --------------------------------------------------------------------------- #
# Path helpers
# --------------------------------------------------------------------------- #

def win_longpath(path):
    """Prefix a path with \\\\?\\ on Windows so long paths are handled correctly."""
    if os.name != 'nt':
        return path
    if path.startswith('\\\\?\\'):
        return path
    return '\\\\?\\' + os.path.abspath(path)


def build_savepaths(output_dir, base_line, aug_idx=None):
    """Return {feature_key: save_path} for a given source line, Windows-safe.

    aug_idx=None -> original (unaugmented) files, same naming as before.
    aug_idx=<int> -> augmented pass N, files get a distinct suffix so they
    never collide with the original or with other augmentation passes.
    """
    savepaths = {}
    suffix = '' if aug_idx is None else f'.aug{aug_idx}'
    for key in EXPECTED_KEYS:
        savepath = os.path.join(
            output_dir, os.path.basename(base_line) + suffix + '.' + key,
        )
        savepaths[key] = win_longpath(savepath)
    return savepaths


def savepaths_to_line(savepaths, sid):
    """Join a savepaths dict + speaker id into the pipe-delimited filelist line."""
    return '|'.join([savepaths[key] for key in EXPECTED_KEYS] + [str(sid)])


# --------------------------------------------------------------------------- #
# Filelist loading / cleanup
# --------------------------------------------------------------------------- #

def load_filelist(filelist_path, shuffle_seed, filepath_regex_pattern, filepath_regex_rep):
    """Read, shuffle, optionally regex-rewrite, and de-duplicate the filelist."""
    with open(filelist_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]

    random.seed(shuffle_seed)
    random.shuffle(lines)

    if filepath_regex_pattern is not None:
        lines = [
            re.sub(filepath_regex_pattern, filepath_regex_rep, line).replace('\\', '/')
            for line in lines
        ]

    # cull potential duplicate lines created by regex/replacement
    return list(set(lines))


def parse_line(line):
    """Split a raw filelist line into (audio_path, speaker_id)."""
    if '|' in line:
        path, sid = line.split('|')[0], line.split('|')[1]
        return path, sid, True
    return line, 0, False


# --------------------------------------------------------------------------- #
# Per-file processing
# --------------------------------------------------------------------------- #

def try_reuse_existing(output_dir, base_line, sid, aug_idx=None):
    """If all feature files for this line already exist, build the line from them."""
    savepaths = build_savepaths(output_dir, base_line, aug_idx=aug_idx)
    if not all(os.path.exists(p) for p in savepaths.values()):
        return None
    print("skip_flag triggered on line ", base_line, '' if aug_idx is None else f'(aug{aug_idx})')
    return savepaths_to_line(savepaths, sid), savepaths['f0']


def _update_sid_avg(sid, feats, sid_avgs, sid_sums):
    if sid not in sid_avgs:
        sid_avgs[sid] = torch.zeros_like(feats['spk'])
        sid_sums[sid] = 0
    sid_avgs[sid] += feats['spk']
    sid_sums[sid] += 1


def extract_and_save(extractor, output_dir, line, base_line, sid, sid_avgs, sid_sums,
                      feats_to_extract, aug_idx=None, aug_seed=None,
                      augment_silence=True, augment_gain=True):
    """Extract features for one file, save them to disk, and update speaker averages.

    If aug_idx is not None, the waveform is augmented (random silence
    insertion + random gain) prior to extraction, using aug_seed for
    reproducibility, and outputs are saved under augmentation-specific
    paths so they don't clobber the original features.
    """
    try:
        if aug_idx is None:
            feats = extractor.extract_features(line)
        else:
            feats = extractor.extract_features_augmented(
                line, seed=aug_seed,
                insert_silence=augment_silence, gain=augment_gain,
            )
        if feats['whisper'].shape[0] < 16:
            print(f'File too short: {line}')
            return None
    except ValueError as e:
        print(f'Error extracting features for {line}: {e}')
        return None

    for key in EXPECTED_KEYS:
        feats.setdefault(key, None)

    savepaths = build_savepaths(output_dir, base_line, aug_idx=aug_idx)
    for key, value in feats.items():
        if value is not None:
            torch.save(value, savepaths[key])

    if feats_to_extract is None or 'spk' in feats_to_extract:
        _update_sid_avg(sid, feats, sid_avgs, sid_sums)

    return savepaths_to_line(savepaths, sid), savepaths['f0']


def build_regen_line(output_dir, line, sid, aug_idx=None):
    """Build a filelist line pointing at feature paths without doing any extraction."""
    savepaths = build_savepaths(output_dir, line, aug_idx=aug_idx)
    return savepaths_to_line(savepaths, sid), savepaths['f0']


# --------------------------------------------------------------------------- #
# Train/val split + output writing
# --------------------------------------------------------------------------- #

def split_train_val(lines, f0_lines, val_fraction):
    if val_fraction > 0:
        val_size = int(len(lines) * val_fraction)
        return (
            lines[:-val_size], lines[-val_size:],
            f0_lines[:-val_size], f0_lines[-val_size:],
        )
    # Ensure at least one val sample even if it's not really a val sample
    return lines, [lines[0]], f0_lines, [f0_lines[0]]


def write_lines(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def save_sid_avgs(output_dir, sid_avgs, sid_sums):
    print('Saving sid_avgs...')
    for sid in sid_avgs:
        sid_avgs[sid] = sid_avgs[sid] / sid_sums[sid]
    torch.save(sid_avgs, os.path.join(output_dir, 'sid_avgs.pt'))


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def process_filelist(filelist_path, config='configs/base_linux.yaml', val_fraction=0.05,
                      output_dir='output', shuffle_seed=42,
                      feats_to_extract=None,
                      regen_filelist=False,
                      skip_exists=False,
                      filepath_regex_pattern=None,
                      filepath_regex_rep=None,
                      do_normalize=True,
                      n_augment=0,
                      augment_silence=True,
                      augment_gain=True):
    """
    n_augment: how many additional augmented copies to generate per source
    file, on top of the original (unaugmented) extraction. Each augmented
    copy gets its own set of feature files (suffixed .aug{i}) and its own
    line in train.txt/val.txt, so the training set effectively grows by
    (1 + n_augment)x. Augmentation is precomputed here (not applied live
    at train time) since it's expensive to redo per-epoch.
    """
    os.makedirs(output_dir, exist_ok=True)

    lines = load_filelist(filelist_path, shuffle_seed, filepath_regex_pattern, filepath_regex_rep)

    extractor = None
    if not regen_filelist:
        extractor_kwargs = {'config': config, 'do_normalize': do_normalize}
        if feats_to_extract is not None:
            extractor_kwargs['feats_to_extract'] = feats_to_extract
        extractor = MyFeatures(**extractor_kwargs)

    is_multispk = False
    new_lines = []
    new_f0_lines = []
    sid_avgs = {}
    sid_sums = {}
    skip_flag = False

    # aug_idx=None means "the original, unaugmented pass". Passes 0..n_augment-1
    # are the augmented copies.
    aug_passes = [None] + list(range(n_augment))

    for raw_line in tqdm(lines, total=len(lines), desc='Preprocessing'):
        if 'longform' in raw_line:
            continue

        base_line, sid, line_is_multispk = parse_line(raw_line)
        if line_is_multispk and not is_multispk:
            print('=== Multispeaker filelist detected! ===')
        is_multispk = is_multispk or line_is_multispk

        line = win_longpath(base_line)
        if not os.path.exists(line):
            print(f'File not found: {line}')
            continue

        for aug_idx in aug_passes:
            # Seed derived from the (shuffle_seed, file, pass) so re-runs are
            # reproducible but different files/passes get different augmentation.
            aug_seed = None
            if aug_idx is not None:
                aug_seed = (hash((shuffle_seed, base_line, aug_idx)) & 0xFFFFFFFF)

            if regen_filelist:
                result = build_regen_line(output_dir, base_line, sid, aug_idx=aug_idx)
            else:
                result = None
                if skip_exists:
                    result = try_reuse_existing(output_dir, base_line, sid, aug_idx=aug_idx)
                    if result is not None:
                        skip_flag = True
                if result is None:
                    result = extract_and_save(
                        extractor, output_dir, line, base_line, sid,
                        sid_avgs, sid_sums, feats_to_extract,
                        aug_idx=aug_idx, aug_seed=aug_seed,
                        augment_silence=augment_silence, augment_gain=augment_gain,
                    )

            if result is None:
                continue

            newline, f0_line = result
            new_lines.append(newline)
            new_f0_lines.append(f0_line)

    train_lines, val_lines, train_f0_lines, val_f0_lines = split_train_val(
        new_lines, new_f0_lines, val_fraction,
    )

    if regen_filelist:
        print('Skipped saving sid_avgs because regen_filelist was True')
    elif skip_flag:
        print('Skipped saving sid_avgs because skip_flag was triggered')  # can't regen sid_avgs if any lines were skipped
    elif feats_to_extract is None or 'sid' in feats_to_extract:
        save_sid_avgs(output_dir, sid_avgs, sid_sums)

    write_lines(os.path.join(output_dir, 'train.txt'), train_lines)
    write_lines(os.path.join(output_dir, 'val.txt'), val_lines)
    write_lines(os.path.join(output_dir, 'train_f0.txt'), train_f0_lines)
    write_lines(os.path.join(output_dir, 'val_f0.txt'), val_f0_lines)

    del extractor


def regen_spk_index(output_dir):
    lines = []
    for filename in ('train.txt', 'val.txt'):
        with open(os.path.join(output_dir, filename), 'r', encoding='utf-8') as f:
            lines.extend([line.strip() for line in f.readlines()])

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

    sid_avgs = {sid: avg / sid_sums[sid] for sid, avg in sid_avgs.items()}
    torch.save(sid_avgs, os.path.join(output_dir, 'sid_avgs.pt'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', type=str, help='path to filelist')
    parser.add_argument('--config', type=str, default='configs/base_linux.yaml')
    parser.add_argument('--val_fraction', type=float, default=0.05)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--shuffle_seed', type=int, default=42)
    parser.add_argument('--regen_filelist', action='store_true')
    parser.add_argument('--regen_spk_index', action='store_true')
    parser.add_argument('--no_normalize', action='store_true')
    parser.add_argument('--n_augment', type=int, default=0,
                         help='number of additional augmented copies to generate per file')
    parser.add_argument('--no_augment_silence', action='store_true',
                         help='disable random silence insertion in augmented passes')
    parser.add_argument('--no_augment_gain', action='store_true',
                         help='disable random gain modification in augmented passes')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.regen_spk_index:
        regen_spk_index(args.output_dir)
        exit(0)

    process_filelist(
        filelist_path=args.filelist,
        config=args.config,
        val_fraction=args.val_fraction,
        output_dir=args.output_dir,
        shuffle_seed=args.shuffle_seed,
        regen_filelist=args.regen_filelist,
        feats_to_extract=None,
        do_normalize=not args.no_normalize,
        n_augment=args.n_augment,
        augment_silence=not args.no_augment_silence,
        augment_gain=not args.no_augment_gain,
        # skip_exists=True,
    )