import librosa
from omegaconf import OmegaConf
import torch
from features import FeatureExtractor
from utils import win_longpath
import ultimate_xc
import random
import os
import argparse
from tqdm import tqdm

# Keys produced by FeatureExtractor.extract_features_ac, in filelist order.
FEATURE_KEYS = ['wave', 'spec', 'f0']

def feature_paths(output_dir, base_line, keys=FEATURE_KEYS):
    """Return the on-disk tensor paths for one source file.

    Tensors are named by *basename* only (see AGENTS.md footgun 11).
    """
    return [win_longpath(os.path.join(output_dir,
        os.path.basename(base_line) + '.' + key)) for key in keys]

def saved_spec_frames(spec_path):
    """Number of frames in a saved stage-1 `.spec` tensor.

    The spec is `[-1, fft_size // 2 + 1]`, so dim 0 is the frame (sequence)
    dim that must be long enough for training. `mmap=True` only maps the file,
    so checking many clips does not read the tensor data.
    """
    try:
        spec = torch.load(spec_path, map_location='cpu', weights_only=True,
                          mmap=True)
    except (RuntimeError, OSError, ValueError):
        # Filesystems without mmap support (or an older save format).
        spec = torch.load(spec_path, map_location='cpu', weights_only=True)
    return int(spec.shape[0])

def process_filelist(filelist_path,
    val_fraction=0.05, output_dir='data/preprocessed', shuffle_seed=42,
    skip_if_one_exists=False, min_frames=None):

    with open(filelist_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]

    os.makedirs(output_dir, exist_ok=True)

    random.seed(shuffle_seed)
    random.shuffle(lines)

    hp = OmegaConf.load("configs/base.yaml")
    # One training segment is `segment_size // hop_length` frames; shorter
    # clips cannot be sliced into a segment and break training.
    if min_frames is None:
        min_frames = int(hp.data.segment_size) // int(hp.data.hop_length)

    # The extractor pulls in Qwen3-ASR/forced aligner + RMVPE. Only build it
    # when at least one file actually has to be extracted, so an incremental
    # `--skip_existing` run over an already-preprocessed corpus stays cheap.
    fe = None

    is_multispk = False
    new_lines = []
    n_reused = 0
    n_processed = 0
    n_skipped_short = 0

    for line in tqdm(lines, total=len(lines), desc='Preprocessing'):
        if not line:
            continue
        if 'longform' in line:
            continue
        if '|' in line:
            if not is_multispk:
                print('=== Multispeaker filelist detected! ===')
            is_multispk = True
            split = line.split('|')
            line = split[0]
            sid = split[1]
        else:
            sid = 0

        base_line = line
        line = win_longpath(line)

        savepaths = feature_paths(output_dir, base_line)

        # Reuse already-preprocessed tensors: no audio load, no feature
        # extraction, just a length check against `min_frames`.
        if skip_if_one_exists and all(os.path.exists(p) for p in savepaths):
            try:
                n_frames = saved_spec_frames(savepaths[1])
            except Exception as e:
                print(f'Error reading {savepaths[1]}: {e}')
                n_frames = None
            if n_frames is not None:
                if min_frames > 0 and n_frames < min_frames:
                    print(f'Skipping {base_line}: {n_frames} frames < '
                          f'min_frames ({min_frames})')
                    n_skipped_short += 1
                    continue
                n_reused += 1
                new_lines.append('|'.join(savepaths) + f'|{sid}')
                continue
            # Unreadable tensors: fall through and re-extract them.

        if not os.path.exists(line):
            print(f'File not found: {line}')
            continue

        try:
            wav, _ = librosa.load(line, sr=48000)
            if fe is None:
                fe = FeatureExtractor(hp=hp)
            feats = fe.extract_features_ac(wav)
            n_frames = int(feats['spec'].shape[0])
            for key, value in feats.items():
                savepath = os.path.join(output_dir,
                    os.path.basename(base_line) + '.' + key)
                savepath = win_longpath(savepath)
                torch.save(value, savepath)

        except Exception as e:
            print(f'Error processing {line}: {e}')
            continue

        # Too-short clips are still written to disk (so a later
        # `--skip_existing` run does not re-extract them) but never make it
        # into train.txt / val.txt.
        if min_frames > 0 and n_frames < min_frames:
            print(f'Skipping {base_line}: {n_frames} frames < '
                  f'min_frames ({min_frames})')
            n_skipped_short += 1
            continue

        n_processed += 1
        new_lines.append('|'.join(savepaths) + f'|{sid}')

    print(f'Kept {len(new_lines)} file(s) '
          f'({n_reused} reused, {n_processed} newly processed); '
          f'omitted {n_skipped_short} clip(s) shorter than {min_frames} '
          f'frame(s).')

    # Split the *kept* items, not the raw line count: omitting short clips
    # must not shrink train and overflow val.
    n_valid = len(new_lines)
    if val_fraction > 0 and n_valid > 1:
        val_size = max(1, int(n_valid * val_fraction))
        val_size = min(val_size, n_valid - 1)
        val_lines = new_lines[-val_size:]
        train_lines = new_lines[:-val_size]
    else:
        val_lines = new_lines[:1] # ensure at least one item in val for checkpointing
        train_lines = new_lines

    with open(os.path.join(output_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_lines))
    with open(os.path.join(output_dir, 'val.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_lines))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', type=str, help='path to filelist', required=True)
    parser.add_argument('--val_fraction', type=float, default=0.05)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--shuffle_seed', type=int, default=42)
    parser.add_argument('--skip_existing', action='store_true',
        help='reuse tensors already present in --output_dir instead of '
             're-extracting them (only the clip length is checked)')
    parser.add_argument('--min_frames', type=int, default=None,
        help='omit clips whose spec has fewer frames from train.txt/val.txt '
             '(default: data.segment_size // data.hop_length, i.e. one '
             'training segment = 16 frames; 0 disables the check)')

    args = parser.parse_args()

    process_filelist(
        filelist_path=args.filelist,
        val_fraction=args.val_fraction,
        output_dir=args.output_dir,
        shuffle_seed=args.shuffle_seed,
        skip_if_one_exists=args.skip_existing,
        min_frames=args.min_frames,
    )