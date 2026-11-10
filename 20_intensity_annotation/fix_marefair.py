"""Rebuild the MareFairData `.pt` cache with unique names and correct 16 kHz audio.

The original v0.5 preprocessing had two bugs for this corpus:

  * `preprocess.py` used `Path(audio_path).stem`, so every
    `MareFairData/<character>/_046.flac` wrote the same `_046.pt`; the last
    character processed won and 428 of 711 clips were overwritten.
  * the surviving `.pt` files were saved at the 48 kHz frame count but read as
    16 kHz, so they play 3x too slow (median F0 89 Hz instead of ~267 Hz).

This script writes a separate, correctly resampled tree:

    data/v0.5_fixed/marefair/<character>/<stem>.pt

and a map/filelist with one row per source annotation.  The original
`data/v0.5` files are not modified.

Usage:
    python fix_marefair.py --annotation-json annotations/v0.5.json \
        --out-dir data/v0.5_fixed/marefair \
        --map-out report/marefair_fixed_map.csv \
        --filelist-out data/v0.5_fixed/marefair_filelist.txt
"""
import argparse
import hashlib
import json
import os

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch

SRC_ROOT = '/mnt/data/DataAugmentation/MareFairData'
TARGET_SR = 16000


def source_to_local(key):
    """Windows annotation path -> local Linux path."""
    return key.replace('D:\\', '/mnt/data/').replace('\\', '/')


def deterministic_split(path, val_fraction=0.1):
    digest = hashlib.md5(path.encode('utf-8')).hexdigest()
    bucket = int(digest[:8], 16) % 1000
    return 'val' if bucket < int(val_fraction * 1000) else 'train'


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--annotation-json', type=str,
                        default='annotations/v0.5.json')
    parser.add_argument('--out-dir', type=str,
                        default='data/v0.5_fixed/marefair')
    parser.add_argument('--map-out', type=str,
                        default='report/marefair_fixed_map.csv')
    parser.add_argument('--filelist-out', type=str,
                        default='data/v0.5_fixed/marefair_filelist.txt')
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    with open(args.annotation_json, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    items = [(k, v.get('intensity', 0)) for k, v in annotations.items()
             if 'MareFairData' in k]
    if args.limit:
        items = items[:args.limit]
    print(f'{len(items)} MareFairData annotations')

    rows = []
    skipped = []
    for i, (key, label) in enumerate(items, 1):
        source = source_to_local(key)
        if not os.path.exists(source):
            skipped.append(source)
            continue
        character = os.path.basename(os.path.dirname(source))
        stem = os.path.splitext(os.path.basename(source))[0]
        out_path = os.path.join(args.out_dir, character, f'{stem}.pt')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        info = sf.info(source)
        if args.overwrite or not os.path.exists(out_path):
            audio, sr = sf.read(source, dtype='float32', always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != TARGET_SR:
                audio = librosa.resample(audio, orig_sr=sr,
                                         target_sr=TARGET_SR, res_type='fft')
            torch.save(torch.from_numpy(np.ascontiguousarray(audio,
                                                             dtype=np.float32)),
                       out_path)
        else:
            sr = info.samplerate
        pt_frames = int(torch.load(out_path, map_location='cpu',
                                   weights_only=False).numel())

        rows.append({
            'source': source,
            'pt': out_path,
            'label': int(label),
            'character': character,
            'split': deterministic_split(key, args.val_fraction),
            'src_sr': int(info.samplerate),
            'src_frames': int(info.frames),
            'src_dur': round(info.duration, 3),
            'pt_frames': pt_frames,
            'pt_dur': round(pt_frames / TARGET_SR, 3),
            'ratio': round(info.frames / max(pt_frames, 1), 3),
        })
        if i % 50 == 0 or i == len(items):
            print(f'\r{i}/{len(items)} written', end='', flush=True)
    print()

    if skipped:
        print(f'{len(skipped)} missing sources (skipped):')
        for p in skipped:
            print('  ', p)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.map_out) or '.', exist_ok=True)
    df.to_csv(args.map_out, index=False)

    with open(args.filelist_out, 'w', encoding='utf-8') as f:
        for _, r in df.iterrows():
            f.write(f'{r["pt"]}|{r["label"]}\n')

    print(f'\nwrote {len(df)} clips')
    print(f'  map      {args.map_out}')
    print(f'  filelist {args.filelist_out}')
    print(f'  split    {df["split"].value_counts().to_dict()}')
    print(f'  ratio    median {df["ratio"].median():.3f}  '
          f'min {df["ratio"].min():.3f}  max {df["ratio"].max():.3f}')
    print(f'  duration source median {df["src_dur"].median():.2f}s -> '
          f'pt median {df["pt_dur"].median():.2f}s')


if __name__ == '__main__':
    main()
