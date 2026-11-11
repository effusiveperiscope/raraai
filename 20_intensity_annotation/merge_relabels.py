"""Merge exported re-label CSVs into the fixed DSP feature cache.

The interactive report exports `path,new_label,note,...`.  This script joins
those labels to the cached unlabeled features (`dsp_features_diverse.csv`),
assigns a train/val split (by speaker hash, so a speaker is not split across
clips), and appends them to `dsp_features_fixed.csv`.

Usage:
    python merge_relabels.py \
        --labels relabeled_diverse_high_intensity_candidates.csv \
        --features dsp_features_diverse.csv \
        --base dsp_features_fixed.csv \
        --out dsp_features_augmented.csv \
        --val-fraction 0.2
"""
import argparse
import hashlib
import os

import numpy as np
import pandas as pd


def read_relabels(paths):
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if 'new_label' in df.columns:
            labels = df['new_label']
        elif 'label' in df.columns:
            labels = df['label']
        else:
            raise ValueError(f'{path}: no new_label/label column')
        notes = df['note'] if 'note' in df.columns else df.get(
            'notes', pd.Series([''] * len(df)))
        frames.append(pd.DataFrame({
            'path': df['path'],
            'label': pd.to_numeric(labels, errors='coerce'),
            'notes': notes.fillna('').astype(str),
        }))
    out = pd.concat(frames, ignore_index=True)
    out = out[out['label'].notna()].copy()
    out['label'] = out['label'].astype(int)
    out = out[out['label'].between(1, 9)]
    if out['path'].duplicated().any():
        out = out.groupby('path', as_index=False).agg(
            label=('label', lambda x: int(np.floor(np.median(x) + 0.5))),
            notes=('notes', 'last'))
    return out.reset_index(drop=True)


def split_for_speaker(speaker, val_fraction):
    if val_fraction <= 0:
        return 'train'
    digest = hashlib.md5(str(speaker).encode('utf-8')).hexdigest()
    bucket = int(digest[:8], 16) % 1000
    return 'val' if bucket < val_fraction * 1000 else 'train'


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--labels', nargs='+',
                        default=['relabeled_diverse_high_intensity_candidates.csv'])
    parser.add_argument('--features', type=str,
                        default='dsp_features_diverse.csv')
    parser.add_argument('--base', type=str, default='dsp_features_fixed.csv')
    parser.add_argument('--out', type=str, default='dsp_features_augmented.csv')
    parser.add_argument('--val-fraction', type=float, default=0.2,
                        help='fraction of new speakers held out for validation')
    args = parser.parse_args()

    rel = read_relabels(args.labels)
    feats = pd.read_csv(args.features).drop_duplicates('path')
    new = rel.merge(feats, on='path', how='inner', suffixes=('', '_feat'))
    if 'label_feat' in new.columns:
        new = new.drop(columns=['label_feat'])
    missing = len(rel) - len(new)
    if missing:
        print(f'[warn] {missing} relabeled paths absent from the feature cache')
    new['split'] = [split_for_speaker(s, args.val_fraction)
                    for s in new['speaker']]
    new['label'] = new['label'].astype(int)

    base = pd.read_csv(args.base)
    combined = pd.concat([base, new], ignore_index=True, sort=False)
    before = len(combined)
    combined = combined.drop_duplicates('path', keep='last')
    if len(combined) != before:
        print(f'[warn] dropped {before - len(combined)} duplicate paths')
    combined.to_csv(args.out, index=False)

    labels = new['label'].value_counts().sort_index().to_dict()
    print(f'relabeled: {len(rel)} rows, {len(new)} matched features')
    print(f'  label distribution {labels}')
    print(f'  label>=7 {int((new["label"] >= 7).sum())}  '
          f'label>=8 {int((new["label"] >= 8).sum())}')
    print(f'  new split {new["split"].value_counts().to_dict()}')
    print(f'combined: {len(base)} base + {len(new)} new = {len(combined)} rows')
    print(f'  all label>=7 {int((combined["label"] >= 7).sum())}  '
          f'label>=8 {int((combined["label"] >= 8).sum())}')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
