"""Deduplicate v0.5 rows by audio path and flag conflicting labels.

The v0.5 filelists contain repeated paths (same audio, multiple annotation
rows) and 58 paths appear in both train and val.  Training on those rows
teaches the model contradictory targets and contaminates validation.

This script produces two tables:

  * `report/dedupe_all.csv`      one row per unique path, with an aggregate
                                 label (median by default) and conflict flags
  * `report/label_conflicts.csv` paths with conflicting labels or split
                                 leakage, sorted by label span (worst first)

Usage:
    python dedupe_annotations.py --features-csv dsp_features.csv \
        --all-out report/dedupe_all.csv \
        --conflicts-out report/label_conflicts.csv
"""
import argparse

import numpy as np
import pandas as pd


def aggregate_labels(labels, how='median'):
    """Aggregate repeated labels for one path."""
    labels = sorted(int(v) for v in labels)
    if how == 'median':
        return int(np.floor(np.median(labels) + 0.5))
    if how == 'mean':
        return int(round(float(np.mean(labels))))
    if how == 'mode':
        vals, counts = np.unique(labels, return_counts=True)
        return int(vals[np.argmax(counts)])
    raise ValueError(f'unknown label aggregation: {how}')


def dedupe(df, label_agg='median', keep_features=False):
    """Return one row per unique `path` with aggregate label + audit flags.

    `keep_features=True` keeps the feature columns from the first row (needed
    by the mining script); the audit CSV uses the default compact form.
    """
    rows = []
    for path, g in df.groupby('path', sort=False):
        labels = sorted({int(v) for v in g['label']})
        splits = sorted({str(v) for v in g['split']})
        first = g.iloc[0]
        row = first.to_dict() if keep_features else {}
        row.pop('split', None)
        row.update({
            'path': path,
            'label': aggregate_labels(labels, label_agg),
            'labels': str(labels),
            'label_min': min(labels),
            'label_max': max(labels),
            'label_span': max(labels) - min(labels),
            'label_nunique': len(labels),
            'n_rows': len(g),
            'label_median': float(np.median(labels)),
            'label_mean': float(np.mean(labels)),
            'label_mode': aggregate_labels(labels, 'mode'),
            'splits': '+'.join(splits),
            'split': splits[0] if len(splits) == 1 else 'leak',
            'leak': int(len(splits) > 1),
            'speaker': first.get('speaker', ''),
            'corpus': first.get('corpus', ''),
            'singing': int(first.get('singing', 0)),
            'whisper': int(first.get('whisper', 0)),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--features-csv', type=str, required=True)
    parser.add_argument('--all-out', type=str, default='report/dedupe_all.csv')
    parser.add_argument('--conflicts-out', type=str,
                        default='report/label_conflicts.csv')
    parser.add_argument('--label-agg', type=str, default='median',
                        choices=['median', 'mean', 'mode'])
    args = parser.parse_args()

    df = pd.read_csv(args.features_csv)
    out = dedupe(df, args.label_agg)

    conflicts = out[(out['label_nunique'] > 1) | (out['leak'] == 1)].copy()
    conflicts = conflicts.sort_values(
        ['label_span', 'label_nunique', 'n_rows'], ascending=False)

    import os
    for path in (args.all_out, args.conflicts_out):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    out.to_csv(args.all_out, index=False)
    conflicts.to_csv(args.conflicts_out, index=False)

    dup = out[out['n_rows'] > 1]
    print(f'rows in        {len(df)}')
    print(f'unique clips   {len(out)}')
    print(f'duplicated     {len(dup)} paths ({int(dup["n_rows"].sum())} rows)')
    print(f'conflicting    {int((out["label_nunique"] > 1).sum())} paths')
    print(f'train+val leak {int((out["leak"] == 1).sum())} paths')
    print(f'span >= 2      {int((out["label_span"] >= 2).sum())} paths')
    print(f'span >= 3      {int((out["label_span"] >= 3).sum())} paths')
    print(f'contain >= 7   {int(((out["label_nunique"] > 1) & (out["label_max"] >= 7)).sum())} paths')
    print(f'\nwrote {args.all_out}')
    print(f'wrote {args.conflicts_out} ({len(conflicts)} audit rows)')

    if len(conflicts):
        cols = ['path', 'labels', 'splits', 'n_rows', 'corpus', 'speaker']
        print('\nworst conflicts:')
        print(conflicts[cols].head(12).to_string(index=False))


if __name__ == '__main__':
    main()
