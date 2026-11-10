"""Scatter plots for the intensity/pitch analysis written by pitch_correlation.py.

Usage:
    python plot_pitch.py --csv pitch_all.csv --out report/pitch_correlation.png
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pitch_correlation import (read_csv_rows, demean_by_speaker, group_indices,
                               corr_pair, partial_corr)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='report/pitch_correlation.png')
    ap.add_argument('--min_agg_clips', type=int, default=5)
    args = ap.parse_args()

    rows = [r for r in read_csv_rows(args.csv) if r['speaker'] != 'unnamed_numeric']
    groups = [r['speaker'] for r in rows]
    inten = np.array([r['intensity_winmean'] for r in rows], float)
    f0 = np.array([r['log2_f0_median'] for r in rows], float)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # pooled
    ax = axes[0]
    c = corr_pair(inten, f0)
    ax.scatter(f0, inten, s=7, alpha=0.30, color='tab:blue', edgecolors='none')
    ax.set_title(f'pooled  r={c["r"]:+.2f} (n={c["n"]})')
    ax.set_xlabel('log2 median F0 (re 1 Hz)')
    ax.set_ylabel('predicted intensity (1-9)')

    # within speaker
    ax = axes[1]
    gidx = group_indices(groups)
    xd = demean_by_speaker(inten, groups, gidx)
    yd = demean_by_speaker(f0, groups, gidx)
    m = np.isfinite(xd) & np.isfinite(yd)
    c = corr_pair(xd, yd)
    ax.scatter(yd[m], xd[m], s=7, alpha=0.25, color='tab:orange', edgecolors='none')
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_title(f'within speaker  r={c["r"]:+.2f} (n={c["n"]})')
    ax.set_xlabel('log2 F0, speaker-demeaned')
    ax.set_ylabel('intensity, speaker-demeaned')

    # across speakers, coloured by corpus (raw r is a between-corpus confound)
    ax = axes[2]
    by = {}
    for r in rows:
        by.setdefault(r['speaker'], []).append(r)
    sp = []
    for nam, rs in by.items():
        rs = [r for r in rs if np.isfinite(r['log2_f0_median'])]
        if len(rs) < args.min_agg_clips:
            continue
        sp.append(dict(corpus=rs[0].get('corpus', 'other'),
                       x=np.median([r['log2_f0_median'] for r in rs]),
                       y=np.mean([r['intensity_winmean'] for r in rs])))
    colors = {'mlp': 'tab:green', 'genshin': 'tab:purple', 'vctk': 'tab:red',
              'justme2': 'tab:gray', 'other': 'tab:brown'}
    for cp in sorted({s['corpus'] for s in sp}):
        sub = [s for s in sp if s['corpus'] == cp]
        ax.scatter([s['x'] for s in sub], [s['y'] for s in sub], s=22,
                   alpha=0.7, color=colors.get(cp, 'tab:blue'), label=cp,
                   edgecolors='none')
    c = corr_pair([s['x'] for s in sp], [s['y'] for s in sp])
    corpora = sorted({s['corpus'] for s in sp})
    dummies = [[1.0 if s['corpus'] == cp else 0.0 for s in sp]
               for cp in corpora[:-1]]
    partial = (np.nan if len(corpora) < 2 else
               partial_corr([s['x'] for s in sp], [s['y'] for s in sp], dummies))
    ax.set_title(f'across speakers  raw r={c["r"]:+.2f}, |corpus r={partial:+.2f}')
    ax.set_xlabel('speaker median log2 F0')
    ax.set_ylabel('speaker mean intensity')
    ax.legend(fontsize=8, frameon=False)


    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
