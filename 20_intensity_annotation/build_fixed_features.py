"""Build a corrected DSP feature cache by replacing the broken MareFairData rows.

The original `dsp_features.csv` contains 710 `unnamed` rows backed by only 283
colliding, 3x-slowed `.pt` files.  `fix_marefair.py` writes 710 unique, correctly
resampled tensors; this script:

  1. drops every `corpus == "unnamed"` row from the old cache,
  2. extracts tier-1 features for the fixed MareFairData tensors,
  3. concatenates them into `dsp_features_fixed.csv`,
  4. rebuilds train/val filelists with the fixed paths.

Usage:
    python build_fixed_features.py \
        --base-csv dsp_features.csv \
        --marefair-map report/marefair_fixed_map.csv \
        --out-csv dsp_features_fixed.csv \
        --out-dir data/v0.5_fixed --workers 4
"""
import argparse
import os
import re

import pandas as pd

from dsp_common import read_filelist
from dsp_features import batch_extract, write_csv


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base-csv', type=str, default='dsp_features.csv')
    parser.add_argument('--marefair-map', type=str,
                        default='report/marefair_fixed_map.csv')
    parser.add_argument('--out-csv', type=str, default='dsp_features_fixed.csv')
    parser.add_argument('--out-dir', type=str, default='data/v0.5_fixed')
    parser.add_argument('--tier', type=int, default=1, choices=[0, 1, 2])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--skip-filelists', action='store_true')
    args = parser.parse_args()

    base = pd.read_csv(args.base_csv)
    n_old = len(base)
    base = base[base['corpus'] != 'unnamed'].reset_index(drop=True)
    print(f'base: {n_old} rows -> {len(base)} after dropping broken `unnamed`')

    fixed = pd.read_csv(args.marefair_map)
    missing = [p for p in fixed['pt'] if not os.path.exists(p)]
    if missing:
        raise SystemExit(f'{len(missing)} fixed tensors missing, e.g. {missing[:3]}')
    items = [(r.pt, int(r.label), str(r.split)) for r in fixed.itertuples()]
    rows = batch_extract(items, tier=args.tier, workers=args.workers)
    print(f'extracted {len(rows)} fixed MareFairData clips')

    meta = fixed.set_index('pt')
    for row in rows:
        m = meta.loc[row['path']]
        row['speaker'] = str(m['character'])
        row['corpus'] = 'marefair'
        row['split'] = str(m['split'])

    new = pd.DataFrame(rows)
    out = pd.concat([base, new], ignore_index=True, sort=False)
    write_csv(out.to_dict('records'), args.out_csv)
    print(f'wrote {args.out_csv}: {len(out)} rows, '
          f'{out["path"].nunique()} unique paths')

    if not args.skip_filelists:
        os.makedirs(args.out_dir, exist_ok=True)
        pattern = re.compile(r'^_\d+\.pt$')
        for split in ('train', 'val'):
            old = read_filelist(f'data/v0.5/{split}_filelist_linux.txt')
            kept = [(p, l) for p, l in old
                    if not pattern.match(os.path.basename(p))]
            added = [(r.pt, int(r.label)) for r in fixed.itertuples()
                     if str(r.split) == split]
            out_path = os.path.join(args.out_dir, f'{split}_filelist_linux.txt')
            with open(out_path, 'w', encoding='utf-8') as f:
                for p, l in kept + added:
                    f.write(f'{p}|{l}\n')
            print(f'{split}: {len(kept)} kept + {len(added)} fixed = '
                  f'{len(kept) + len(added)}  -> {out_path}')


if __name__ == '__main__':
    main()
