"""Mine high-intensity candidates from the existing DSP features.

The DSP *regressor* compresses the top end (it misses most label-7+ clips), so
this script does not use its absolute score as the miner.  Instead it trains two
small binary classifiers on the existing labels:

    clf6: P(label >= 6)
    clf7: P(label >= 7)

with speaker-grouped out-of-fold predictions (honest for unseen speakers), then
ranks clips by

    rank(clf6) + rank(clf7) + rank(rms_dbfs)

and selects low/mid-labeled clips that the high classifiers are confident
about.  The output is a listening/re-labeling queue, not automatic labels.

Usage:
    python mine_high_intensity.py --features-csv dsp_features.csv \
        --out-csv report/high_intensity_candidates.csv \
        --all-out report/high_intensity_all.csv
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from dedupe_annotations import dedupe
from dsp_features import TIER1


def oof_classifier(X, y, groups, target_thr, n_splits=5, random_state=0):
    """Speaker-grouped out-of-fold P(label >= target_thr)."""
    yb = (y >= target_thr).astype(int)
    n_groups = len(np.unique(groups))
    n_splits = max(2, min(n_splits, n_groups))
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, yb, groups):
        if len(np.unique(yb[tr])) < 2:
            oof[te] = float(yb[tr].mean())
            continue
        model = HGBClassifier(
            max_iter=300, learning_rate=0.05, max_depth=3,
            l2_regularization=1.0, random_state=random_state,
            class_weight='balanced')
        model.fit(X[tr], yb[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]
    return oof


def _rank(x):
    return pd.Series(x).rank(pct=True).to_numpy()


def assign_reason(df, clf7_thr=0.7, clf6_thr=0.8):
    """Priority order: A (7+ low), C (borderline 5), B (6+ low)."""
    label = df['label'].to_numpy()
    reason = np.full(len(df), '', dtype=object)
    b = (label <= 4) & (df['clf6'].to_numpy() >= clf6_thr)
    c = (label == 5) & (df['clf6'].to_numpy() >= 0.9)
    a = (label <= 4) & (df['clf7'].to_numpy() >= clf7_thr)
    reason[b] = 'B_low_clf6'
    reason[c] = 'C_borderline5'
    reason[a] = 'A_low_clf7'
    return reason


def precision_at_k(score, positive, ks):
    order = np.argsort(-np.asarray(score, dtype=float))
    out = {}
    for k in ks:
        k = min(k, len(order))
        out[k] = float(positive[order[:k]].mean())
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--features-csv', type=str, required=True)
    parser.add_argument('--out-csv', type=str,
                        default='report/high_intensity_candidates.csv')
    parser.add_argument('--all-out', type=str,
                        default='report/high_intensity_all.csv')
    parser.add_argument('--model', type=str,
                        default='checkpoints/dsp_intensity/tier1.pkl',
                        help='DSP regressor used only for the dsp_pred column')
    parser.add_argument('--corpora', nargs='+',
                        default=['genshin', 'mlp', 'marefair'])
    parser.add_argument('--clf7-thr', type=float, default=0.7)
    parser.add_argument('--clf6-thr', type=float, default=0.8)
    parser.add_argument('--keep-whisper', action='store_true',
                        help='include whisper/Whispering clips (off by default)')
    parser.add_argument('--label-agg', type=str, default='median',
                        choices=['median', 'mean', 'mode'])
    parser.add_argument('--n-splits', type=int, default=5)
    args = parser.parse_args()

    raw = pd.read_csv(args.features_csv)
    df = dedupe(raw, args.label_agg, keep_features=True)
    print(f'{len(raw)} rows -> {len(df)} unique clips')

    X = df[TIER1].to_numpy(dtype=float)
    y = df['label'].to_numpy(dtype=float)
    groups = df['speaker'].to_numpy()
    df['clf6'] = oof_classifier(X, y, groups, 6, args.n_splits)
    df['clf7'] = oof_classifier(X, y, groups, 7, args.n_splits)
    print(f'clf6 AUC {roc_auc_score((y >= 6).astype(int), df["clf6"]):.3f}  '
          f'positives {int((y >= 6).sum())}')
    print(f'clf7 AUC {roc_auc_score((y >= 7).astype(int), df["clf7"]):.3f}  '
          f'positives {int((y >= 7).sum())}')

    df['score'] = (_rank(df['clf6'].to_numpy())
                   + _rank(df['clf7'].to_numpy())
                   + _rank(df['rms_dbfs'].to_numpy()))
    df['reason'] = assign_reason(df, args.clf7_thr, args.clf6_thr)
    df['conflict'] = (df['label_nunique'] > 1).astype(int)

    if args.model:
        import os
        if os.path.exists(args.model):
            from dsp_intensity import DSPIntensityModel
            dsp = DSPIntensityModel.load(args.model)
            df['dsp_pred'] = dsp.predict(df)
        else:
            print(f'[warn] model not found: {args.model}; skipping dsp_pred')
            df['dsp_pred'] = np.nan
    else:
        df['dsp_pred'] = np.nan

    print('\nOOF ranking yield (deduped):')
    for thr in (7, 6):
        pos = (y >= thr)
        pk = precision_at_k(df['score'].to_numpy(), pos, [25, 50, 100, 200])
        print(f'  label>={thr}  base={pos.mean()*100:.1f}%  '
              + '  '.join(f'P@{k}={v*100:.0f}%' for k, v in pk.items()))

    keep = df['corpus'].isin(args.corpora).to_numpy()
    if not args.keep_whisper:
        keep &= (df['whisper'].to_numpy() == 0)
        keep &= ~df['path'].str.lower().str.contains(
            'whispering').fillna(False).to_numpy()
    df['in_scope'] = keep.astype(int)

    cand = df[keep & (df['reason'] != '')].copy()
    priority = {'A_low_clf7': 0, 'C_borderline5': 1, 'B_low_clf6': 2}
    cand['priority'] = cand['reason'].map(priority).fillna(9).astype(int)
    cand = cand.sort_values(['priority', 'score'], ascending=[True, False])
    cand['new_label'] = ''
    cand['notes'] = ''

    cols = ['path', 'label', 'labels', 'label_span', 'label_nunique', 'leak',
            'splits', 'speaker', 'corpus', 'singing', 'whisper', 'rms_dbfs',
            'clf6', 'clf7', 'dsp_pred', 'score', 'reason', 'conflict',
            'new_label', 'notes']
    import os
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    cand[cols].to_csv(args.out_csv, index=False)

    all_cols = cols[:-2] + ['in_scope']
    df.sort_values('score', ascending=False)[all_cols].to_csv(args.all_out,
                                                              index=False)

    print(f'\n{len(cand)} candidates')
    for reason in ['A_low_clf7', 'C_borderline5', 'B_low_clf6']:
        s = cand[cand['reason'] == reason]
        if len(s):
            print(f'  {reason:14s} {len(s):3d}  '
                  f'by corpus {s["corpus"].value_counts().to_dict()}')
    print(f'\nwrote {args.out_csv}')
    print(f'wrote {args.all_out}')


if __name__ == '__main__':
    main()

