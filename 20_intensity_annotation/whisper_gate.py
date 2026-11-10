"""Whisper / breathiness gate for the DSP intensity model.

A continuous regressor pulls whisper toward the dataset mean (measured: ~4.5
when the model never saw a whisper speaker).  Whisper is a *qualitative*
phonation mode, not a low point on the same continuum, so the DSP pipeline uses
two stages:

    stage A:  P(whisper) from cheap phonation features
    stage B:  continuous intensity for everything else

If stage A fires, the score is forced to `whisper_score` (default 1.0), which is
what the annotation scale means by "1".

The gate is trained on `ex0X_whisper_*` clips (the true unvoiced whispers) and
deliberately *not* on MLP `*_Whispering__*` clips, which are acted/breathy but
voiced and labelled 2-4 in v0.5.

Usage:
    python whisper_gate.py --features-csv dsp_features.csv \
        --out checkpoints/dsp_intensity/whisper_gate.pkl
"""
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Theoretical phonation/breathiness cues.  HNR is optional (Tier 2 only).
GATE_FEATURES = [
    'cpp_mean', 'voiced_frac', 'con2_mean', 'mfcc2_mean',
    'flat_mean', 'zcr_mean', 'cen_mean', 'rms_dbfs', 'hnr_mean',
]


class WhisperGate:
    """Logistic gate with median imputation; NaN-safe for Tier 2 rows.

    The classifier is deliberately combined with a physical rule: whisper is
    *quiet*.  A loud aperiodic shout (e.g. a 0.97 s label-8 clip at -13 dBFS)
    can look whisper-like to the spectral classifier, so we require the clip to
    be below `max_rms_dbfs` before forcing it to 1.
    """

    def __init__(self, threshold=0.8, C=1.0, features=None, whisper_score=1.0,
                 max_rms_dbfs=-21.0):
        self.threshold = threshold
        self.C = C
        self.features = features
        self.whisper_score = whisper_score
        self.max_rms_dbfs = max_rms_dbfs
        self.pipeline = None

    def _columns(self, df):
        feats = self.features
        if feats is None:
            # drop columns that are entirely missing in this cache (e.g. HNR
            # when features were extracted at tier 1)
            feats = [f for f in GATE_FEATURES
                     if f in df.columns and df[f].notna().any()]
        return feats

    def fit(self, df, target='whisper'):
        self.features = self._columns(df)
        self.pipeline = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('lr', LogisticRegression(max_iter=2000, class_weight='balanced',
                                      C=self.C)),
        ])
        self.pipeline.fit(df[self.features].to_numpy(dtype=float),
                          df[target].to_numpy(dtype=int))
        return self

    def predict_proba(self, df):
        X = df[self._columns(df)].to_numpy(dtype=float)
        return self.pipeline.predict_proba(X)[:, 1]

    def predict(self, df):
        fired = self.predict_proba(df) >= self.threshold
        if self.max_rms_dbfs is not None and 'rms_dbfs' in df.columns:
            fired &= df['rms_dbfs'].to_numpy(dtype=float) < self.max_rms_dbfs
        return fired.astype(int)

    def apply(self, df, scores):
        """Force gated rows to `whisper_score`."""
        scores = np.asarray(scores, dtype=float).copy()
        scores[self.predict(df) == 1] = self.whisper_score
        return scores

    # -- persistence --------------------------------------------------------
    def save(self, path):
        payload = {
            'threshold': self.threshold, 'C': self.C,
            'features': self.features, 'whisper_score': self.whisper_score,
            'max_rms_dbfs': self.max_rms_dbfs,
            'pipeline': self.pipeline,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path):
        payload = joblib.load(path)
        gate = cls(threshold=payload['threshold'], C=payload['C'],
                   features=payload['features'],
                   whisper_score=payload.get('whisper_score', 1.0),
                   max_rms_dbfs=payload.get('max_rms_dbfs', -21.0))
        gate.pipeline = payload['pipeline']
        return gate


def cross_val_proba(df, target='whisper', group_col='speaker', n_splits=5):
    """Out-of-fold whisper probabilities with speaker-grouped folds."""
    n_groups = df[group_col].nunique()
    n_splits = max(2, min(n_splits, n_groups))
    oof = np.full(len(df), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(df, df[target], df[group_col]):
        gate = WhisperGate().fit(df.iloc[tr], target=target)
        oof[te] = gate.predict_proba(df.iloc[te])
    return oof


def report_gate(df, oof, target='whisper', max_rms_dbfs=-21.0):
    y = df[target].to_numpy(dtype=int)
    print(f'gate: {int(y.sum())} positives / {len(y)} clips')
    print(f'  AUC = {roc_auc_score(y, oof):.3f}')
    for t in (0.5, 0.7, 0.8, 0.9):
        pred = oof >= t
        if max_rms_dbfs is not None and 'rms_dbfs' in df.columns:
            pred &= df['rms_dbfs'].to_numpy(dtype=float) < max_rms_dbfs
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int(((~pred) & (y == 1)).sum())
        rec = tp / max(tp + fn, 1)
        prec = tp / max(tp + fp, 1)
        print(f'  threshold {t:.1f}: recall={rec:.2f}  precision={prec:.2f}  '
              f'flagged={int(pred.sum())}')
    pos = oof[y == 1]
    if len(pos):
        print(f'  whisper probabilities: min={pos.min():.2f} '
              f'median={np.median(pos):.2f} max={pos.max():.2f}')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--features-csv', type=str, required=True)
    parser.add_argument('--out', type=str, default=None,
                        help='save the fitted gate here')
    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--max-rms-dbfs', type=float, default=-21.0,
                        help='only gate clips quieter than this (whisper is quiet)')
    parser.add_argument('--whisper-score', type=float, default=1.0,
                        help='score assigned when the gate fires')
    parser.add_argument('--eval-only', action='store_true')
    args = parser.parse_args()

    df = pd.read_csv(args.features_csv)
    if 'whisper' not in df.columns:
        raise SystemExit('features csv needs a `whisper` column')

    oof = cross_val_proba(df)
    report_gate(df, oof, max_rms_dbfs=args.max_rms_dbfs)

    if not args.eval_only:
        gate = WhisperGate(threshold=args.threshold,
                           whisper_score=args.whisper_score,
                           max_rms_dbfs=args.max_rms_dbfs).fit(df)
        if args.out:
            import os
            os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
            gate.save(args.out)
            print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
