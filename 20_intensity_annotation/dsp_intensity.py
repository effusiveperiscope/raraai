"""Fit / evaluate the DSP-only intensity model (Tier 0/1/2 features).

This is the cheap, interpretable alternative to the whisper model:

    features (dsp_features.py)  ->  HGB / ridge  ->  optional whisper gate

The model predicts the 1-9 annotation directly.  `--cv group` holds out whole
speakers (hard); `--cv random` is the comparable-to-the-whisper-model setting
because 401/440 val clips share a speaker with train.

Usage:
    # speaker-grouped CV + fit on train + evaluate on val, save the model
    python dsp_intensity.py --features-csv dsp_features.csv --tier 1 \
        --cv group --fit-gate --save checkpoints/dsp_intensity/tier1.pkl

    # evaluate a saved model
    python dsp_intensity.py --features-csv dsp_features.csv \
        --load checkpoints/dsp_intensity/tier1.pkl --eval-splits val
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dsp_common import metrics, within_speaker_r
from dsp_features import FEATURE_TIERS
from whisper_gate import WhisperGate


class DSPIntensityModel:
    """HGB/ridge regressor with optional whisper gate and linear calibration."""

    def __init__(self, tier=1, kind='hgb', gate=None, calibration=True,
                 random_state=0):
        self.tier = int(tier)
        self.kind = kind
        self.features = list(FEATURE_TIERS[self.tier])
        self.gate = WhisperGate.load(gate) if isinstance(gate, str) else gate
        self.calibration = calibration
        self.random_state = random_state
        self.model = None
        self.cal = (1.0, 0.0)

    # -- estimator ----------------------------------------------------------
    def _make_estimator(self):
        if self.kind == 'hgb':
            return HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, max_depth=3,
                l2_regularization=1.0, random_state=self.random_state)
        if self.kind == 'ridge':
            return Pipeline([
                ('impute', SimpleImputer(strategy='median')),
                ('scale', StandardScaler()),
                ('ridge', RidgeCV(alphas=np.logspace(-3, 3, 25))),
            ])
        raise ValueError(f'unknown model kind: {self.kind}')

    def _X(self, df):
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise ValueError(
                f'features csv is missing columns for tier {self.tier}: {missing}. '
                f'Re-extract with `dsp_features.py --tier {self.tier}`'
                + (' --include-hnr' if self.tier >= 2 else '') + '.')
        return df[self.features].to_numpy(dtype=float)

    # -- fit ----------------------------------------------------------------
    def fit(self, df, target='label', groups=None, calibrate=None):
        X = self._X(df)
        y = df[target].to_numpy(dtype=float)
        calibrate = self.calibration if calibrate is None else calibrate
        if calibrate:
            groups = (df['speaker'].to_numpy() if groups is None
                      else np.asarray(groups))
            oof = self._oof_predictions(df, target, groups)
            mask = np.isfinite(oof) & np.isfinite(y)
            if mask.sum() >= 10 and np.std(oof[mask]) > 1e-6:
                a, b = np.polyfit(oof[mask], y[mask], 1)
                # keep the correction mild; a wild slope is a small-fold artefact
                self.cal = (float(np.clip(a, 0.5, 2.0)),
                            float(np.clip(b, -2.0, 2.0)))
        self.model = self._make_estimator().fit(X, y)
        return self

    def _oof_predictions(self, df, target, groups, n_splits=5):
        n_groups = len(np.unique(groups))
        if n_groups < 2:
            return np.full(len(df), np.nan)
        n_splits = max(2, min(n_splits, n_groups))
        oof = np.full(len(df), np.nan)
        for tr, te in GroupKFold(n_splits=n_splits).split(df, df[target], groups):
            m = self._make_estimator()
            m.fit(self._X(df.iloc[tr]),
                  df[target].iloc[tr].to_numpy(dtype=float))
            oof[te] = m.predict(self._X(df.iloc[te]))
        return oof

    # -- predict ------------------------------------------------------------
    def predict(self, df, apply_gate=True, calibrate=True):
        raw = self.model.predict(self._X(df))
        if calibrate and self.calibration:
            a, b = self.cal
            raw = a * raw + b
        raw = np.clip(raw, 1.0, 9.0)
        if apply_gate and self.gate is not None:
            raw = self.gate.apply(df, raw)
        return raw

    def predict_raw(self, df):
        return self.model.predict(self._X(df))

    # -- persistence --------------------------------------------------------
    def save(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        joblib.dump({
            'tier': self.tier, 'kind': self.kind, 'features': self.features,
            'calibration': self.calibration, 'cal': self.cal,
            'random_state': self.random_state, 'model': self.model,
            'gate': self.gate,
        }, path)

    @classmethod
    def load(cls, path):
        payload = joblib.load(path)
        m = cls(tier=payload['tier'], kind=payload['kind'],
                gate=payload.get('gate'),
                calibration=payload.get('calibration', True),
                random_state=payload.get('random_state', 0))
        m.features = payload.get('features', m.features)
        m.cal = tuple(payload.get('cal', (1.0, 0.0)))
        m.model = payload['model']
        return m



# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def filter_splits(df, splits):
    if not splits or 'split' not in df.columns:
        return df.reset_index(drop=True)
    return df[df['split'].isin(splits)].reset_index(drop=True)


def fit_gate(df, threshold=0.8, whisper_score=1.0, max_rms_dbfs=-21.0):
    """Fit a whisper gate, or return None if the fold has no whisper clips."""
    if 'whisper' not in df.columns or df['whisper'].nunique() < 2:
        return None
    return WhisperGate(threshold=threshold, whisper_score=whisper_score,
                       max_rms_dbfs=max_rms_dbfs).fit(df)


def evaluate(df, pred, target='label', name='model'):
    y = df[target].to_numpy(dtype=float)
    pred = np.asarray(pred, dtype=float)
    r, mae, r2 = metrics(pred, y)
    wr = within_speaker_r(pred, y, df['speaker'])
    print(f'{name}: n={len(df)}  pooled r={r:+.3f}  MAE={mae:.3f}  '
          f'R2={r2:+.3f}  within-speaker r={wr:+.3f}')

    per_corpus = {}
    for corpus in sorted(df['corpus'].dropna().unique()):
        m = (df['corpus'] == corpus).to_numpy()
        if m.sum() < 20:
            continue
        cr, cmae, _ = metrics(pred[m], y[m])
        cwr = within_speaker_r(pred[m], y[m], df['speaker'].to_numpy()[m])
        per_corpus[corpus] = {'n': int(m.sum()), 'r': cr, 'mae': cmae,
                              'within_r': cwr}
        print(f'  {corpus:8s} n={m.sum():4d}  r={cr:+.3f}  MAE={cmae:.3f}  '
              f'within={cwr:+.3f}')

    whisper = {}
    if 'whisper' in df.columns:
        m = df['whisper'].to_numpy() == 1
        if m.any():
            whisper = {
                'n': int(m.sum()),
                'mae': float(np.abs(pred[m] - y[m]).mean()),
                'pred_mean': float(pred[m].mean()),
                'label_mean': float(y[m].mean()),
            }
            print(f'  whisper  n={m.sum():4d}  MAE={whisper["mae"]:.3f}  '
                  f'pred mean={whisper["pred_mean"]:.2f}  '
                  f'label mean={whisper["label_mean"]:.2f}')

    return {'n': int(len(df)), 'r': r, 'mae': mae, 'r2': r2,
            'within_r': wr, 'per_corpus': per_corpus, 'whisper': whisper}


def cross_validate(df, tier=1, kind='hgb', cv='group', target='label',
                   gate=True, gate_threshold=0.8, whisper_score=1.0,
                   gate_max_rms=-21.0, n_splits=5):
    """Out-of-fold predictions with speaker-grouped or random folds."""
    if cv == 'group':
        n_groups = df['speaker'].nunique()
        n_splits = max(2, min(n_splits, n_groups))
        splits = list(GroupKFold(n_splits=n_splits).split(
            df, df[target], df['speaker']))
    else:
        splits = list(KFold(n_splits=n_splits, shuffle=True,
                            random_state=0).split(df))

    pred = np.full(len(df), np.nan)
    for tr, te in splits:
        tr_df, te_df = df.iloc[tr], df.iloc[te]
        gate_obj = (fit_gate(tr_df, gate_threshold, whisper_score,
                             gate_max_rms) if gate else None)
        model = DSPIntensityModel(tier=tier, kind=kind, gate=gate_obj,
                                  calibration=True)
        model.fit(tr_df, target=target, groups=tr_df['speaker'])
        pred[te] = model.predict(te_df)
    return pred


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--features-csv', type=str, required=True)
    parser.add_argument('--dedupe', action='store_true',
                        help='aggregate repeated paths before training/eval')
    parser.add_argument('--label-agg', type=str, default='median',
                        choices=['median', 'mean', 'mode'])
    parser.add_argument('--tier', type=int, default=1, choices=[0, 1, 2])
    parser.add_argument('--kind', type=str, default='hgb', choices=['hgb', 'ridge'])
    parser.add_argument('--cv', type=str, default='group',
                        choices=['group', 'random', 'none'])
    parser.add_argument('--cv-splits', nargs='+', default=['train', 'val'])
    parser.add_argument('--fit-splits', nargs='+', default=['train'])
    parser.add_argument('--eval-splits', nargs='+', default=['val'])
    parser.add_argument('--target', type=str, default='label')
    parser.add_argument('--gate', type=str, default=None,
                        help='load a pre-fitted whisper gate (.pkl)')
    parser.add_argument('--fit-gate', action='store_true',
                        help='fit a whisper gate on the fit split')
    parser.add_argument('--no-gate', action='store_true')
    parser.add_argument('--gate-threshold', type=float, default=0.8)
    parser.add_argument('--gate-max-rms', type=float, default=-21.0,
                        help='only gate clips quieter than this (whisper is quiet)')
    parser.add_argument('--whisper-score', type=float, default=1.0)
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--report', type=str, default=None,
                        help='write a JSON summary here')
    args = parser.parse_args()

    df = pd.read_csv(args.features_csv)
    df = df[df['path'].notna()].reset_index(drop=True)
    if args.dedupe:
        from dedupe_annotations import dedupe
        n_raw = len(df)
        df = dedupe(df, args.label_agg, keep_features=True)
        print(f'deduped: {n_raw} rows -> {len(df)} unique clips '
              f'({int((df["leak"] == 1).sum())} train/val leaks marked split=leak)')

    if args.load:
        model = DSPIntensityModel.load(args.load)
        eval_df = filter_splits(df, args.eval_splits)
        pred = model.predict(eval_df)
        result = evaluate(eval_df, pred,
                          target=args.target,
                          name=f'loaded {os.path.basename(args.load)}')
        result.update({'tier': model.tier, 'kind': model.kind,
                       'cal': list(model.cal),
                       'gate': model.gate is not None})
        if args.report:
            os.makedirs(os.path.dirname(args.report) or '.', exist_ok=True)
            with open(args.report, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            print(f'wrote {args.report}')
        return

    # -- cross-validation (optional) ---------------------------------------
    if args.cv != 'none':
        cv_df = filter_splits(df, args.cv_splits)
        if len(cv_df) >= 10:
            print(f'\n=== {args.cv} CV on {len(cv_df)} clips '
                  f'({"+".join(args.cv_splits)}) ===')
            cv_pred = cross_validate(
                cv_df, tier=args.tier, kind=args.kind, cv=args.cv,
                target=args.target, gate=not args.no_gate,
                gate_threshold=args.gate_threshold,
                whisper_score=args.whisper_score,
                gate_max_rms=args.gate_max_rms, n_splits=args.n_splits)
            evaluate(cv_df, cv_pred, target=args.target,
                     name=f'{args.kind} tier{args.tier} CV')

    # -- final fit ----------------------------------------------------------
    fit_df = filter_splits(df, args.fit_splits)
    if len(fit_df) < 10:
        raise SystemExit(f'not enough fit clips ({len(fit_df)})')

    gate_obj = None
    if not args.no_gate:
        if args.gate:
            gate_obj = WhisperGate.load(args.gate)
        elif args.fit_gate:
            gate_obj = fit_gate(fit_df, args.gate_threshold,
                                args.whisper_score, args.gate_max_rms)
            if gate_obj is None:
                print('[gate] no whisper clips in fit split; gate disabled')

    model = DSPIntensityModel(tier=args.tier, kind=args.kind, gate=gate_obj,
                              calibration=True)
    model.fit(fit_df, target=args.target, groups=fit_df['speaker'])
    print(f'\nfit on {len(fit_df)} clips  tier={args.tier}  kind={args.kind}  '
          f'gate={"yes" if gate_obj else "no"}  cal={model.cal[0]:.2f}x'
          f'{model.cal[1]:+.2f}')

    if args.save:
        model.save(args.save)
        print(f'wrote {args.save}')

    # -- held-out evaluation ------------------------------------------------
    eval_df = filter_splits(df, args.eval_splits)
    if len(eval_df):
        print(f'\n=== held-out: {"+".join(args.eval_splits)} ===')
        pred = model.predict(eval_df)
        result = evaluate(eval_df, pred, target=args.target,
                          name=f'{args.kind} tier{args.tier}')
        result.update({'tier': args.tier, 'kind': args.kind,
                       'cal': list(model.cal),
                       'gate': gate_obj is not None,
                       'fit_splits': args.fit_splits,
                       'eval_splits': args.eval_splits})
        if args.report:
            os.makedirs(os.path.dirname(args.report) or '.', exist_ok=True)
            with open(args.report, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            print(f'wrote {args.report}')


if __name__ == '__main__':
    main()

