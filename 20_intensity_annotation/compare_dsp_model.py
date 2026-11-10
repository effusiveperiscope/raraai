"""Head-to-head comparison: DSP-only model vs the whisper intensity model.

Both are evaluated on the same clips and the same labels.  The honest comparison
is `within-speaker r`; pooled r is inflated by speaker/corpus identity (401/440
val clips share a speaker with train).

Usage:
    python compare_dsp_model.py --features-csv dsp_features.csv \
        --dsp-model checkpoints/dsp_intensity/tier1.pkl \
        --ckpt checkpoints/intensity_whisper_small_local/best-checkpoint.ckpt \
        --eval-splits val --n 120
"""
import argparse

import librosa
import numpy as np
import pandas as pd
import torch

from dsp_common import metrics, within_speaker_r
from dsp_intensity import DSPIntensityModel, filter_splits


def score_whisper(df, ckpt, config, whisper_model, window, hop,
                  batch_size, device):
    """Whole-clip pass + sliding-window mean for every row in df."""
    from inference import (load_model_and_context, extract_interp_features,
                           score_features, load_audio)
    from eval_windows import score_windows

    model, context, _ = load_model_and_context(
        ckpt, config, whisper_model=whisper_model, device=device)
    whole, winmean = [], []
    for i in range(0, len(df), batch_size):
        chunk = df.iloc[i:i + batch_size]
        waves = [librosa.util.normalize(load_audio(p)[0]) for p in chunk['path']]
        feats, mask = extract_interp_features(context, waves, device=device)
        pooled, _, _ = score_features(model, feats, mask)
        for j, path in enumerate(chunk['path']):
            whole.append(float(pooled[j].float().cpu()) * 8 + 1)
            _, scores = score_windows(model, context, waves[j],
                                      window, hop, device)
            winmean.append(float(np.mean(scores)))
        print(f'\r{i + len(chunk)}/{len(df)} whisper-scored', end='', flush=True)
    print()
    return np.asarray(whole), np.asarray(winmean)


def _print_row(name, pred, df):
    y = df['label'].to_numpy(dtype=float)
    r, mae, r2 = metrics(pred, y)
    wr = within_speaker_r(pred, y, df['speaker'])
    print(f'{name:22s} pooled r={r:+.3f}  MAE={mae:.3f}  R2={r2:+.3f}  '
          f'within r={wr:+.3f}')
    return {'r': r, 'mae': mae, 'r2': r2, 'within_r': wr}


def _per_corpus(name, pred, df):
    y = df['label'].to_numpy(dtype=float)
    for corpus in sorted(df['corpus'].dropna().unique()):
        m = (df['corpus'] == corpus).to_numpy()
        if m.sum() < 20:
            continue
        r, mae, _ = metrics(pred[m], y[m])
        wr = within_speaker_r(pred[m], y[m], df['speaker'].to_numpy()[m])
        print(f'  {name:10s} {corpus:8s} n={m.sum():4d}  r={r:+.3f}  '
              f'MAE={mae:.3f}  within={wr:+.3f}')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--features-csv', type=str, required=True)
    parser.add_argument('--dsp-model', type=str, default=None,
                        help='saved DSP model; if omitted, fit on --fit-splits')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--config', type=str, default='config/base.yaml')
    parser.add_argument('--whisper_model', type=str, default=None)
    parser.add_argument('--tier', type=int, default=1, choices=[0, 1, 2])
    parser.add_argument('--kind', type=str, default='hgb', choices=['hgb', 'ridge'])
    parser.add_argument('--eval-splits', nargs='+', default=['val'])
    parser.add_argument('--fit-splits', nargs='+', default=['train'])
    parser.add_argument('--n', type=int, default=120, help='0 = all eval clips')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--window', type=float, default=2.0)
    parser.add_argument('--hop', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--out-csv', type=str, default=None)
    args = parser.parse_args()

    all_df = pd.read_csv(args.features_csv)
    all_df = all_df[all_df['path'].notna()].reset_index(drop=True)
    df = filter_splits(all_df, args.eval_splits)
    if args.n and args.n < len(df):
        df = df.sample(n=args.n, random_state=args.seed).sort_index()
        df = df.reset_index(drop=True)
    print(f'{len(df)} eval clips ({args.eval_splits})')

    if args.dsp_model:
        dsp = DSPIntensityModel.load(args.dsp_model)
    else:
        fit_df = filter_splits(all_df, args.fit_splits)
        dsp = DSPIntensityModel(tier=args.tier, kind=args.kind)
        dsp.fit(fit_df, groups=fit_df['speaker'])
    dsp_pred = dsp.predict(df)
    dsp_ungated = dsp.predict(df, apply_gate=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    whole, winmean = score_whisper(
        df, args.ckpt, args.config, args.whisper_model,
        args.window, args.hop, args.batch_size, device)

    print(f'\n=== {len(df)} clips, {args.window:g}s / {args.hop:g}s ===')
    out = {}
    out['whisper_whole'] = _print_row('whisper whole', whole, df)
    out['whisper_window'] = _print_row('whisper window mean', winmean, df)
    out['dsp_ungated'] = _print_row(f'DSP tier{dsp.tier} no gate', dsp_ungated, df)
    out['dsp_gated'] = _print_row(f'DSP tier{dsp.tier} gated', dsp_pred, df)

    print('\nper corpus:')
    _per_corpus('whisper', winmean, df)
    _per_corpus('DSP', dsp_pred, df)

    if 'whisper' in df.columns:
        m = df['whisper'].to_numpy() == 1
        if m.any():
            y = df['label'].to_numpy(dtype=float)
            print(f'\nwhisper subset (n={m.sum()}):')
            for name, pred in (('whisper window', winmean),
                               ('DSP gated', dsp_pred)):
                print(f'  {name:16s} MAE={np.abs(pred[m] - y[m]).mean():.3f}  '
                      f'pred mean={pred[m].mean():.2f}  '
                      f'label mean={y[m].mean():.2f}')

    if args.out_csv:
        out_df = df[['path', 'label', 'speaker', 'corpus', 'whisper']].copy()
        out_df['whisper_whole'] = whole
        out_df['whisper_window'] = winmean
        out_df['dsp_ungated'] = dsp_ungated
        out_df['dsp_gated'] = dsp_pred
        out_df.to_csv(args.out_csv, index=False)
        print(f'\nwrote {args.out_csv}')


if __name__ == '__main__':
    main()

