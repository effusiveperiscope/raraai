"""Clip / window level metrics over a filelist.

Reports, on a held-out filelist:
  * whole-clip pass        -> r, MAE, R^2 against the clip label
  * mean of window scores  -> r, MAE, R^2 (the inference-time path)
  * every window vs label  -> r, MAE (weakly supervised: all windows of a clip
                              share the clip label, so this is a lower bound)
  * within-clip std of window scores (how much profile a clip actually has)

Usage:
    python eval_val.py --filelist data/v0.5/val_filelist_linux.txt \
        --ckpt checkpoints/<exp>/last.ckpt --n 200
"""
import argparse

import numpy as np
import torch

from inference import (load_model_and_context, extract_interp_features,
                       score_features, load_audio)


def metrics(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    r = float(np.corrcoef(pred, target)[0, 1]) if pred.std() > 0 else float('nan')
    mae = float(np.abs(pred - target).mean())
    ss_res = float(((pred - target) ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return r, mae, r2


def read_filelist(path):
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            audio_path, label = line.rsplit('|', 1)
            items.append((audio_path.strip(), int(label)))
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--filelist', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--config', type=str, default='config/base.yaml')
    parser.add_argument('--n', type=int, default=0, help='0 = all clips')
    parser.add_argument('--window', type=float, default=2.0)
    parser.add_argument('--hop', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--whisper_model', type=str, default=None)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, context, _ = load_model_and_context(
        args.ckpt, args.config, whisper_model=args.whisper_model, device=device)

    items = read_filelist(args.filelist)
    if args.n:
        items = items[:args.n]

    win, hop = int(args.window * 16000), int(args.hop * 16000)
    whole_scores, window_means, labels, all_windows, all_window_labels = [], [], [], [], []
    within_std = []

    for i in range(0, len(items), args.batch_size):
        chunk = items[i:i + args.batch_size]
        waves = []
        for path, _ in chunk:
            audio, _ = load_audio(path)
            waves.append(audio)
        feats, mask = extract_interp_features(context, waves, device=device)
        pooled, _, _ = score_features(model, feats, mask)
        for j, (path, label) in enumerate(chunk):
            whole_scores.append(float(pooled[j].float().cpu()) * 8 + 1)
            labels.append(label)

            audio = waves[j]
            starts = ([0] if len(audio) <= win
                      else list(range(0, len(audio) - win + 1, hop)))
            windows = [audio[s:s + win].copy() for s in starts]
            wfeats, wmask = extract_interp_features(context, windows, device=device)
            wpooled, _, _ = score_features(model, wfeats, wmask)
            ws = wpooled.float().cpu().numpy() * 8 + 1
            window_means.append(float(ws.mean()))
            within_std.append(float(ws.std()))
            all_windows.extend(ws.tolist())
            all_window_labels.extend([label] * len(ws))
        print(f'\r{i + len(chunk)}/{len(items)} clips', end='', flush=True)
    print()

    all_windows = np.array(all_windows)
    all_window_labels = np.array(all_window_labels, dtype=np.float64)
    labels = np.array(labels, dtype=np.float64)

    print(f'\n=== {len(items)} clips, {args.window:g}s window / {args.hop:g}s hop ===')
    for name, pred in (('whole-clip pass', whole_scores), ('mean of windows', window_means)):
        r, mae, r2 = metrics(pred, labels)
        print(f'{name:18s} r={r:+.3f}  MAE={mae:.3f}  R2={r2:+.3f}')

    r, mae, r2 = metrics(all_windows, all_window_labels)
    print(f'{"per-window":18s} r={r:+.3f}  MAE={mae:.3f}  R2={r2:+.3f}   '
          f'({len(all_windows)} windows)')
    print(f'within-clip window std: mean {np.mean(within_std):.3f}  '
          f'p90 {np.percentile(within_std, 90):.3f}  '
          f'(label std {labels.std():.3f})')
    print(f'window score range: {all_windows.min():.2f} .. {all_windows.max():.2f}  '
          f'(labels {labels.min():.0f} .. {labels.max():.0f})')


if __name__ == '__main__':
    main()
