"""Sliding-window intensity scoring.

The model is trained (and only ever calibrated) on short crops, so the honest
way to get an intensity *profile* over time is to score overlapping windows and
treat each window score as one point of the profile. Unlike the per-frame head /
attention in the network, each of these numbers is an independent forward pass
against a target the model was actually supervised on.

Usage:
    python eval_windows.py audio.wav --ckpt checkpoints/xxx/last.ckpt \
        --window 2.0 --hop 0.5 --out profile.png

Prints one line per window (start, end, score 1-9) plus the clip score
(= duration-weighted mean of window scores) and, for reference, a single pass
over the whole clip.
"""
import argparse
import json

import librosa
import numpy as np
import torch

from inference import (load_model_and_context, extract_interp_features,
                       score_features, load_audio)


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def score_windows(model, context, audio, window_sec, hop_sec, device, batch_size=8):
    win = int(window_sec * 16000)
    hop = int(hop_sec * 16000)
    if len(audio) <= win:
        starts = [0]
    else:
        starts = list(range(0, len(audio) - win + 1, hop))
    windows = [librosa.util.normalize(audio[s:s + win]) for s in starts]

    scores = []
    for chunk in batched(windows, batch_size):
        feats, mask = extract_interp_features(context, chunk, device=device)
        pooled, _, _ = score_features(model, feats, mask)
        scores.extend((pooled.float().cpu().numpy() * 8 + 1).tolist())

    times = [(s / 16000, (s + win) / 16000) for s in starts]
    return times, scores

def score_whole(model, context, audio, device):
    """Reference: one forward pass over the whole clip (not how the model was trained)."""
    feats, mask = extract_interp_features(
        context, [librosa.util.normalize(audio)], device=device)
    pooled, _, _ = score_features(model, feats, mask)
    return float(pooled.float().cpu().item() * 8 + 1)




def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('audio_file', type=str)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--config', type=str, default='config/base.yaml')
    parser.add_argument('--window', type=float, default=2.0, help='window length in seconds')
    parser.add_argument('--hop', type=float, default=0.5, help='hop in seconds')
    parser.add_argument('--whisper_model', type=str, default=None,
                        help='default: picked from the checkpoint feature width')
    parser.add_argument('--out', type=str, default=None, help='save plot to this path')
    parser.add_argument('--json', type=str, default=None, help='save scores to this path')
    parser.add_argument('--no_plot', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, context, _ = load_model_and_context(
        args.ckpt, args.config, whisper_model=args.whisper_model, device=device)

    audio, sr = load_audio(args.audio_file)
    if sr != 16000:
        raise ValueError(f'expected 16 kHz, got {sr}')

    times, scores = score_windows(model, context, audio, args.window, args.hop, device)
    whole = score_whole(model, context, audio, device)

    durations = np.array([t1 - t0 for t0, t1 in times])
    weighted = float(np.dot(scores, durations) / durations.sum())
    print(f'file:            {args.audio_file}')
    print(f'duration:        {len(audio)/16000:.2f}s, window {args.window}s hop {args.hop}s, '
          f'{len(scores)} windows')
    print(f'clip score:      {weighted:.2f}  (duration-weighted mean of window scores)')
    print(f'whole-clip pass: {whole:.2f}  (reference, model was not trained on long inputs)')
    print(f'window score:    min {min(scores):.2f}  max {max(scores):.2f}  '
          f'std {np.std(scores):.2f}')
    print('\n  start    end   score')
    for (t0, t1), s in zip(times, scores):
        bar = '#' * int(max(0, min(8, round(s - 1))))
        print(f'  {t0:6.2f} {t1:6.2f}  {s:5.2f}  {bar}')

    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump({
                'file': args.audio_file,
                'window_sec': args.window,
                'hop_sec': args.hop,
                'clip_score': weighted,
                'whole_clip_score': whole,
                'windows': [{'start': t0, 'end': t1, 'score': s}
                            for (t0, t1), s in zip(times, scores)],
            }, f, indent=2)
        print(f'\nwrote {args.json}')

    if args.out and not args.no_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        audio_time = np.arange(len(audio)) / 16000
        centers = np.array([(t0 + t1) / 2 for t0, t1 in times])
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                       gridspec_kw={'height_ratios': [1, 1.4]})
        ax1.plot(audio_time, audio, color='steelblue', linewidth=0.4, alpha=0.7)
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Audio', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, audio_time[-1])

        ax2.plot(centers, scores, 'o-', color='orangered', linewidth=2)
        ax2.axhline(weighted, color='k', linestyle='--', linewidth=1,
                    label=f'clip score {weighted:.2f}')
        ax2.axhline(whole, color='gray', linestyle=':', linewidth=1,
                    label=f'whole-clip pass {whole:.2f}')
        ax2.set_ylim(1, 9)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Intensity (1-9)')
        ax2.set_title(f'Window intensity profile ({args.window:g}s window, '
                      f'{args.hop:g}s hop)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        ax2.set_xlim(0, audio_time[-1])
        plt.tight_layout()
        plt.savefig(args.out, dpi=130)
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()

