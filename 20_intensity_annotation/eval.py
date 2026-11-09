"""Per-clip diagnostic plot: waveform, window profile, per-frame head, attention.

Read this plot carefully:
  * the *window profile* is the trustworthy part -- each point is an
    independent forward pass on a short window, the granularity the model is
    trained/calibrated for;
  * the per-frame intensity and the attention curve are NOT calibrated and are
    not a contribution map (see attribution_test.py). They are shown only so you
    can see what the internal heads are doing.
"""
import argparse

import librosa
import numpy as np
import torch

from inference import (load_model_and_context, extract_interp_features,
                       score_features, load_audio)
from eval_windows import score_windows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('audio_file', type=str)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--config', type=str, default='config/base.yaml')
    parser.add_argument('--window', type=float, default=2.0)
    parser.add_argument('--hop', type=float, default=0.5)
    parser.add_argument('--whisper_model', type=str, default=None)
    parser.add_argument('--out', type=str, default=None,
                        help='save plot instead of showing it')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, context, _ = load_model_and_context(
        args.ckpt, args.config, whisper_model=args.whisper_model, device=device)

    audio, _ = load_audio(args.audio_file)
    normalized = librosa.util.normalize(audio)

    # per-frame head + attention (diagnostic only)
    feats, mask = extract_interp_features(context, [normalized], device=device)
    pooled, frame_intensity, attn = score_features(model, feats, mask)
    T = int(mask.sum())
    frame_intensity = frame_intensity[0, :T].float().cpu().numpy() * 8 + 1
    attn = attn[0, :T].float().cpu().numpy()
    feature_time = np.arange(T) / 100.0

    # sliding-window profile (the part that is actually calibrated)
    times, scores = score_windows(model, context, audio, args.window, args.hop, device)
    centers = np.array([(t0 + t1) / 2 for t0, t1 in times])
    clip_score = float(pooled.float().cpu().item() * 8 + 1)

    print(f'whole-clip pass : {clip_score:.2f}')
    print(f'window mean     : {np.mean(scores):.2f}  (std {np.std(scores):.2f}, '
          f'{len(scores)} windows of {args.window:g}s)')

    import matplotlib
    if args.out:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    audio_time = np.arange(len(audio)) / 16000
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    ax1.plot(audio_time, audio, color='steelblue', linewidth=0.4, alpha=0.7)
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Audio: {args.audio_file}', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, audio_time[-1])

    ax2.plot(centers, scores, 'o-', color='orangered', linewidth=2,
             label=f'window score ({args.window:g}s windows)')
    ax2.plot(feature_time, frame_intensity, color='tab:blue', linewidth=0.8, alpha=0.5,
             label='per-frame head (uncalibrated)')
    ax2.axhline(np.mean(scores), color='k', linestyle='--', linewidth=1,
                label=f'window mean {np.mean(scores):.2f}')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Intensity (1-9)')
    ax2.set_ylim(1, 9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, audio_time[-1])

    ax3 = ax2.twinx()
    ax3.fill_between(feature_time, attn, color='green', alpha=0.15, step='mid',
                     label='attention (not a contribution map)')
    ax3.set_ylabel('Attention weight', color='green')
    ax3.tick_params('y', colors='green')
    ax3.set_ylim(0, max(0.02, float(attn.max()) * 4))

    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=8)

    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=130)
        print(f'wrote {args.out}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
