"""DSP-only intensity profile over time.

Frame features are computed **once** over the whole clip; each sliding window is
then just an aggregation of the frames inside it, so the profile costs almost
nothing beyond the initial extraction.  This is the cheap alternative to
re-running the whisper encoder for every window.

The whisper gate is applied at the *clip* level: if the clip is a whisper, the
whole profile is forced to the whisper score (default 1) instead of letting a
continuous regressor smear it toward the dataset mean.

Usage:
    python dsp_profile.py some.wav \
        --model checkpoints/dsp_intensity/tier1.pkl \
        --window 2.0 --hop 0.5 --out profile.png --json profile.json
"""
import argparse
import json

import numpy as np
import pandas as pd

from dsp_features import SR, clip_features, frame_features, load_wave
from dsp_intensity import DSPIntensityModel


def window_row(frames, times, raw_wave, start_s, end_s):
    """Aggregate the frame features inside [start_s, end_s) into one row."""
    i0 = int(np.searchsorted(times, start_s, side='left'))
    i1 = int(np.searchsorted(times, end_s, side='right'))
    i0 = max(0, min(i0, len(times) - 1))
    i1 = max(i0 + 1, min(i1, len(times)))

    wf = {}
    for k, v in frames.items():
        if k in ('_f0_raw', '_voiced_raw', '_hnr_raw', '_pitch_times'):
            continue
        wf[k] = np.asarray(v)[i0:i1]

    voiced = np.asarray(wf['voiced']) > 0.5
    wf['_f0_raw'] = np.where(voiced, wf['f0_hz'], 0.0)
    wf['_voiced_raw'] = voiced.astype(float)
    wf['_hnr_raw'] = np.asarray(wf['hnr'])
    wf['_pitch_times'] = times[i0:i1]

    s0 = int(start_s * SR)
    s1 = int(end_s * SR)
    raw = raw_wave[s0:s1]
    if len(raw) < SR * 0.05:          # guard tiny tail windows
        raw = raw_wave
    return clip_features(wf, raw)


def predict_row(model, row, apply_gate=True):
    return float(model.predict(pd.DataFrame([row]), apply_gate=apply_gate)[0])


def analyze(wave, model, window=2.0, hop=0.5):
    """Return a dict with the window profile and clip-level scores."""
    raw = np.asarray(wave, dtype=np.float32).squeeze()
    if raw.ndim > 1:
        raw = raw.mean(axis=0)
    norm = raw / (np.max(np.abs(raw)) + 1e-9)
    frames, times = frame_features(norm, include_hnr=model.tier >= 2)

    whole = clip_features(frames, raw)
    whole_score = predict_row(model, whole, apply_gate=True)
    whole_score_raw = predict_row(model, whole, apply_gate=False)

    whisper_prob = 0.0
    if model.gate is not None:
        whole_df = pd.DataFrame([whole])
        whisper_prob = float(model.gate.predict_proba(whole_df)[0])
        is_whisper = bool(model.gate.predict(whole_df)[0])
    else:
        is_whisper = False

    win = int(window * SR)
    hop_n = int(hop * SR)
    if len(raw) <= win:
        starts = [0]
    else:
        starts = list(range(0, len(raw) - win + 1, hop_n))

    rows, centers, durations = [], [], []
    for s in starts:
        e = min(s + win, len(raw))
        rows.append(window_row(frames, times, raw, s / SR, e / SR))
        centers.append((s + e) / 2 / SR)
        durations.append((e - s) / SR)

    wdf = pd.DataFrame(rows)
    if is_whisper:
        whisper_score = model.gate.whisper_score if model.gate else 1.0
        scores = np.full(len(rows), whisper_score)
    else:
        scores = model.predict(wdf, apply_gate=False)

    durations = np.asarray(durations)
    clip_score = float(np.dot(scores, durations) / durations.sum())

    return {
        'times': [s / SR for s in starts],
        'ends': [s / SR + d for s, d in zip(starts, durations)],
        'centers': np.asarray(centers),
        'scores': np.asarray(scores),
        'clip_score': clip_score,
        'whole_score': whole_score,
        'whole_score_ungated': whole_score_raw,
        'whisper_prob': whisper_prob,
        'is_whisper': is_whisper,
        'frames': frames,
        'frame_times': times,
        'whole_features': whole,
        'raw': raw,
    }


def _plot(result, out_path, title=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    audio = result['raw']
    t = np.arange(len(audio)) / SR
    centers = result['centers']
    scores = result['scores']
    ft = result['frame_times']
    frames = result['frames']

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={'height_ratios': [1.0, 1.5, 0.9]})
    axes[0].plot(t, audio, color='steelblue', linewidth=0.4, alpha=0.8)
    axes[0].set_ylabel('amplitude')
    axes[0].set_title(title or 'DSP intensity profile', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, max(t[-1], 0.1))
    axes[0].set_yticks([])

    axes[1].plot(centers, scores, 'o-', color='orangered', linewidth=2,
                 markersize=4, label='window score')
    axes[1].axhline(result['clip_score'], color='k', linestyle='--',
                    linewidth=1, label=f"clip {result['clip_score']:.2f}")
    axes[1].axhline(result['whole_score'], color='gray', linestyle=':',
                    linewidth=1, label=f"whole {result['whole_score']:.2f}")
    axes[1].set_ylim(1, 9)
    axes[1].set_yticks([1, 3, 5, 7, 9])
    axes[1].set_ylabel('intensity (1-9)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right', fontsize=8)

    cpp = np.asarray(frames['cpp'], dtype=float)
    voiced = np.asarray(frames['voiced'], dtype=float)
    cpp_z = (cpp - np.nanmean(cpp)) / (np.nanstd(cpp) + 1e-9)
    axes[2].fill_between(ft, voiced, color='#3ddc84', alpha=0.35,
                         step='mid', label='voiced')
    axes[2].plot(ft, np.clip(cpp_z, -3, 3), color='#9fc2ff', linewidth=0.8,
                 label='CPP (z)')
    axes[2].set_ylim(-3.2, 3.2)
    axes[2].set_ylabel('phonation')
    axes[2].set_xlabel('time (s)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right', fontsize=8)
    axes[2].set_xlim(0, max(t[-1], 0.1))

    if result['is_whisper']:
        axes[1].text(0.01, 0.05,
                     f"whisper gate fired (p={result['whisper_prob']:.2f})",
                     transform=axes[1].transAxes, color='#f0d68a', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)



def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('audio_file', type=str)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--window', type=float, default=2.0)
    parser.add_argument('--hop', type=float, default=0.5)
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--json', type=str, default=None)
    parser.add_argument('--no_plot', action='store_true')
    args = parser.parse_args()

    model = DSPIntensityModel.load(args.model)
    audio = load_wave(args.audio_file)
    result = analyze(audio, model, args.window, args.hop)

    print(f'file:            {args.audio_file}')
    print(f'duration:        {len(audio) / SR:.2f}s, window {args.window}s '
          f'hop {args.hop}s, {len(result["scores"])} windows')
    print(f'clip score:      {result["clip_score"]:.2f}')
    print(f'whole-clip pass: {result["whole_score"]:.2f}')
    print(f'whisper p:       {result["whisper_prob"]:.2f} '
          f'({"fired" if result["is_whisper"] else "not fired"})')
    print(f'window score:    min {result["scores"].min():.2f}  '
          f'max {result["scores"].max():.2f}  std {result["scores"].std():.2f}')
    print('\n  start    end   score')
    for t0, t1, s in zip(result['times'], result['ends'], result['scores']):
        bar = '#' * int(max(0, min(8, round(s - 1))))
        print(f'  {t0:6.2f} {t1:6.2f}  {s:5.2f}  {bar}')

    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump({
                'file': args.audio_file,
                'window_sec': args.window,
                'hop_sec': args.hop,
                'clip_score': result['clip_score'],
                'whole_clip_score': result['whole_score'],
                'whisper_prob': result['whisper_prob'],
                'is_whisper': result['is_whisper'],
                'windows': [{'start': a, 'end': b, 'score': float(s)}
                            for a, b, s in zip(result['times'], result['ends'],
                                               result['scores'])],
            }, f, indent=2)
        print(f'\nwrote {args.json}')

    if args.out and not args.no_plot:
        _plot(result, args.out, title=args.audio_file)
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()

