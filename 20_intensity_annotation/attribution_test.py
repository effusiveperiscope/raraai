"""Attribution faithfulness test.

Question answered: "does this part of the clip contribute to the intensity?"
The model's own answer is the attention head. This script checks whether that
attention is *faithful* by occlusion: silence the most-attended span and a
random span of the same length, and compare the drop in predicted intensity.

A faithful attribution must make the most-attended span matter much more than a
random one. On a sum-pooled MIL model it usually does not -- this test exists so
that failure is visible instead of being read off a pretty per-frame plot.

Usage:
    python attribution_test.py --filelist my_filelist.txt --ckpt ckpt.ckpt --n 48

Exit code is 1 if attention is not faithful (top-attended span <= random span).

Also available: --mode window, which compares the 2 s window with the most
attention mass against a random 2 s window (the granularity at which the model
is actually calibrated).
"""
import argparse
import random

import librosa
import numpy as np
import torch

from inference import (load_model_and_context, extract_interp_features,
                       score_features, load_audio)


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


def silence_span(wave, start_sample, end_sample, fade_samples=160):
    out = wave.copy()
    out[start_sample:end_sample] = 0.0
    fade = min(fade_samples, (end_sample - start_sample) // 2)
    if fade > 0:
        ramp = np.linspace(1.0, 0.0, fade)
        out[start_sample:start_sample + fade] *= ramp
        out[end_sample - fade:end_sample] *= ramp[::-1]
    return out


def infer_one(model, context, wave, device):
    feats, mask = extract_interp_features(context, [wave], device=device)
    pooled, _, attn = score_features(model, feats, mask)
    length = int(mask.sum())
    return (float(pooled.float().cpu().item() * 8 + 1),
            attn[0, :length].float().cpu().numpy(),
            length)


def span_mode(model, context, files, window_sec, span_frac, device):
    drops = {'top': [], 'random': [], 'bottom': []}
    for path, _ in files:
        audio, _ = load_audio(path)
        win = min(len(audio), int(window_sec * 16000))
        start = random.randint(0, max(0, len(audio) - win))
        crop = librosa.util.normalize(audio[start:start + win])

        base, attn, T = infer_one(model, context, crop, device)
        L = max(4, int(round(span_frac * T)))
        cum = np.concatenate([[0.0], np.cumsum(attn)])
        starts = np.arange(0, T - L + 1)
        masses = cum[starts + L] - cum[starts]
        spans = {
            'top': int(starts[masses.argmax()]),
            'bottom': int(starts[masses.argmin()]),
            'random': int(random.choice(starts)),
        }
        for name, s in spans.items():
            occluded = silence_span(crop, s * 160, (s + L) * 160)
            score, _, _ = infer_one(model, context, occluded, device)
            drops[name].append(base - score)
    return drops


def window_mode(model, context, files, window_sec, hop_sec, device):
    """Compare the most-attended window against a random window.

    Tests attention at the granularity the model is actually calibrated for.
    """
    scores = {'top': [], 'random': []}
    win, hop = int(window_sec * 16000), int(hop_sec * 16000)
    for path, _ in files:
        audio, _ = load_audio(path)
        if len(audio) < win:
            continue
        starts = list(range(0, len(audio) - win + 1, hop))
        if len(starts) < 3:
            continue

        # attention mass per window, from a single pass over the whole clip
        _, attn, T = infer_one(model, context, librosa.util.normalize(audio), device)
        mass = []
        for s in starts:
            lo, hi = s // 160, min(T, (s + win) // 160)
            mass.append(attn[lo:hi].sum())

        best = int(np.argmax(mass))
        other = random.choice([i for i in range(len(starts)) if i != best])
        for name, idx in (('top', best), ('random', other)):
            seg = librosa.util.normalize(audio[starts[idx]:starts[idx] + win])
            score, _, _ = infer_one(model, context, seg, device)
            scores[name].append(score)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--filelist', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--config', type=str, default='config/base.yaml')
    parser.add_argument('--n', type=int, default=48, help='number of clips to test')
    parser.add_argument('--window', type=float, default=2.0)
    parser.add_argument('--hop', type=float, default=0.5)
    parser.add_argument('--span', type=float, default=0.25,
                        help='fraction of the crop to silence in span mode')
    parser.add_argument('--mode', choices=['span', 'window'], default='span')
    parser.add_argument('--whisper_model', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, context, _ = load_model_and_context(
        args.ckpt, args.config, whisper_model=args.whisper_model, device=device)

    items = read_filelist(args.filelist)
    random.shuffle(items)
    files = items[:args.n]
    print(f'testing {len(files)} clips, mode={args.mode}')

    if args.mode == 'span':
        drops = span_mode(model, context, files, args.window, args.span, device)
        top, rnd, bot = (np.array(drops[k]) for k in ('top', 'random', 'bottom'))
        print(f'\ndrop in predicted intensity (1-9 units) when silencing '
              f'{args.span:.0%} of the audio:')
        print(f'  most-attended span : {top.mean():+.3f} +/- {top.std():.3f}')
        print(f'  random span        : {rnd.mean():+.3f} +/- {rnd.std():.3f}')
        print(f'  least-attended span: {bot.mean():+.3f} +/- {bot.std():.3f}')
        frac = float((top > rnd).mean())
        print(f'  top > random in {frac:.0%} of clips (chance = 50%)')
        faithful = (frac > 0.65) and (top.mean() > rnd.mean() + 0.1)
        print(f'\nverdict: attention is '
              f'{"FAITHFUL" if faithful else "NOT faithful"} as a per-time attribution')
        raise SystemExit(0 if faithful else 1)

    scores = window_mode(model, context, files, args.window, args.hop, device)
    top, rnd = np.array(scores['top']), np.array(scores['random'])
    print(f'\nwindow score of the most-attended {args.window:g}s window: {top.mean():.2f}')
    print(f'window score of a random {args.window:g}s window        : {rnd.mean():.2f}')
    frac = float((top > rnd).mean())
    print(f'  top > random in {frac:.0%} of clips (chance = 50%)')
    faithful = (frac > 0.65) and (top.mean() > rnd.mean() + 0.1)
    print(f'\nverdict: attention '
          f'{"agrees with" if faithful else "does not agree with"} the window scorer')
    raise SystemExit(0 if faithful else 1)


if __name__ == '__main__':
    main()

