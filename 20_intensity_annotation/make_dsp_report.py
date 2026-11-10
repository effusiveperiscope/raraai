"""Self-contained HTML report of DSP intensity ratings (audio + profile).

This mirrors `make_report.py` (dark cards, embedded audio/plots, one file) but
uses the DSP model, so you can *listen* to examples across the 1-9 range and
see the phonation cues behind each score.

By default it samples a couple of clips per intensity bin from the eval split,
falling back to train clips (clearly marked) for bins that are empty in eval
(e.g. 8-9), and always includes true whisper examples.

Usage:
    python make_dsp_report.py --features-csv dsp_features.csv \
        --model checkpoints/dsp_intensity/tier1.pkl \
        --eval-splits val --per-bin 2 --out report/dsp_intensity.html
"""
import argparse
import base64
import html
import io
import os

import numpy as np
import pandas as pd

from dsp_common import metrics, within_speaker_r
from dsp_intensity import DSPIntensityModel, filter_splits
from dsp_profile import analyze
from make_report import build_css, fig_data_uri, wav_data_uri

BINS = [(1, 2), (3, 4), (5, 6), (7, 7), (8, 8), (9, 9)]
COLORS = {'profile': '#ff7a45', 'voiced': '#3ddc84', 'cpp': '#9fc2ff'}


def collect_examples(df, per_bin=2, seed=0, include_whisper=True):
    """Sample clips across the intensity range; fall back to train per bin."""
    rng = np.random.default_rng(seed)
    picks = []
    used = set()
    for lo, hi in BINS:
        sub = df[(df['label'] >= lo) & (df['label'] <= hi)]
        if len(sub) == 0:
            continue
        k = min(per_bin, len(sub))
        idx = rng.choice(len(sub), k, replace=False)
        for _, row in sub.iloc[idx].iterrows():
            if row['path'] not in used:
                picks.append(row.to_dict())
                used.add(row['path'])
    if include_whisper and 'whisper' in df.columns:
        wh = df[df['whisper'] == 1]
        for _, row in wh.head(per_bin).iterrows():
            if row['path'] not in used:
                picks.append(row.to_dict())
                used.add(row['path'])
    return picks


def make_dsp_figure(result, title):
    """3-panel dark figure: waveform, DSP profile, phonation cues."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    style = {'facecolor': '#12161f', 'axes': '#12161f',
             'text': '#c9d4e5', 'grid': '#28303f'}
    audio = result['raw']
    sr = 16000
    t = np.arange(len(audio)) / sr
    ft = result['frame_times']
    frames = result['frames']

    fig, axes = plt.subplots(3, 1, figsize=(8.4, 5.6), sharex=True,
                             gridspec_kw={'height_ratios': [1.0, 1.6, 0.8]})
    fig.patch.set_facecolor(style['facecolor'])
    for ax in axes:
        ax.set_facecolor(style['axes'])
        ax.tick_params(colors=style['text'], labelsize=8)
        ax.grid(True, color=style['grid'], linewidth=0.6)
        for spine in ax.spines.values():
            spine.set_color(style['grid'])
        ax.xaxis.label.set_color(style['text'])
        ax.yaxis.label.set_color(style['text'])

    axes[0].plot(t, audio, color='#5b7fb5', linewidth=0.4)
    axes[0].set_ylabel('waveform', fontsize=8)
    axes[0].set_yticks([])
    axes[0].margins(y=0.15)

    axes[1].plot(result['centers'], result['scores'], 'o-',
                 color=COLORS['profile'], linewidth=1.8, markersize=4,
                 label='window score')
    axes[1].axhline(result['clip_score'], color='#eaf1fb', linestyle='--',
                    linewidth=1, label=f"clip {result['clip_score']:.2f}")
    axes[1].axhline(result['whole_score'], color='#7d8ba1', linestyle=':',
                    linewidth=1, label=f"whole {result['whole_score']:.2f}")
    axes[1].set_ylim(1, 9)
    axes[1].set_yticks([1, 3, 5, 7, 9])
    axes[1].set_ylabel('intensity (1-9)', fontsize=8)
    leg = axes[1].legend(loc='upper right', fontsize=7, framealpha=0.25)
    for txt in leg.get_texts():
        txt.set_color(style['text'])
    if result['is_whisper']:
        axes[1].text(0.01, 0.06,
                     f"whisper gate fired (p={result['whisper_prob']:.2f})",
                     transform=axes[1].transAxes, color='#f0d68a', fontsize=8)

    voiced = np.asarray(frames['voiced'], dtype=float)
    cpp = np.asarray(frames['cpp'], dtype=float)
    cpp_z = (cpp - np.nanmean(cpp)) / (np.nanstd(cpp) + 1e-9)
    axes[2].fill_between(ft, voiced, color=COLORS['voiced'], alpha=0.35,
                         step='mid', label='voiced')
    axes[2].plot(ft, np.clip(cpp_z, -3, 3), color=COLORS['cpp'],
                 linewidth=0.8, label='CPP (z)')
    axes[2].set_ylim(-3.2, 3.2)
    axes[2].set_yticks([])
    axes[2].set_ylabel('phonation', fontsize=7)
    axes[2].set_xlabel('time (s)', fontsize=8)
    leg2 = axes[2].legend(loc='upper right', fontsize=6, framealpha=0.2)
    for txt in leg2.get_texts():
        txt.set_color(style['text'])

    axes[0].set_title(title, fontsize=9, color=style['text'], loc='left')
    axes[-1].set_xlim(0, max(t[-1], 0.1))
    fig.tight_layout(h_pad=0.4)
    return fig_data_uri(fig)


def build_html(args, model, cards, summary):
    parts = []
    parts.append('<!doctype html><html lang="en"><head><meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    parts.append('<title>DSP intensity examples</title>')
    parts.append(f'<style>{build_css()}</style></head><body>')
    parts.append('<header><h1>Perceptual intensity &mdash; DSP examples</h1>')
    parts.append('<div class="sub">')
    gate = 'gate' if model.gate is not None else 'no gate'
    parts.append(f'<code>tier {model.tier}</code>&nbsp;{html.escape(model.kind)}'
                 f'&nbsp;&middot;&nbsp;<code>{gate}</code>'
                 f'&nbsp;&middot;&nbsp;window {args.window:g}s / hop {args.hop:g}s'
                 f'&nbsp;&middot;&nbsp;{len(cards)} samples</div>')
    parts.append('</header><main>')

    if summary:
        parts.append('<div class="metrics">')
        for name, value in summary:
            parts.append(f'<div class="metric"><div class="k">{html.escape(name)}</div>'
                         f'<div class="v">{html.escape(value)}</div></div>')
        parts.append('</div>')

    parts.append('<div class="grid">')
    for card in cards:
        parts.append('<div class="card">')
        parts.append(f'<h2>{html.escape(card["name"])}</h2>')
        parts.append(f'<div class="path">{html.escape(card["path"])}'
                     f' &nbsp;&middot;&nbsp; {card["duration"]:.2f}s'
                     f' &nbsp;&middot;&nbsp; {html.escape(card["split"])}</div>')
        parts.append('<div class="badges">')
        if card['label'] is not None:
            parts.append(f'<span class="badge label">label {card["label"]}</span>')
            err = card['pred'] - card['label']
            cls = 'good' if abs(err) < 0.75 else 'bad'
            parts.append(f'<span class="badge {cls}">DSP {card["pred"]:.2f} '
                         f'({err:+.2f})</span>')
            if card.get('in_sample'):
                parts.append('<span class="badge tag">train (in-sample)</span>')
        else:
            parts.append(f'<span class="badge">DSP {card["pred"]:.2f}</span>')
        if card.get('is_whisper'):
            parts.append(f'<span class="badge tag">whisper p={card["whisper_prob"]:.2f} '
                         f'&rarr; {card["pred"]:.1f}</span>')
        parts.append(f'<span class="badge">{html.escape(card["corpus"])}</span>')
        parts.append(f'<span class="badge">{html.escape(card["speaker"])}</span>')
        parts.append('</div>')
        parts.append(f'<audio controls preload="metadata" src="{card["audio"]}"></audio>')
        parts.append(f'<img alt="DSP intensity profile" src="{card["image"]}">')
        parts.append('<div class="legend">'
                     '<span><i style="background:#ff7a45"></i>window score</span>'
                     '<span><i style="background:#eaf1fb"></i>clip mean</span>'
                     '<span><i style="background:#3ddc84;opacity:.5"></i>voiced</span>'
                     '<span><i style="background:#9fc2ff"></i>CPP (z)</span>'
                     '</div>')
        scores = list(card['scores'])
        if len(scores) > 12:
            vals = ' '.join(f'{s:.1f}' for s in scores[:10]) + ' ... ' \
                + ' '.join(f'{s:.1f}' for s in scores[-2:])
        else:
            vals = ' '.join(f'{s:.1f}' for s in scores)
        parts.append(f'<div class="windows">window scores ({len(scores)}): {vals}</div>')
        parts.append('</div>')
    parts.append('</div></main>')

    parts.append(
        '<footer><div class="note"><b>How to read this:</b> each point is the DSP '
        'model applied to one window; the features are aggregated from the frames '
        'inside that window, computed once for the whole clip. The <b>clip</b> line '
        'is the duration-weighted mean, the <b>whole</b> line is one pass over all '
        'frames. The green strip is <b>voicing</b> and the blue line is <b>CPP</b> '
        '(z-scored periodicity): whisper examples should show low voicing, low CPP, '
        'and a flat score at 1. Listen to a 1-2, a 5, and an 8-9 below to calibrate '
        'your ear against the labels. The model compresses the extremes because '
        'labels 8-9 are rare (10 and 1 clips in train), so treat the high-end '
        'numbers as a ranking, not an absolute calibration.'
        '</div></footer></body></html>')
    return ''.join(parts)



def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--features-csv', type=str, required=True)
    parser.add_argument('--dedupe', action='store_true',
                        help='aggregate repeated paths before reporting')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--eval-splits', nargs='+', default=['val'])
    parser.add_argument('--train-splits', nargs='+', default=['train'])
    parser.add_argument('--per-bin', type=int, default=2,
                        help='examples per intensity bin (1-2, 3-4, ..., 9)')
    parser.add_argument('--n', type=int, default=0,
                        help='if >0, sample this many clips instead of per-bin')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--window', type=float, default=2.0)
    parser.add_argument('--hop', type=float, default=0.5)
    parser.add_argument('--max_audio_sec', type=float, default=30.0)
    parser.add_argument('--out', type=str, default='report/dsp_intensity.html')
    args = parser.parse_args()

    all_df = pd.read_csv(args.features_csv)
    all_df = all_df[all_df['path'].notna()].reset_index(drop=True)
    if args.dedupe:
        from dedupe_annotations import dedupe
        all_df = dedupe(all_df, keep_features=True)
    eval_df = filter_splits(all_df, args.eval_splits)
    train_df = filter_splits(all_df, args.train_splits)
    model = DSPIntensityModel.load(args.model)

    # -- summary metrics on the eval split ---------------------------------
    summary = []
    if len(eval_df):
        pred_all = model.predict(eval_df)
        y = eval_df['label'].to_numpy(dtype=float)
        r, mae, r2 = metrics(pred_all, y)
        wr = within_speaker_r(pred_all, y, eval_df['speaker'])
        summary.append(('eval clips', str(len(eval_df))))
        summary.append(('pooled r / MAE', f'{r:+.3f} / {mae:.3f}'))
        summary.append(('within-speaker r', f'{wr:+.3f}'))
        if 'whisper' in eval_df.columns:
            m = eval_df['whisper'].to_numpy() == 1
            if m.any():
                summary.append(('whisper MAE',
                                f'{np.abs(pred_all[m] - y[m]).mean():.3f} '
                                f'(n={int(m.sum())})'))

    # -- sample examples ----------------------------------------------------
    if args.n:
        rng = np.random.default_rng(args.seed)
        n = min(args.n, len(eval_df))
        idx = rng.choice(len(eval_df), n, replace=False)
        examples = [eval_df.iloc[i].to_dict() for i in idx]
        examples = [dict(e, in_sample=False) for e in examples]
    else:
        examples = collect_examples(eval_df, args.per_bin, args.seed)
        examples = [dict(e, in_sample=False) for e in examples]

        # bins empty in eval (usually 8-9): fall back to train, clearly marked
        covered = {int(e['label']) for e in examples}
        for lo, hi in BINS:
            if any(lo <= lab <= hi for lab in covered):
                continue
            sub = train_df[(train_df['label'] >= lo) & (train_df['label'] <= hi)]
            if len(sub) == 0:
                continue
            idx = np.random.default_rng(args.seed).choice(
                len(sub), min(args.per_bin, len(sub)), replace=False)
            for _, row in sub.iloc[idx].iterrows():
                examples.append(dict(row.to_dict(), in_sample=True))
                covered.add(int(row['label']))

    # de-duplicate by path
    seen = set()
    examples = [e for e in examples
                if not (e['path'] in seen or seen.add(e['path']))]

    print(f'{len(examples)} examples, {len(summary)} summary metrics')
    cards = []
    for i, row in enumerate(examples, 1):
        try:
            from dsp_features import load_wave
            audio = load_wave(row['path'])
            result = analyze(audio, model, args.window, args.hop)
            pred = result['clip_score']
            if row.get('label') is not None:
                label = int(row['label'])
            else:
                label = None
            card = {
                'name': os.path.basename(row['path']),
                'path': row['path'],
                'duration': len(audio) / 16000.0,
                'split': row.get('split', '?'),
                'label': label,
                'pred': pred,
                'whisper_prob': result['whisper_prob'],
                'is_whisper': result['is_whisper'],
                'corpus': row.get('corpus', '?'),
                'speaker': row.get('speaker', '?'),
                'in_sample': bool(row.get('in_sample', False)),
                'audio': wav_data_uri(audio, max_sec=args.max_audio_sec),
                'image': make_dsp_figure(result, os.path.basename(row['path'])),
                'scores': result['scores'].tolist(),
            }
            cards.append(card)
            print(f'[{i}/{len(examples)}] {card["name"]}: '
                  f'label {label} pred {pred:.2f} '
                  f'(whisper p={result["whisper_prob"]:.2f})')
        except Exception as exc:
            print(f'[skip] {row["path"]}: {type(exc).__name__}: {exc}')

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(build_html(args, model, cards, summary))
    size_mb = os.path.getsize(args.out) / 1e6
    print(f'\nwrote {args.out} ({size_mb:.1f} MB, self-contained)')


if __name__ == '__main__':
    main()

