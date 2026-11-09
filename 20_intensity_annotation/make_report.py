"""Generate a self-contained HTML report of eval samples + model output.

One card per sample: audio player, waveform, sliding-window intensity profile
(one line per model), and the model's internal per-frame head + attention strip.
Audio and plots are embedded as base64 data URIs, so the output is a single
portable .html file - no server, no assets.

Usage:
    python make_report.py \
        --ckpt checkpoints/intensity_whisper_small_local/best-checkpoint.ckpt \
        --compare checkpoints/test8/last.ckpt \
        --audio-dir eval_sets/eval1 \
        --filelist data/v0.5/val_filelist_linux.txt --n 10 \
        --out report/index.html
"""
import argparse
import base64
import glob
import html
import io
import os

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from inference import (load_model_and_context, extract_interp_features,
                       score_features, load_audio)
from eval_windows import score_windows
from eval_val import metrics, read_filelist

AUDIO_EXTS = ('.wav', '.flac', '.mp3', '.ogg', '.m4a')


def wav_data_uri(wave, sr=16000, max_sec=30.0):
    import soundfile as sf
    wave = np.asarray(wave, dtype=np.float32).squeeze()
    wave = wave[:int(max_sec * sr)]
    buf = io.BytesIO()
    sf.write(buf, wave, sr, format='WAV', subtype='PCM_16')
    return 'data:audio/wav;base64,' + base64.b64encode(buf.getvalue()).decode()


def fig_data_uri(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=105, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def make_figure(audio, curves, attn, primary_color, sr=16000):
    """curves: list of (label, centers, scores, color). attn: (T,) or None."""
    style = {
        'facecolor': '#12161f', 'axes': '#12161f',
        'text': '#c9d4e5', 'grid': '#28303f',
    }
    n_rows = 3 if attn is not None else 2
    ratios = [1.0, 1.7, 0.45][:n_rows]
    fig, axes = plt.subplots(n_rows, 1, figsize=(8.4, 5.0), sharex=True,
                             gridspec_kw={'height_ratios': ratios})
    fig.patch.set_facecolor(style['facecolor'])
    for ax in axes:
        ax.set_facecolor(style['axes'])
        ax.tick_params(colors=style['text'], labelsize=8)
        ax.grid(True, color=style['grid'], linewidth=0.6)
        for spine in ax.spines.values():
            spine.set_color(style['grid'])
        ax.xaxis.label.set_color(style['text'])
        ax.yaxis.label.set_color(style['text'])

    t = np.arange(len(audio)) / sr
    axes[0].plot(t, audio, color='#5b7fb5', linewidth=0.4)
    axes[0].set_ylabel('waveform', fontsize=8)
    axes[0].set_xlim(0, max(t[-1], 0.1))
    axes[0].margins(y=0.15)
    axes[0].set_yticks([])

    for label, centers, scores, color in curves:
        axes[1].plot(centers, scores, 'o-', color=color, linewidth=1.8,
                     markersize=4, label=label)
    axes[1].set_ylim(1, 9)
    axes[1].set_yticks([1, 3, 5, 7, 9])
    axes[1].set_ylabel('intensity (1-9)', fontsize=8)
    axes[1].set_xlim(0, max(t[-1], 0.1))
    leg = axes[1].legend(loc='upper right', fontsize=7, framealpha=0.25)
    for txt in leg.get_texts():
        txt.set_color(style['text'])

    if attn is not None:
        ft = np.arange(len(attn)) / 100.0
        axes[2].fill_between(ft, attn, color='#3ddc84', alpha=0.55, step='mid')
        axes[2].set_ylim(0, max(0.005, float(attn.max()) * 1.05))
        axes[2].set_yticks([])
        axes[2].set_ylabel('attention', fontsize=7, color='#3ddc84')

    axes[-1].set_xlabel('time (s)', fontsize=8)
    fig.tight_layout(h_pad=0.4)
    return fig_data_uri(fig)


def collect_samples(args):
    samples = []
    if args.audio_dir:
        for ext in AUDIO_EXTS:
            samples.extend(sorted(glob.glob(os.path.join(args.audio_dir, f'*{ext}'))))
        samples = [(p, None) for p in samples]
    if args.filelist:
        items = read_filelist(args.filelist)
        if args.n:
            items = items[:args.n]
        samples.extend(items)
    return samples


def evaluate(models, audio, window, hop, device):
    """models: list of dicts with model/context/label/color."""
    import librosa
    normalized = librosa.util.normalize(np.asarray(audio, dtype=np.float32).squeeze())
    out = []
    for m in models:
        times, scores = score_windows(m['model'], m['context'], audio, window, hop, device)
        centers = np.array([(a + b) / 2 for a, b in times])
        feats, mask = extract_interp_features(m['context'], [normalized], device=device)
        pooled, frame, attn = score_features(m['model'], feats, mask)
        T = int(mask.sum())
        out.append({
            'label': m['label'],
            'color': m['color'],
            'centers': centers,
            'scores': np.array(scores),
            'clip': float(pooled.item()) * 8 + 1,
            'frame': frame[0, :T].float().cpu().numpy() * 8 + 1,
            'attn': attn[0, :T].float().cpu().numpy(),
        })
    return out


def build_css():
    return """
    :root { color-scheme: dark; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #0b0e14; color: #c9d4e5;
           font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
    header { padding: 28px 32px 18px; border-bottom: 1px solid #1e2634;
             background: linear-gradient(180deg, #131a26, #0b0e14); }
    h1 { margin: 0 0 6px; font-size: 21px; color: #eaf1fb; letter-spacing: .2px; }
    .sub { color: #7d8ba1; font-size: 12.5px; }
    .sub code { color: #9fc2ff; background: #131b28; padding: 1px 5px; border-radius: 4px; }
    main { padding: 22px 32px 60px; }
    .metrics { display: flex; flex-wrap: wrap; gap: 14px; margin: 16px 0 26px; }
    .metric { background: #12161f; border: 1px solid #1e2634; border-radius: 10px;
              padding: 12px 16px; min-width: 172px; }
    .metric .k { color: #7d8ba1; font-size: 11px; text-transform: uppercase; letter-spacing: .6px; }
    .metric .v { font-size: 19px; color: #eaf1fb; font-variant-numeric: tabular-nums; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(430px, 1fr)); gap: 18px; }
    .card { background: #10141c; border: 1px solid #1e2634; border-radius: 12px;
            padding: 14px 16px 16px; overflow: hidden; }
    .card h2 { font-size: 13px; margin: 0 0 2px; color: #dbe6f6; font-weight: 600;
               white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .path { color: #5e6b80; font-size: 10.5px; margin-bottom: 10px;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .badges { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 10px; }
    .badge { font-size: 11.5px; padding: 3px 9px; border-radius: 999px; border: 1px solid #263048;
             background: #151b26; color: #a9b8cf; font-variant-numeric: tabular-nums; }
    .badge.label { border-color: #6b5a1f; background: #241f0f; color: #f0d68a; }
    .badge.good { border-color: #1d5c3a; background: #0f2419; color: #7ee2a8; }
    .badge.bad { border-color: #6b2531; background: #240f13; color: #f39caa; }
    audio { width: 100%; height: 30px; margin: 4px 0 10px; }
    .card img { width: 100%; border-radius: 8px; display: block; border: 1px solid #1a2130; }
    .legend { display: flex; gap: 14px; margin-top: 9px; font-size: 11.5px; color: #8b99ad; }
    .legend i { width: 22px; height: 3px; border-radius: 2px; display: inline-block;
                vertical-align: middle; margin-right: 6px; }
    .windows { margin-top: 9px; font-size: 11px; color: #6f7d92; font-variant-numeric: tabular-nums; }
    footer { padding: 0 32px 40px; color: #5e6b80; font-size: 11.5px; max-width: 900px; }
    footer b { color: #93a3ba; }
    .note { background: #12161f; border-left: 3px solid #3b7bd4; padding: 10px 14px;
            border-radius: 6px; margin-top: 14px; }
    """


def build_html(args, models, cards, summary):
    parts = []
    parts.append('<!doctype html><html lang="en"><head><meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    parts.append('<title>intensity eval samples</title>')
    parts.append(f'<style>{build_css()}</style></head><body>')
    parts.append('<header><h1>Perceptual intensity &mdash; eval samples</h1>')
    parts.append('<div class="sub">')
    parts.append(' &nbsp;|&nbsp; '.join(
        f'<code>{html.escape(m["label"])}</code>&nbsp;{html.escape(m["detail"])}'
        for m in models))
    parts.append(f'<br>windows {args.window:g}s / hop {args.hop:g}s &nbsp;&middot;&nbsp; '
                 f'{len(cards)} samples</div>')
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
                     f' &nbsp;&middot;&nbsp; {card["duration"]:.2f}s</div>')
        parts.append('<div class="badges">')
        if card['label'] is not None:
            parts.append(f'<span class="badge label">label {card["label"]}</span>')
            for r in card['results']:
                err = r['clip'] - card['label']
                cls = 'good' if abs(err) < 0.75 else 'bad'
                parts.append(f'<span class="badge {cls}">{html.escape(r["label"])} '
                             f'pred {r["clip"]:.2f} ({err:+.2f})</span>')
        else:
            for r in card['results']:
                parts.append(f'<span class="badge">{html.escape(r["label"])} '
                             f'pred {r["clip"]:.2f}</span>')
        parts.append('</div>')
        parts.append(f'<audio controls preload="metadata" src="{card["audio"]}"></audio>')
        parts.append(f'<img alt="intensity profile" src="{card["image"]}">')
        parts.append('<div class="legend">')
        for r in card['results']:
            parts.append(f'<span><i style="background:{r["color"]}"></i>'
                         f'{html.escape(r["label"])}</span>')
        parts.append('<span><i style="background:#3ddc84;opacity:.5"></i>attention</span>')
        parts.append('</div>')
        primary = card['results'][0]
        scores = list(primary['scores'])
        if len(scores) > 12:
            vals = ' '.join(f'{s:.1f}' for s in scores[:10]) + ' ... ' \
                + ' '.join(f'{s:.1f}' for s in scores[-2:])
        else:
            vals = ' '.join(f'{s:.1f}' for s in scores)
        parts.append(f'<div class="windows">window scores ({html.escape(primary["label"])}, '
                     f'{len(scores)}): {vals}</div>')
        parts.append('</div>')
    parts.append('</div></main>')

    parts.append(
        '<footer><div class="note"><b>How to read this:</b> the line with dots is the '
        '<b>sliding-window score</b> &mdash; each point is a separate forward pass on one '
        'window, the granularity the model is trained and calibrated for. That is the '
        'trustworthy output. The green shaded strip is the model&rsquo;s internal attention, '
        'which is <b>not</b> a contribution map: occlusion testing shows that silencing the '
        'most-attended 25% of a clip changes the prediction no more than silencing a random '
        '25% (see <code>attribution_test.py</code>).</div></footer></body></html>')
    return ''.join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ckpt', type=str, required=True, help='primary checkpoint')
    parser.add_argument('--compare', type=str, default=None, help='second checkpoint')
    parser.add_argument('--config', type=str, default='config/base.yaml')
    parser.add_argument('--audio-dir', type=str, default=None,
                        help='directory of audio files to include (no labels)')
    parser.add_argument('--filelist', type=str, default=None,
                        help='path|label filelist (labels shown + metrics)')
    parser.add_argument('--n', type=int, default=10, help='max clips taken from --filelist')
    parser.add_argument('--window', type=float, default=2.0)
    parser.add_argument('--hop', type=float, default=0.5)
    parser.add_argument('--max_audio_sec', type=float, default=30.0,
                        help='cap on embedded audio length')
    parser.add_argument('--out', type=str, default='report/index.html')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    colors = ['#ff7a45', '#4ea1ff']

    specs = [('A', args.ckpt, colors[0])]
    if args.compare:
        specs.append(('B', args.compare, colors[1]))
    models = []
    for tag, path, color in specs:
        model, context, _ = load_model_and_context(path, args.config, device=device)
        whisper = str(getattr(context, 'model').config._name_or_path)
        detail = f'{os.path.basename(whisper)} · {os.path.basename(os.path.dirname(path))}'
        models.append({'tag': tag, 'model': model, 'context': context, 'color': color,
                       'label': f'model {tag}', 'detail': detail})

    samples = collect_samples(args)
    if not samples:
        raise SystemExit('nothing to evaluate: pass --audio-dir and/or --filelist')
    print(f'{len(models)} model(s), {len(samples)} samples')

    cards = []
    for i, (path, label) in enumerate(samples, 1):
        audio, _ = load_audio(path)
        results = evaluate(models, audio, args.window, args.hop, device)
        image = make_figure(audio, [(r['label'], r['centers'], r['scores'], r['color'])
                                    for r in results],
                            results[0]['attn'], models[0]['color'])
        cards.append({
            'name': os.path.basename(path),
            'path': path,
            'duration': len(audio) / 16000.0,
            'label': label,
            'results': results,
            'audio': wav_data_uri(audio, max_sec=args.max_audio_sec),
            'image': image,
        })
        scores = ', '.join(f'{r["label"]} {r["clip"]:.2f}' for r in results)
        print(f'[{i}/{len(samples)}] {os.path.basename(path)}: {scores}'
              + (f' (label {label})' if label is not None else ''))

    # summary metrics over the labeled subset (window mean vs label)
    summary = []
    labeled = [c for c in cards if c['label'] is not None]
    if labeled:
        y = np.array([c['label'] for c in labeled], dtype=float)
        summary.append(('labeled samples', str(len(labeled))))
        for j, r0 in enumerate(labeled[0]['results']):
            name = r0['label']
            clip_pred = [c['results'][j]['clip'] for c in labeled]
            win_pred = [float(c['results'][j]['scores'].mean()) for c in labeled]
            r_c, mae_c, _ = metrics(clip_pred, y)
            r_w, mae_w, _ = metrics(win_pred, y)
            summary.append((f'{name} window mean  r / MAE', f'{r_w:+.3f} / {mae_w:.3f}'))
            summary.append((f'{name} whole clip   r / MAE', f'{r_c:+.3f} / {mae_c:.3f}'))

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(build_html(args, models, cards, summary))
    size_mb = os.path.getsize(args.out) / 1e6
    print(f'\nwrote {args.out} ({size_mb:.1f} MB, self-contained)')


if __name__ == '__main__':
    main()

