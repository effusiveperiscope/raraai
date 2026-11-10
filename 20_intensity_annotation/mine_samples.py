"""Mine the full (mostly unlabeled) filelist for standout samples.

Scores a random sample of clips from a big path|sid filelist with the intensity
model, then picks the most interesting ones by several criteria and writes a
self-contained HTML report (same card layout as make_report.py).

Samples already annotated in data/v0.5 are excluded automatically (matched by
file stem), so everything in the report is unlabeled material.

Usage:
    python mine_samples.py \
        --filelist diverse_full_linux.txt \
        --ckpt checkpoints/intensity_whisper_small_local/best-checkpoint.ckpt \
        --compare checkpoints/test8/last.ckpt \
        --limit 600 --per-category 2 --out report/mined.html
"""
import argparse
import os
import random
import shutil

import librosa
import numpy as np
import torch

from inference import (load_model_and_context, extract_interp_features,
                       score_features, load_audio)
from make_report import make_figure, wav_data_uri, build_html, build_css

DEFAULT_LABELED = [
    'data/v0.5/train_filelist_linux.txt',
    'data/v0.5/val_filelist_linux.txt',
    'data/v0.5/train_filelist.txt',
    'data/v0.5/val_filelist.txt',
]


def read_rows(path, labeled, exclude_stems=True):
    """path|sid rows -> list of (path, sid). Rows may be path|label for labeled sets."""
    stems = set()
    for lp in labeled:
        if not os.path.isfile(lp):
            continue
        with open(lp, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                p = line.rsplit('|', 1)[0].strip()
                stems.add(os.path.splitext(os.path.basename(p))[0])

    rows, skipped = [], 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit('|', 1)
            p = parts[0].strip()
            sid = parts[1].strip() if len(parts) > 1 else ''
            if exclude_stems and os.path.splitext(os.path.basename(p))[0] in stems:
                skipped += 1
                continue
            rows.append((p, sid))
    return rows, skipped


def score_batch(model, context, waves, window_sec, hop_sec, device, batch_size=16):
    """Score a list of waves -> list of dicts with window scores + whole-clip score.

    Windows from *all* clips in the list are flattened into shared forward
    passes, which is what makes mining thousands of clips practical.
    """
    win, hop = int(window_sec * 16000), int(hop_sec * 16000)
    spans, flat, owner = [], [], []
    for ci, w in enumerate(waves):
        starts = [0] if len(w) <= win else list(range(0, len(w) - win + 1, hop))
        spans.append(starts)
        for s in starts:
            flat.append(librosa.util.normalize(w[s:s + win]))
            owner.append(ci)

    window_scores = [[] for _ in waves]
    for i in range(0, len(flat), batch_size):
        chunk = flat[i:i + batch_size]
        feats, mask = extract_interp_features(context, chunk, device=device)
        pooled, _, _ = score_features(model, feats, mask)
        vals = pooled.float().cpu().numpy() * 8 + 1
        for j, ci in enumerate(owner[i:i + batch_size]):
            window_scores[ci].append(float(vals[j]))

    normalized = [librosa.util.normalize(w) for w in waves]
    clip_scores, attn_stats = [], []
    for i in range(0, len(normalized), batch_size):
        chunk = normalized[i:i + batch_size]
        feats, mask = extract_interp_features(context, chunk, device=device)
        pooled, _, attn = score_features(model, feats, mask)
        clip_scores.extend((pooled.float().cpu().numpy() * 8 + 1).tolist())
        for j in range(len(chunk)):
            T = int(mask[j].sum())
            a = attn[j, :T].float().cpu().numpy().astype(np.float64)
            a = a / max(a.sum(), 1e-9)
            attn_stats.append({
                'attn_conc': float(a.max() / (a.mean() + 1e-12)),
                'attn_entropy': float(-(a * np.log(a + 1e-12)).sum() / np.log(len(a) + 1e-12)),
                'attn_peak_t': float(a.argmax()) / 100.0,
            })

    out = []
    for ci in range(len(waves)):
        out.append({'windows': np.array(window_scores[ci]),
                    'clip': float(clip_scores[ci]), **attn_stats[ci]})
    return out


def attention_profile(model, context, wave, device):
    """(stats, curve) for the attention head on one clip."""
    feats, mask = extract_interp_features(
        context, [librosa.util.normalize(np.asarray(wave, dtype=np.float32))], device=device)
    _, _, attn = score_features(model, feats, mask)
    T = int(mask.sum())
    a = attn[0, :T].float().cpu().numpy().astype(np.float64)
    a = a / max(a.sum(), 1e-9)
    stats = {
        'attn_conc': float(a.max() / (a.mean() + 1e-12)),
        'attn_entropy': float(-(a * np.log(a + 1e-12)).sum() / np.log(len(a) + 1e-12)),
        'attn_peak_t': float(a.argmax()) / 100.0,
    }
    return stats, a


def clip_stats(item, sr=16000):
    ws = item['windows']
    w = np.asarray(item['audio'], dtype=np.float32)
    n = len(ws)
    rms_db = float(20.0 * np.log10(np.sqrt((w ** 2).mean()) + 1e-9))
    slope = 0.0
    if n >= 3:
        slope = float(np.polyfit(np.linspace(0, 1, n), ws, 1)[0])
    return {
        'mean': float(ws.mean()),
        'std': float(ws.std()),
        'n_windows': n,
        'duration': len(w) / sr,
        'rms_db': rms_db,
        'slope': slope,
    }


def rank(items, key, reverse, n, need=lambda c: True):
    cand = [c for c in items if need(c)]
    cand.sort(key=key, reverse=reverse)
    return cand[:n]


def select_standouts(items, per_category, has_compare, min_windows=3):
    """Rank clips by each criterion and return the picks, duplicates dropped."""
    med_rms = float(np.median([c['stats']['rms_db'] for c in items]))
    med_std = float(np.median([c['stats']['std'] for c in items]))
    long_enough = lambda c: c['stats']['n_windows'] >= min_windows
    dynamic = lambda c: long_enough(c) and c['stats']['std'] > med_std

    plan = [
        ('highest predicted intensity', lambda c: c['stats']['mean'], True, long_enough),
        ('lowest predicted intensity', lambda c: c['stats']['mean'], False, long_enough),
        ('biggest internal swing', lambda c: c['stats']['std'], True, dynamic),
        ('strongest rise', lambda c: c['stats']['slope'], True, dynamic),
        ('strongest fall', lambda c: c['stats']['slope'], False, dynamic),
        ('quiet but intense', lambda c: c['stats']['mean'], True,
         lambda c: long_enough(c) and c['stats']['rms_db'] < med_rms),
        ('loud but reads low', lambda c: c['stats']['mean'], False,
         lambda c: long_enough(c) and c['stats']['rms_db'] > med_rms),
        ('attention spike', lambda c: c['attn']['attn_conc'], True, long_enough),
    ]
    if has_compare:
        plan.append(('model disagreement', lambda c: abs(c['results'][0]['clip'] -
                                                        c['results'][1]['clip']), True,
                     long_enough))

    picked, seen, out = [], set(), []
    for category, key, reverse, need in plan:
        for cand in rank(items, key, reverse, per_category * 2, need):
            if cand['path'] in seen:
                continue
            seen.add(cand['path'])
            cand.setdefault('tags', []).append(category)
            out.append(cand)
            if len([c for c in out if category in c['tags']]) >= per_category:
                break
    return out


def build_cards(models, picked, args, device):
    primary = models[0]
    cards = []
    for c in picked:
        _, attn_curve = attention_profile(primary['model'], primary['context'],
                                          c['audio'], device)
        curves = []
        for r in c['results']:
            n = len(r['scores'])
            duration = c['stats']['duration']
            if n == 1:
                centers = np.array([min(duration, args.window) / 2])
            else:
                centers = np.arange(n) * args.hop + min(args.window, duration) / 2
            curves.append((r['label'], centers, np.array(r['scores']), r['color']))
        image = make_figure(c['audio'], curves, attn_curve, primary['color'])
        s = c['stats']
        tags = list(c['tags'])
        tags.append(f'{s["duration"]:.1f}s · {s["n_windows"]} win · win-mean {s["mean"]:.2f} · '
                    f'std {s["std"]:.2f} · slope {s["slope"]:+.2f} · {s["rms_db"]:.0f} dBFS')
        cards.append({
            'name': os.path.basename(c['path']),
            'path': f'{c["path"]}  [sid {c["sid"]}]',
            'duration': s['duration'],
            'label': None,
            'tags': tags,
            'results': c['results'],
            'audio': wav_data_uri(c['audio'], max_sec=args.max_sec),
            'image': image,
        })
    return cards


def overview_figure(items, picked):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from make_report import fig_data_uri

    style = {'facecolor': '#12161f', 'axes': '#12161f', 'text': '#c9d4e5', 'grid': '#28303f'}
    means = [c['stats']['mean'] for c in items]
    stds = [c['stats']['std'] for c in items]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.4))
    fig.patch.set_facecolor(style['facecolor'])
    for ax in axes:
        ax.set_facecolor(style['axes'])
        ax.tick_params(colors=style['text'], labelsize=8)
        ax.grid(True, color=style['grid'], linewidth=0.6)
        for s in ax.spines.values():
            s.set_color(style['grid'])
        ax.xaxis.label.set_color(style['text'])
        ax.yaxis.label.set_color(style['text'])
        ax.title.set_color(style['text'])

    axes[0].hist(means, bins=40, color='#3d5a8a', edgecolor='none')
    axes[0].set_xlabel('predicted intensity (mean of windows)', fontsize=8)
    axes[0].set_ylabel('clips', fontsize=8)
    axes[0].set_title('prediction distribution (unlabeled pool)', fontsize=9)
    for c in picked:
        axes[0].axvline(c['stats']['mean'], color='#ff7a45', alpha=0.35, linewidth=0.8)

    axes[1].scatter(means, stds, s=6, color='#3d5a8a', alpha=0.55, linewidths=0)
    axes[1].scatter([c['stats']['mean'] for c in picked],
                    [c['stats']['std'] for c in picked],
                    s=42, facecolors='none', edgecolors='#ff7a45', linewidths=1.4)
    axes[1].set_xlabel('predicted intensity', fontsize=8)
    axes[1].set_ylabel('within-clip std of window scores', fontsize=8)
    axes[1].set_title('profile contrast vs level (orange = picked)', fontsize=9)

    fig.tight_layout()
    return fig_data_uri(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--filelist', type=str, default='diverse_full_linux.txt',
                        help='big path|sid filelist to mine')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--compare', type=str, default=None)
    parser.add_argument('--config', type=str, default='config/base.yaml')
    parser.add_argument('--labeled', type=str, nargs='*', default=DEFAULT_LABELED,
                        help='filelists whose clips are excluded (already annotated)')
    parser.add_argument('--limit', type=int, default=600, help='clips to score')
    parser.add_argument('--per-category', type=int, default=2)
    parser.add_argument('--window', type=float, default=2.0)
    parser.add_argument('--hop', type=float, default=0.5)
    parser.add_argument('--min-sec', type=float, default=2.0)
    parser.add_argument('--min-windows', type=int, default=3,
                        help='picked clips must have at least this many windows')
    parser.add_argument('--max-sec', type=float, default=30.0)
    parser.add_argument('--workers', type=int, default=8, help='parallel audio loads')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default='report/mined.html')
    args = parser.parse_args()

    args.title = 'Unlabeled pool &mdash; standout samples'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows, skipped = read_rows(args.filelist, args.labeled)
    print(f'{args.filelist}: {len(rows) + skipped} rows, {skipped} already annotated '
          f'-> {len(rows)} unlabeled candidates')
    random.seed(args.seed)
    random.shuffle(rows)
    sample = rows[:args.limit]

    from concurrent.futures import ThreadPoolExecutor

    def safe_load(path):
        try:
            audio, _ = load_audio(path)
        except Exception:
            return None
        if audio is None or len(audio) < int(args.min_sec * 16000):
            return None
        return np.asarray(audio, dtype=np.float32).squeeze()[:int(args.max_sec * 16000)]

    print(f'loading {len(sample)} clips with {args.workers} workers...')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        waves = list(pool.map(safe_load, [p for p, _ in sample]))
    items = [{'path': p, 'sid': sid, 'audio': w}
             for (p, sid), w in zip(sample, waves) if w is not None]
    print(f'{len(items)} clips loaded')

    specs = [('A', args.ckpt, '#ff7a45')]
    if args.compare:
        specs.append(('B', args.compare, '#4ea1ff'))
    models = []
    for tag, path, color in specs:
        model, context, _ = load_model_and_context(path, args.config, device=device)
        whisper = str(getattr(context, 'model').config._name_or_path)
        models.append({'tag': tag, 'model': model, 'context': context, 'color': color,
                       'label': f'model {tag}',
                       'detail': f'{os.path.basename(whisper)} · '
                                 f'{os.path.basename(os.path.dirname(path))}'})

    print('scoring...')
    step = 16
    for i in range(0, len(items), step):
        chunk = items[i:i + step]
        waves = [c['audio'] for c in chunk]
        for m in models:
            scored = score_batch(m['model'], m['context'], waves, args.window,
                                 args.hop, device)
            for c, s in zip(chunk, scored):
                c.setdefault('results', []).append(
                    {'label': m['label'], 'color': m['color'], 'clip': s['clip'],
                     'scores': s['windows'], 'attn_conc': s['attn_conc'],
                     'attn_entropy': s['attn_entropy'], 'attn_peak_t': s['attn_peak_t']})
        for c in chunk:
            c['windows'] = c['results'][0]['scores']
            c['stats'] = clip_stats(c)
            c['attn'] = {k: c['results'][0][k]
                         for k in ('attn_conc', 'attn_entropy', 'attn_peak_t')}
        print(f'\r  {min(i + step, len(items))}/{len(items)}', end='', flush=True)
    print()

    if not items:
        raise SystemExit('nothing scored - check --min-sec/--max-sec and the filelist')

    picked = select_standouts(items, args.per_category, has_compare=args.compare,
                              min_windows=args.min_windows)
    print(f'selected {len(picked)} standout clips')
    cards = build_cards(models, picked, args, device)

    means = [c['stats']['mean'] for c in items]
    summary = [
        ('unlabeled candidates', f'{len(rows):,}'),
        ('already annotated (excluded)', f'{skipped:,}'),
        ('clips scored', f'{len(items):,}'),
        ('standouts shown', str(len(cards))),
        ('pool pred mean / std', f'{np.mean(means):.2f} / {np.std(means):.2f}'),
        ('pool within-clip std',
         f'{np.median([c["stats"]["std"] for c in items]):.2f} (median)'),
    ]
    overview = overview_figure(items, picked)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    page = build_html(args, models, cards, summary)
    page = page.replace('<div class="grid">',
                        f'<img class="overview" src="{overview}"><div class="grid">', 1)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(page)
    print(f'wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)')

    by_cat = {}
    for c in picked:
        for t in c['tags']:
            by_cat.setdefault(t, []).append(c)
    for cat, cs in by_cat.items():
        print(f'  {cat}: ' + ', '.join(
            f'{os.path.basename(c["path"])[:36]} ({c["results"][0]["clip"]:.2f})' for c in cs))


if __name__ == '__main__':
    main()

