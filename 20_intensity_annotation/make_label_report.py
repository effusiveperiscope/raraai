"""Interactive, self-contained re-labeling report.

One card per clip: embedded audio, optional DSP profile, the current/competing
labels, the mining scores, and a 1-9 button row.  Selections are kept in the
browser's localStorage keyed by path; "Export labels CSV" downloads
`path,new_label,note,...` so nothing is uploaded and no server is required.

Usage:
    # conflicting / duplicate labels (worst first)
    python make_label_report.py --csv report/label_conflicts.csv \
        --title "Conflicting labels" --min-span 2 \
        --out report/label_conflicts.html

    # high-intensity candidates
    python make_label_report.py --csv report/high_intensity_candidates.csv \
        --title "High-intensity candidates" \
        --out report/high_intensity_candidates.html
"""
import argparse
import ast
import base64
import html
import io
import json
import os
import re

import numpy as np
import pandas as pd

from dsp_features import load_wave
from make_report import build_css

REASON = {
    'A_low_clf7': '7+ model / low label',
    'C_borderline5': 'borderline 5',
    'B_low_clf6': '6+ model / low label',
    'A_clf7': '7+ model',
    'B_clf6': '6+ model',
    'C_top': 'top score',
}


def audio_data_uri(wave, sr=16000, max_sec=15.0, fmt='mp3'):
    """Compact embedded audio (MP3 by default; ~5x smaller than PCM WAV)."""
    import soundfile as sf
    wave = np.asarray(wave, dtype=np.float32).squeeze()
    wave = wave[:int(max_sec * sr)]
    buf = io.BytesIO()
    if fmt == 'mp3':
        sf.write(buf, wave, sr, format='MP3')
        mime = 'audio/mpeg'
    elif fmt == 'flac':
        sf.write(buf, wave, sr, format='FLAC', subtype='PCM_16')
        mime = 'audio/flac'
    else:
        sf.write(buf, wave, sr, format='WAV', subtype='PCM_16')
        mime = 'audio/wav'
    return f'data:{mime};base64,' + base64.b64encode(buf.getvalue()).decode()


def _parse_labels(row):
    value = row.get('labels')
    if isinstance(value, str) and value.strip().startswith('['):
        try:
            return [int(v) for v in ast.literal_eval(value)]
        except (ValueError, SyntaxError):
            pass
    label = row.get('label', row.get('current_label'))
    if label is not None and not pd.isna(label):
        return [int(label)]
    return []


def _fmt(value, digits=2):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return f'{value:.{digits}f}'


def load_rows(csv_path, min_span=0, limit=0):
    df = pd.read_csv(csv_path)
    if min_span and 'label_span' in df.columns:
        df = df[df['label_span'] >= min_span]
    # conflict tables: worst first; candidate tables are already ranked
    if 'label_span' in df.columns and 'reason' not in df.columns:
        df = df.sort_values(['label_span', 'n_rows'], ascending=False)
    if limit:
        df = df.head(limit)
    return df.reset_index(drop=True)


def collect_cards(df, model_path, window, hop, max_audio_sec, with_profile,
                  audio_format='mp3'):
    model = None
    if with_profile and model_path:
        from dsp_intensity import DSPIntensityModel
        from dsp_profile import analyze
        from make_dsp_report import make_dsp_figure
        model = DSPIntensityModel.load(model_path)
    else:
        analyze = None
        make_dsp_figure = None

    cards = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        path = row['path']
        try:
            audio = load_wave(path)
        except Exception as exc:
            print(f'[skip] {path}: {type(exc).__name__}: {exc}')
            continue

        labels = _parse_labels(row)
        suggested = row.get('label', row.get('current_label'))
        suggested = int(suggested) if suggested is not None and not pd.isna(suggested) else None
        if suggested is None and labels:
            suggested = int(round(float(np.median(labels))))

        image = ''
        if model is not None:
            try:
                result = analyze(audio, model, window, hop)
                image = make_dsp_figure(result, os.path.basename(path))
            except Exception as exc:
                print(f'[profile-skip] {path}: {exc}')

        card = {
            'path': path,
            'name': os.path.basename(path),
            'duration': len(audio) / 16000.0,
            'current_labels': labels,
            'suggested': suggested,
            'clf6': _fmt(row.get('clf6')),
            'clf7': _fmt(row.get('clf7')),
            'dsp_pred': _fmt(row.get('dsp_pred')),
            'rms': _fmt(row.get('rms_dbfs'), 1),
            'span': int(row['label_span']) if 'label_span' in row and not pd.isna(row['label_span']) else None,
            'splits': str(row.get('splits', '')),
            'leak': int(row['leak']) if 'leak' in row and not pd.isna(row['leak']) else 0,
            'n_rows': int(row['n_rows']) if 'n_rows' in row and not pd.isna(row['n_rows']) else 1,
            'corpus': str(row.get('corpus', '')),
            'speaker': str(row.get('speaker', '')),
            'reason': str(row.get('reason', '')),
            'audio': audio_data_uri(audio, max_sec=max_audio_sec,
                                    fmt=audio_format),
            'image': image,
        }
        cards.append(card)
        print(f'[{i}/{len(df)}] {card["name"]}: labels {labels} '
              f'clf6={card["clf6"]} clf7={card["clf7"]}')
    return cards


def build_html(cards, title, csv_name, report_key):
    extra_css = """
    .toolbar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
               margin: 0 0 18px; }
    .btn { background: #1d3b63; border: 1px solid #4ea1ff; color: #eaf1fb;
           padding: 7px 12px; border-radius: 7px; cursor: pointer; font-size: 12px; }
    .btn:hover { background: #24486f; }
    .btn.ghost { background: #151b26; border-color: #263048; color: #a9b8cf; }
    .progress { color: #7d8ba1; font-size: 12px; font-variant-numeric: tabular-nums; }
    .hint { color: #7d8ba1; font-size: 11.5px; }
    .labrow { display: flex; gap: 4px; flex-wrap: wrap; margin-top: 10px;
              align-items: center; }
    .lab { min-width: 30px; height: 28px; border: 1px solid #263048;
           background: #151b26; color: #a9b8cf; border-radius: 6px;
           cursor: pointer; font-variant-numeric: tabular-nums; font-size: 12px; }
    .lab:hover { border-color: #4ea1ff; }
    .lab.active { background: #1d3b63; border-color: #4ea1ff; color: #eaf1fb;
                  font-weight: 700; }
    .lab.current { outline: 1px dashed #6b5a1f; outline-offset: 1px; }
    .lab.clear { padding: 0 8px; color: #7d8ba1; }
    .notes { width: 100%; margin-top: 7px; background: #0f131a;
             border: 1px solid #263048; color: #c9d4e5; border-radius: 6px;
             padding: 5px 8px; font-size: 11.5px; }
    .notes:focus { outline: 1px solid #3b7bd4; }
    """

    parts = []
    parts.append('<!doctype html><html lang="en"><head><meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    parts.append(f'<title>{html.escape(title)}</title>')
    parts.append(f'<style>{build_css()}{extra_css}</style></head><body>')
    parts.append(f'<header><h1>{html.escape(title)}</h1>')
    parts.append('<div class="sub">')
    parts.append(f'<code>{html.escape(csv_name)}</code>'
                 f' &nbsp;&middot;&nbsp; {len(cards)} clips'
                 ' &nbsp;&middot;&nbsp; click 1-9 on each card')
    parts.append('</div></header><main>')

    parts.append('<div class="toolbar">')
    parts.append('<button class="btn" id="export">Export labels CSV</button>')
    parts.append('<button class="btn ghost" id="clear">Clear all</button>')
    parts.append(f'<span class="progress" id="progress">0 / {len(cards)} labeled</span>')
    parts.append('<span class="hint">Selections are stored in this browser '
                 '(localStorage). Export when done; nothing is uploaded.</span>')
    parts.append('</div>')

    parts.append('<div class="grid">')
    for i, c in enumerate(cards):
        label_txt = ', '.join(str(v) for v in c['current_labels']) or '?'
        parts.append(f'<div class="card" data-idx="{i}">')
        parts.append(f'<h2>{html.escape(c["name"])}</h2>')
        parts.append(f'<div class="path">{html.escape(c["path"])}'
                     f' &nbsp;&middot;&nbsp; {c["duration"]:.2f}s</div>')
        parts.append('<div class="badges">')
        parts.append(f'<span class="badge label">current {html.escape(label_txt)}</span>')
        if c['span'] is not None and c['span'] > 0:
            parts.append(f'<span class="badge bad">span {c["span"]}</span>')
        if c['suggested'] is not None:
            parts.append(f'<span class="badge">suggested {c["suggested"]}</span>')
        if c['clf6'] is not None:
            parts.append(f'<span class="badge tag">clf6 {c["clf6"]}</span>')
        if c['clf7'] is not None:
            parts.append(f'<span class="badge tag">clf7 {c["clf7"]}</span>')
        if c['dsp_pred'] is not None:
            parts.append(f'<span class="badge">DSP {c["dsp_pred"]}</span>')
        if c['rms'] is not None:
            parts.append(f'<span class="badge">rms {c["rms"]}</span>')
        if c['reason']:
            parts.append(f'<span class="badge tag">'
                         f'{html.escape(REASON.get(c["reason"], c["reason"]))}</span>')
        if c['leak']:
            parts.append('<span class="badge bad">train+val</span>')
        if c['n_rows'] > 1:
            parts.append(f'<span class="badge">{c["n_rows"]} rows</span>')
        parts.append(f'<span class="badge">{html.escape(c["corpus"])}</span>')
        parts.append(f'<span class="badge">{html.escape(c["speaker"])}</span>')
        parts.append('</div>')

        parts.append(f'<audio controls preload="metadata" src="{c["audio"]}"></audio>')
        if c['image']:
            parts.append(f'<img alt="DSP profile" src="{c["image"]}">')

        parts.append('<div class="labrow">')
        for lab in range(1, 10):
            parts.append(f'<button class="lab" data-idx="{i}" data-label="{lab}"'
                         f' title="label {lab}">{lab}</button>')
        parts.append(f'<button class="lab clear" data-idx="{i}" data-label="0"'
                     ' title="clear">clear</button>')
        parts.append('</div>')
        parts.append(f'<input class="notes" data-idx="{i}"'
                     ' placeholder="note (optional)">')
        parts.append('</div>')
    parts.append('</div></main>')



    rows_js = [{
        'path': c['path'],
        'current_labels': c['current_labels'],
        'suggested': c['suggested'],
        'corpus': c['corpus'],
        'speaker': c['speaker'],
        'reason': c['reason'],
    } for c in cards]
    rows_json = json.dumps(rows_js).replace('</', '<\\/')

    parts.append('<script>')
    parts.append(f'const ROWS = {rows_json};')
    parts.append(f"const KEY = 'intensity_relabel_v1:{report_key}';")
    parts.append("""
let STATE = {};
try { STATE = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { STATE = {}; }
function save() { try { localStorage.setItem(KEY, JSON.stringify(STATE)); } catch (e) {} }
function refresh() {
  let n = 0;
  ROWS.forEach((row, i) => {
    const card = document.querySelector('.card[data-idx="' + i + '"]');
    if (!card) return;
    const cur = STATE[row.path] || {};
    card.querySelectorAll('button.lab').forEach(b => {
      const lab = parseInt(b.dataset.label, 10);
      b.classList.toggle('active', lab !== 0 && cur.label === lab);
      b.classList.toggle('current', lab !== 0 && row.current_labels.includes(lab));
    });
    const note = card.querySelector('.notes');
    if (note && document.activeElement !== note) note.value = cur.note || '';
    if (cur.label) n++;
  });
  document.getElementById('progress').textContent =
    n + ' / ' + ROWS.length + ' labeled';
}
function csvCell(v) { return '"' + String(v).replace(/"/g, '""') + '"'; }
document.addEventListener('click', e => {
  const b = e.target.closest('button.lab');
  if (!b) return;
  const i = parseInt(b.dataset.idx, 10);
  const lab = parseInt(b.dataset.label, 10);
  const path = ROWS[i].path;
  const cur = STATE[path] || {};
  if (lab === 0) delete cur.label; else cur.label = lab;
  STATE[path] = cur;
  save(); refresh();
});
document.addEventListener('input', e => {
  const t = e.target.closest('input.notes');
  if (!t) return;
  const i = parseInt(t.dataset.idx, 10);
  const path = ROWS[i].path;
  const cur = STATE[path] || {};
  cur.note = t.value;
  STATE[path] = cur;
  save();
});
document.getElementById('export').addEventListener('click', () => {
  const header = ['path','new_label','note','current_labels',
                  'suggested','corpus','speaker','reason'];
  const lines = [header.map(csvCell).join(',')];
  ROWS.forEach(row => {
    const cur = STATE[row.path] || {};
    const vals = [row.path, cur.label || '', cur.note || '',
                  row.current_labels.join(';'), row.suggested || '',
                  row.corpus || '', row.speaker || '', row.reason || ''];
    lines.push(vals.map(csvCell).join(','));
  });
  const blob = new Blob([lines.join(String.fromCharCode(10))], {type: 'text/csv;charset=utf-8'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'relabeled_REPORT_KEY.csv';
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});
document.getElementById('clear').addEventListener('click', () => {
  if (!confirm('Clear all labels stored in this browser for this report?')) return;
  STATE = {}; save(); refresh();
});
refresh();
""".replace('REPORT_KEY', report_key))
    parts.append('</script>')

    parts.append(
        '<footer><div class="note"><b>How this works:</b> click <b>1-9</b> on each '
        'card. The dashed outline marks the existing label(s); <b>suggested</b> is '
        'the median of repeated annotations. Selections are saved in this browser '
        '(localStorage) keyed by path. <b>Export labels CSV</b> downloads '
        '<code>path,new_label,note,...</code>; rows you did not label are included '
        'with a blank <code>new_label</code> so skipped clips are visible. Nothing '
        'is uploaded.</div></footer></body></html>')
    return ''.join(parts)



def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--title', type=str, default=None)
    parser.add_argument('--model', type=str, default=None,
                        help='DSP model for profile images (auto-detected)')
    parser.add_argument('--no-profile', action='store_true')
    parser.add_argument('--min-span', type=int, default=0,
                        help='only rows with label_span >= this')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--window', type=float, default=2.0)
    parser.add_argument('--hop', type=float, default=0.5)
    parser.add_argument('--max-audio-sec', type=float, default=30.0)
    parser.add_argument('--audio-format', type=str, default='mp3',
                        choices=['mp3', 'flac', 'wav'])
    args = parser.parse_args()

    df = load_rows(args.csv, args.min_span, args.limit)
    if not len(df):
        raise SystemExit('no rows after filtering')

    title = args.title or os.path.basename(args.csv)
    model_path = args.model
    if model_path is None and os.path.exists('checkpoints/dsp_intensity/tier1.pkl'):
        model_path = 'checkpoints/dsp_intensity/tier1.pkl'
    with_profile = not args.no_profile and model_path is not None

    cards = collect_cards(df, model_path, args.window, args.hop,
                          args.max_audio_sec, with_profile,
                          audio_format=args.audio_format)
    if not cards:
        raise SystemExit('no readable clips')

    report_key = re.sub(r'[^A-Za-z0-9]+', '_',
                        os.path.splitext(os.path.basename(args.csv))[0])
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(build_html(cards, title, os.path.basename(args.csv), report_key))
    size_mb = os.path.getsize(args.out) / 1e6
    print(f'\nwrote {args.out} ({size_mb:.1f} MB, self-contained)')


if __name__ == '__main__':
    main()

