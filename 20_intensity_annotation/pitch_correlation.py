"""Is the model's predicted intensity correlated with pitch (F0)?

Scores clips with the trained intensity model, extracts Praat F0 (parselmouth),
and reports:

  * overall (pooled) Pearson / Spearman correlation,
  * across-speaker correlation (speaker mean intensity vs speaker mean F0),
  * within-speaker correlation (per-speaker r and a fixed-effects pooled r
    computed after removing speaker means),
  * the same for F0 variability (std in semitones) and voiced fraction,
  * loudness (RMS dBFS) as a control: model input is peak-normalised, so a low
    intensity/loudness correlation is expected and shows the pitch relation is
    not just a level artefact.

Also emits a per-clip CSV so the analysis can be redone/plotted elsewhere.

Usage:
    python pitch_correlation.py --ckpt checkpoints/intensity_whisper_small_local/best-checkpoint.ckpt \
        --filelist data/v0.5/val_filelist_linux.txt --out-csv pitch_val.csv

    python pitch_correlation.py --ckpt ... \
        --filelist data/v0.5/train_filelist_linux.txt data/v0.5/val_filelist_linux.txt \
        --out-csv pitch_all.csv
"""
import argparse
import csv
import os
import re

import librosa
import numpy as np
import parselmouth
import torch
from scipy import stats

from inference import (load_model_and_context, extract_interp_features,
                       score_features, load_audio)


# ----------------------------------------------------------------------------
# metadata
# ----------------------------------------------------------------------------

def parse_speaker(path):
    """Best-effort speaker id from the corpus filename."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r'^(p\d+)_', stem)                       # VCTK pXXX_...
    if m:
        return m.group(1)
    m = re.match(r'^(\d{2})_(\d{2})_(\d{2})_([^_]+)_', stem)  # MLP dialogue
    if m:
        return m.group(4)
    m = re.match(r'^(.+?)_English__', stem)               # Genshin NAME_English__id
    if m:
        return m.group(1)
    m = re.match(r'^([a-z]\d+)_\d+_mic\d*$', stem)        # VCTK variant s5_016_mic1
    if m:
        return m.group(1)
    m = re.match(r'^clip_(\d+)$', stem)                   # JustMe2
    if m:
        return 'justme2'
    if re.match(r'^_\d+$', stem):                         # unnamed numeric clips
        return 'unnamed_numeric'
    if stem.startswith('_English__'):
        return 'genshin_unnamed'
    m = re.match(r'^(ex\d+)_', stem)                      # ad-hoc test clips
    if m:
        return m.group(1)
    return stem.split('_')[0]


def is_singing(path):
    return 'singing' in os.path.basename(path).lower()


def parse_corpus(path):
    """Coarse corpus/source label (recording conditions differ a lot)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    low = path.lower()
    if re.match(r'^(p\d+|[a-z]\d+_\d+_mic)', stem):
        return 'vctk'
    if re.match(r'^\d{2}_\d{2}_\d{2}_', stem):
        return 'mlp'
    if '_english__' in low:
        return 'genshin'
    if stem.startswith('clip_'):
        return 'justme2'
    if re.match(r'^_\d+$', stem):
        return 'unnamed'
    return 'other'



# ----------------------------------------------------------------------------
# pitch / loudness descriptors
# ----------------------------------------------------------------------------

def acoustic_stats(wave, sr=16000):
    """F0 (Praat autocorrelation) and raw-level descriptors for one clip.

    F0 is computed on the raw waveform: pitch is invariant to gain, and the
    model itself only ever sees peak-normalised audio.
    """
    out = dict(duration_s=len(wave) / sr)
    x = np.asarray(wave, dtype=np.float64)
    out['rms_dbfs'] = float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-10))
    out['peak_dbfs'] = float(20 * np.log10(np.max(np.abs(x)) + 1e-10))

    empty = dict(f0_median_hz=np.nan, f0_mean_hz=np.nan, f0_std_hz=np.nan,
                 f0_std_st=np.nan, f0_iqr_st=np.nan, voiced_frac=np.nan)
    if len(x) < int(0.05 * sr):
        out.update(empty)
        return out

    snd = parselmouth.Sound(x, sampling_frequency=sr)
    pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=60.0,
                            pitch_ceiling=500.0, very_accurate=True)
    f0 = pitch.selected_array['frequency']
    voiced = f0[f0 > 0]
    out['voiced_frac'] = float(len(voiced) / max(len(f0), 1))
    if len(voiced) < 5:
        out.update(empty)
        return out

    st = 12.0 * np.log2(voiced / 100.0)  # semitones re 100 Hz
    out['f0_median_hz'] = float(np.median(voiced))
    out['f0_mean_hz'] = float(np.mean(voiced))
    out['f0_std_hz'] = float(np.std(voiced))
    out['f0_std_st'] = float(np.std(st))
    out['f0_iqr_st'] = float(np.percentile(st, 75) - np.percentile(st, 25))
    return out


# ----------------------------------------------------------------------------
# model scoring
# ----------------------------------------------------------------------------

def score_batch(model, context, waves, device, window_sec, hop_sec, win_batch=32):
    """Return whole-clip score and per-clip window mean/std for a batch of waves."""
    norm = [librosa.util.normalize(np.asarray(w, dtype=np.float32)) for w in waves]
    feats, mask = extract_interp_features(context, norm, device=device)
    pooled, _, _ = score_features(model, feats, mask)
    whole = pooled.float().cpu().numpy() * 8.0 + 1.0  # 0..1 -> 1..9

    win = int(window_sec * 16000)
    hop = int(hop_sec * 16000)
    windows, counts = [], []
    for w in norm:
        starts = ([0] if len(w) <= win
                  else list(range(0, len(w) - win + 1, hop)))
        counts.append(len(starts))
        windows.extend([w[s:s + win] for s in starts])

    scores = []
    for i in range(0, len(windows), win_batch):
        wf, wm = extract_interp_features(context, windows[i:i + win_batch], device=device)
        wp, _, _ = score_features(model, wf, wm)
        scores.extend((wp.float().cpu().numpy() * 8.0 + 1.0).tolist())

    scores = np.asarray(scores, dtype=np.float64)
    means = np.empty(len(counts))
    stds = np.empty(len(counts))
    k = 0
    for j, c in enumerate(counts):
        seg = scores[k:k + c]
        k += c
        means[j] = seg.mean()
        stds[j] = seg.std()
    return whole, means, stds


# ----------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------

def corr_pair(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    n = int(m.sum())
    if n < 3:
        return dict(n=n, r=np.nan, p=np.nan, rho=np.nan, p_spearman=np.nan)
    r, p = stats.pearsonr(a[m], b[m])
    rho, ps = stats.spearmanr(a[m], b[m])
    return dict(n=n, r=float(r), p=float(p), rho=float(rho), p_spearman=float(ps))


def bootstrap_ci(a, b, n_boot=2000, seed=0):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 5:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        if np.std(a[idx]) == 0 or np.std(b[idx]) == 0:
            continue
        rs.append(np.corrcoef(a[idx], b[idx])[0, 1])
    return (float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5)))


def group_indices(groups):
    """dict speaker -> np.array of row indices (compute once, reuse)."""
    d = {}
    for i, g in enumerate(groups):
        d.setdefault(g, []).append(i)
    return {g: np.asarray(idx, dtype=np.int64) for g, idx in d.items()}


def demean_by_speaker(x, groups, gidx=None):
    x = np.asarray(x, float)
    out = np.full_like(x, np.nan)
    if gidx is None:
        gidx = group_indices(groups)
    for idx in gidx.values():
        if len(idx) >= 2:
            vals = x[idx]
            if np.all(np.isfinite(vals)):
                out[idx] = vals - vals.mean()
    return out


def within_permutation_p(x, y, groups, n_perm=2000, seed=0):
    """Permutation test for the fixed-effects (within-speaker) correlation.

    Predictions are shuffled *within* each speaker, so speaker means and the
    number of clips per speaker are preserved.  p = fraction of |r_perm| >= |r|.
    """
    gidx = group_indices(groups)
    multi = [idx for idx in gidx.values() if len(idx) > 1]
    xd = demean_by_speaker(x, groups, gidx)
    yd = demean_by_speaker(y, groups, gidx)
    m = np.isfinite(xd) & np.isfinite(yd)
    r0 = np.corrcoef(xd[m], yd[m])[0, 1]
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    ge = valid = 0
    for _ in range(n_perm):
        yp = y.copy()
        for idx in multi:
            yp[idx] = yp[rng.permutation(idx)]
        yd = demean_by_speaker(yp, groups, gidx)
        m = np.isfinite(xd) & np.isfinite(yd)
        if m.sum() > 3 and np.std(xd[m]) > 0 and np.std(yd[m]) > 0:
            rp = np.corrcoef(xd[m], yd[m])[0, 1]
            valid += 1
            ge += abs(rp) >= abs(r0)
    return float((ge + 1) / (valid + 1))



def variance_share_between(x, groups, gidx=None):
    """Fraction of variance that is between-speaker (ICC-like, one-way)."""
    x = np.asarray(x, float)
    if gidx is None:
        gidx = group_indices(groups)
    finite = x[np.isfinite(x)]
    grand = finite.mean()
    ss_between = 0.0
    for idx in gidx.values():
        vals = x[idx]
        vals = vals[np.isfinite(vals)]
        if len(vals):
            ss_between += len(vals) * (vals.mean() - grand) ** 2
    ss_total = ((finite - grand) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else np.nan


def speaker_aggregate(rows, min_clips, value):

    groups = {}
    for r in rows:
        groups.setdefault(r['speaker'], []).append(r)
    out = []
    for sp, rs in groups.items():
        vals = np.array([r[value] for r in rs], float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < min_clips:
            continue
        out.append(dict(speaker=sp, n=len(vals), mean=float(vals.mean())))
    return out


def partial_corr(x, y, controls):
    """Pearson r between x and y after linear residualisation on controls."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    C = np.column_stack([np.ones(len(x))] + [np.asarray(c, float) for c in controls])
    m = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(C), axis=1)
    if m.sum() < C.shape[1] + 3:
        return np.nan
    x, y, C = x[m], y[m], C[m]
    rx = x - C @ np.linalg.lstsq(C, x, rcond=None)[0]
    ry = y - C @ np.linalg.lstsq(C, y, rcond=None)[0]
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def cluster_bootstrap_within(x, y, groups, n_boot=1000, seed=0):
    """CI for the fixed-effects r, resampling speakers (clips are not independent)."""
    xd = demean_by_speaker(x, groups)
    yd = demean_by_speaker(y, groups)
    by = {}
    for i, g in enumerate(groups):
        if np.isfinite(xd[i]) and np.isfinite(yd[i]):
            by.setdefault(g, []).append(i)
    spk = list(by)
    if len(spk) < 5:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n_boot):
        chosen = rng.choice(len(spk), len(spk), replace=True)
        idx = np.concatenate([by[spk[j]] for j in chosen])
        if np.std(xd[idx]) == 0 or np.std(yd[idx]) == 0:
            continue
        rs.append(np.corrcoef(xd[idx], yd[idx])[0, 1])
    return (float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5)))



def report(rows, title, min_speaker_clips=8, min_agg_clips=5):
    # The `_001.pt`-style numeric stems collide across source folders during
    # preprocessing (several different speakers map to the same file), so they
    # cannot be assigned to a speaker and would also duplicate audio in pooled
    # stats. Drop them from the pitch analysis and say so.
    dropped = [r for r in rows if r['speaker'] == 'unnamed_numeric']
    rows = [r for r in rows if r['speaker'] != 'unnamed_numeric']
    print('\n' + '=' * 78)
    n_sp = len({r['speaker'] for r in rows})
    print(f'{title} ({len(rows)} clips, {n_sp} speakers'
          + (f'; dropped {len(dropped)} ambiguous _NNN clips' if dropped else '') + ')')
    print('=' * 78)


    groups = [r['speaker'] for r in rows]
    inten = np.array([r['intensity_winmean'] for r in rows], float)
    inten_whole = np.array([r['intensity_whole'] for r in rows], float)

    targets = [
        ('F0 median (Hz)', 'f0_median_hz'),
        ('log2 F0 median', 'log2_f0_median'),
        ('F0 std (semitones)', 'f0_std_st'),
        ('voiced fraction', 'voiced_frac'),
        ('RMS (dBFS, control)', 'rms_dbfs'),
        ('duration (s, control)', 'duration_s'),
    ]

    print('\n-- pooled across all clips (intensity = mean of 2 s windows) --')
    print(f'{"variable":24s} {"n":>5s} {"Pearson r":>10s} {"p":>10s} '
          f'{"Spearman":>9s} {"95% CI":>18s}')
    for label, key in targets:
        v = np.array([r[key] for r in rows], float)
        c = corr_pair(inten, v)
        lo, hi = bootstrap_ci(inten, v)
        print(f'{label:24s} {c["n"]:5d} {c["r"]:+10.3f} {c["p"]:10.2e} '
              f'{c["rho"]:+9.3f}   [{lo:+.3f}, {hi:+.3f}]')

    # ---- across speakers -------------------------------------------------
    print(f'\n-- across speakers (speaker means, >= {min_agg_clips} clips/speaker) --')
    by_sp = {}
    for r in rows:
        by_sp.setdefault(r['speaker'], []).append(r)
    sp_rows = []
    for sp, rs in by_sp.items():
        rs = [r for r in rs if np.isfinite(r['f0_median_hz'])]
        if len(rs) < min_agg_clips:
            continue
        sp_rows.append(dict(
            speaker=sp, n=len(rs), corpus=rs[0].get('corpus', 'other'),
            intensity=np.mean([r['intensity_winmean'] for r in rs]),
            f0=np.median([r['f0_median_hz'] for r in rs]),
            f0std=np.mean([r['f0_std_st'] for r in rs]),
            rms=np.mean([r['rms_dbfs'] for r in rs]),
        ))
    si = np.array([r['intensity'] for r in sp_rows])
    sf = np.array([r['f0'] for r in sp_rows])
    sl = np.log2(sf)
    print(f'{len(sp_rows)} speakers with >= {min_agg_clips} clips')
    for label, v in (('F0 median (Hz)', sf), ('log2 F0 median', sl),
                     ('F0 std (semitones)', np.array([r['f0std'] for r in sp_rows])),
                     ('RMS (dBFS, control)', np.array([r['rms'] for r in sp_rows]))):
        c = corr_pair(si, v)
        lo, hi = bootstrap_ci(si, v)
        print(f'{label:24s} {c["n"]:5d} {c["r"]:+10.3f} {c["p"]:10.2e} '
              f'{c["rho"]:+9.3f}   [{lo:+.3f}, {hi:+.3f}]')

    # source/corpus is a confound for across-speaker comparisons
    corpora = sorted({r['corpus'] for r in sp_rows})
    print('  per corpus (>= 8 speakers):')
    for cp in corpora:
        sub = [r for r in sp_rows if r['corpus'] == cp]
        if len(sub) < 8:
            continue
        c = corr_pair([r['intensity'] for r in sub], [r['f0'] for r in sub])
        print(f'    {cp:10s} {c["n"]:3d} speakers  r={c["r"]:+.3f} (p={c["p"]:.2e})')
    if len(corpora) > 1:
        # one column per corpus dummy (not one row per speaker)
        dummies = [[1.0 if r['corpus'] == cp else 0.0 for r in sp_rows]
                   for cp in corpora[:-1]]
        print(f'  across-speaker F0 | corpus r={partial_corr(si, sf, dummies):+.3f}')


    # ---- within speakers -------------------------------------------------
    print('\n-- within speakers --')
    per = []
    for sp, rs in by_sp.items():
        if len(rs) < min_speaker_clips:
            continue
        x = np.array([r['intensity_winmean'] for r in rs], float)
        y = np.array([r['f0_median_hz'] for r in rs], float)
        c = corr_pair(x, y)
        if np.isfinite(c['r']):
            per.append(dict(speaker=sp, n=c['n'], r=c['r'], rho=c['rho']))
    if per:
        rs = np.array([p['r'] for p in per])
        pos = int((rs > 0).sum())
        sign_p = stats.binomtest(pos, len(rs), 0.5).pvalue
        print(f'{len(per)} speakers with >= {min_speaker_clips} clips: '
              f'mean r {rs.mean():+.3f}, median {np.median(rs):+.3f}, '
              f'{pos}/{len(rs)} positive, sign-test p={sign_p:.2e}')
        print('  per-speaker r (8 speakers with most clips):')
        for p in sorted(per, key=lambda p: -p['n'])[:8]:
            print(f'    {p["speaker"]:28s} n={p["n"]:3d} r={p["r"]:+.3f}')
        print('  most negative per-speaker r:')
        for p in sorted(per, key=lambda p: p['r'])[:4]:
            print(f'    {p["speaker"]:28s} n={p["n"]:3d} r={p["r"]:+.3f}')

    # fixed-effects pooled within-speaker correlation
    print('\n  fixed-effects (speaker-demeaned) pooled correlation:')
    for label, key in (('F0 median (Hz)', 'f0_median_hz'),
                       ('log2 F0 median', 'log2_f0_median'),
                       ('F0 std (semitones)', 'f0_std_st'),
                       ('voiced fraction', 'voiced_frac'),
                       ('RMS (dBFS, control)', 'rms_dbfs')):
        y = np.array([r[key] for r in rows], float)
        xd = demean_by_speaker(inten, groups)
        yd = demean_by_speaker(y, groups)
        c = corr_pair(xd, yd)
        p_perm = within_permutation_p(inten, y, groups, n_perm=1000)
        lo, hi = cluster_bootstrap_within(inten, y, groups)
        print(f'    {label:24s} n={c["n"]:5d} r={c["r"]:+.3f} '
              f'[{lo:+.3f}, {hi:+.3f}] (perm p={p_perm:.4f})  rho={c["rho"]:+.3f}')

    f0_all = np.array([r['f0_median_hz'] for r in rows], float)
    print(f'  between-speaker share of variance: '
          f'intensity {variance_share_between(inten, groups):.2f}, '
          f'F0 {variance_share_between(f0_all, groups):.2f}')


    # ---- does the pitch relation merely reproduce the labels? -----------
    print('\n-- label control (are predictions just following pitch-correlated labels?) --')
    labels = np.array([r['label'] for r in rows], float)
    f0 = np.array([r['f0_median_hz'] for r in rows], float)
    c = corr_pair(inten, labels)
    print(f'  pooled intensity~label      r={c["r"]:+.3f} (p={c["p"]:.2e})')
    c = corr_pair(labels, f0)
    print(f'  pooled label~F0             r={c["r"]:+.3f} (p={c["p"]:.2e})')
    print(f'  pooled intensity~F0 | label r={partial_corr(inten, f0, [labels]):+.3f}')
    lab_d = demean_by_speaker(labels, groups)
    f0_d = demean_by_speaker(f0, groups)
    c = corr_pair(lab_d, f0_d)
    print(f'  within label~F0             r={c["r"]:+.3f} (n={c["n"]})')
    print(f'  within intensity~F0 | label r={partial_corr(demean_by_speaker(inten, groups), f0_d, [lab_d]):+.3f}')
    # across speakers
    sp_lab, sp_f0, sp_int = [], [], []
    for sp, rs in by_sp.items():
        rs = [r for r in rs if np.isfinite(r['f0_median_hz'])]
        if len(rs) < min_agg_clips:
            continue
        sp_lab.append(np.mean([r['label'] for r in rs]))
        sp_f0.append(np.median([r['f0_median_hz'] for r in rs]))
        sp_int.append(np.mean([r['intensity_winmean'] for r in rs]))
    c = corr_pair(sp_lab, sp_f0)
    print(f'  across-speaker label~F0     r={c["r"]:+.3f} (n={c["n"]})')
    print(f'  across-speaker intensity~F0 | label '
          f'r={partial_corr(sp_int, sp_f0, [sp_lab]):+.3f}')

    # ---- within-speaker partials controlling level and duration ---------
    print('\n-- within-speaker partial correlations (speaker-demeaned) --')
    int_d = demean_by_speaker(inten, groups)
    rms_d = demean_by_speaker(np.array([r['rms_dbfs'] for r in rows], float), groups)
    dur_d = demean_by_speaker(np.array([r['duration_s'] for r in rows], float), groups)
    print(f'  intensity~F0 | RMS, duration        r={partial_corr(int_d, f0_d, [rms_d, dur_d]):+.3f}')
    print(f'  intensity~F0 | label, RMS, duration r={partial_corr(int_d, f0_d, [lab_d, rms_d, dur_d]):+.3f}')



    # ---- whole-clip path as a robustness check --------------------------
    print('\n-- robustness: whole-clip pass (single forward pass) --')
    for label, key in (('F0 median (Hz)', 'f0_median_hz'),
                       ('log2 F0 median', 'log2_f0_median')):
        v = np.array([r[key] for r in rows], float)
        c = corr_pair(inten_whole, v)
        cw = corr_pair(demean_by_speaker(inten_whole, groups),
                       demean_by_speaker(v, groups))
        print(f'{label:24s} pooled r={c["r"]:+.3f} (p={c["p"]:.2e})  '
              f'within-speaker r={cw["r"]:+.3f}')


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def read_csv_rows(path):
    rows = []
    numeric = ['duration_s', 'rms_dbfs', 'peak_dbfs', 'f0_median_hz', 'f0_mean_hz',
               'f0_std_hz', 'f0_std_st', 'f0_iqr_st', 'voiced_frac',
               'log2_f0_median', 'intensity_whole', 'intensity_winmean',
               'intensity_winststd']
    with open(path, newline='', encoding='utf-8') as f:
        for d in csv.DictReader(f):
            rec = dict(d)
            rec['label'] = int(float(d['label']))
            for k in numeric:
                if k in d and d[k] not in ('', 'nan', 'None'):
                    rec[k] = float(d[k])
                else:
                    rec[k] = np.nan
            rec['singing'] = str(d.get('singing', '')).lower() in ('true', '1')
            rec['speaker'] = d.get('speaker') or parse_speaker(d['path'])
            rec['corpus'] = d.get('corpus') or parse_corpus(d['path'])
            rec['split'] = d.get('split', 'train')
            rows.append(rec)
    return rows


def report_all(rows, min_speaker_clips, min_agg_clips):
    report(rows, 'ALL CLIPS', min_speaker_clips, min_agg_clips)
    val_rows = [r for r in rows if r.get('split') == 'val']
    if 20 <= len(val_rows) < len(rows):
        report(val_rows, 'VAL SPLIT ONLY', min_speaker_clips, min_agg_clips)
    sing = [r for r in rows if r.get('singing')]
    if 20 <= len(sing) < len(rows):
        report(sing, 'SINGING CLIPS ONLY', min_speaker_clips, min_agg_clips)
    speech = [r for r in rows if not r.get('singing')]
    if 20 <= len(speech) < len(rows):
        report(speech, 'NON-SINGING (SPEECH) ONLY', min_speaker_clips, min_agg_clips)



def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt')
    ap.add_argument('--filelist', nargs='*')
    ap.add_argument('--from-csv', default=None,
                    help='skip scoring, analyse a CSV written by --out-csv')
    ap.add_argument('--config', default='config/base.yaml')
    ap.add_argument('--whisper_model', default=None)
    ap.add_argument('--window', type=float, default=2.0)
    ap.add_argument('--hop', type=float, default=0.5)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--n', type=int, default=0, help='0 = all clips (debug/smoke only)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-csv', default=None)
    ap.add_argument('--min_speaker_clips', type=int, default=8)
    ap.add_argument('--min_agg_clips', type=int, default=5)
    args = ap.parse_args()

    if args.from_csv:
        rows = read_csv_rows(args.from_csv)
        report_all(rows, args.min_speaker_clips, args.min_agg_clips)
        return
    if not args.ckpt or not args.filelist:
        ap.error('--ckpt and --filelist are required unless --from-csv is used')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, context, _ = load_model_and_context(
        args.ckpt, args.config, whisper_model=args.whisper_model, device=device)

    items, seen = [], {}
    for fl in args.filelist:
        split = 'val' if 'val' in os.path.basename(fl) else 'train'
        with open(fl, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path, label = line.rsplit('|', 1)
                path = path.strip()
                if not os.path.exists(path):
                    continue
                if path in seen:
                    if split == 'val':
                        seen[path]['split'] = 'val'
                    continue
                rec = dict(path=path, label=int(label), split=split,
                           speaker=parse_speaker(path), corpus=parse_corpus(path),
                           singing=is_singing(path))
                seen[path] = rec
                items.append(rec)

    if args.n:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(items), min(args.n, len(items)), replace=False)
        items = [items[i] for i in idx]

    print(f'{len(items)} clips, {len({r["speaker"] for r in items})} speakers, '
          f'device={device}')

    for i in range(0, len(items), args.batch_size):
        chunk = items[i:i + args.batch_size]
        waves = []
        for rec in chunk:
            wave, _ = load_audio(rec['path'])
            rec.update(acoustic_stats(wave))
            rec['log2_f0_median'] = (np.log2(rec['f0_median_hz'])
                                     if np.isfinite(rec['f0_median_hz']) else np.nan)
            waves.append(wave)
        whole, wmean, wstd = score_batch(model, context, waves, device,
                                         args.window, args.hop)
        for rec, w, m, s in zip(chunk, whole, wmean, wstd):
            rec['intensity_whole'] = float(w)
            rec['intensity_winmean'] = float(m)
            rec['intensity_winststd'] = float(s)
        done = min(i + args.batch_size, len(items))
        print(f'\rscored {done}/{len(items)}', end='', flush=True)
    print()

    rows = items
    if args.out_csv:
        fields = list(rows[0].keys())
        with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
            wr = csv.DictWriter(f, fieldnames=fields)
            wr.writeheader()
            wr.writerows(rows)
        print(f'\nwrote {args.out_csv}')

    report_all(rows, args.min_speaker_clips, args.min_agg_clips)



if __name__ == '__main__':
    main()


