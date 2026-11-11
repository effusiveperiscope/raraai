"""Compute cheap, theoretically motivated DSP features for perceptual intensity.

The feature set is organised around the two axes that matter for a 1-9
perceptual-intensity scale anchored at whisper:

  * **phonation / voicing** (whisper = no periodic vibration): CPP, voiced
    fraction, F0 (masked when unvoiced), spectral contrast, MFCC2
  * **spectral excitation / effort** (conditional on voicing): tilt, alpha
    ratio, Hammarberg index, HF energy ratio, MFCC3, contrast band 3, flatness,
    ZCR, RMS dynamics, spectral flux

Everything is derived from **one STFT** plus **one Praat pitch pass**.  Praat
HNR is optional because it costs ~30 ms/clip (the whole Tier-1 extractor is
~20 ms/clip); CPP from the same STFT is a cheaper and more label-relevant
periodicity measure.

Usage:
    # cache features for training / evaluation
    python dsp_features.py \
        --filelist data/v0.5/train_filelist_linux.txt data/v0.5/val_filelist_linux.txt \
        --split train val --tier 1 --out-csv dsp_features.csv

    # quick look at one file
    python dsp_features.py --audio some.wav --tier 1
"""
import argparse
import csv
import os
import re
import sys

import librosa
import numpy as np
import parselmouth

SR = 16000
N_FFT = 1024
HOP = 160          # 10 ms, aligns with Praat's default time step
N_MELS = 26
N_MFCC = 5

# Frame columns that are aggregated with mean + std into clip features.
_FRAME_COLUMNS = [
    'cpp', 'tilt', 'alpha', 'hamm', 'hfr', 'cen', 'flat', 'roll', 'bw',
    'zcr', 'rms', 'flux',
] + [f'con{i}' for i in range(7)] + [f'mfcc{i}' for i in range(N_MFCC)]

# The three tiers validated in the DSP notes.  Tier 1 is the recommended
# compute/accuracy point; Tier 2 adds HNR and a few shape features.
TIER0 = [
    'cpp_mean', 'log2_f0_median', 'f0_std_st', 'voiced_frac',
    'con2_mean', 'mfcc2_mean',
]
TIER1 = TIER0 + [
    'tilt_mean', 'alpha_mean', 'hamm_mean', 'hfr_mean',
    'mfcc3_mean', 'con3_mean', 'flat_mean', 'zcr_mean', 'rms_std', 'flux_mean',
]
TIER2 = TIER1 + [
    'cpp_std', 'hnr_mean', 'hnr_std', 'f0_iqr_st',
    'cen_mean', 'roll_mean', 'mfcc1_mean', 'mfcc4_mean', 'bw_mean', 'duration_s',
]

FEATURE_TIERS = {0: TIER0, 1: TIER1, 2: TIER2}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_wave(path, sr=SR):
    """Load a preprocessed .pt tensor or any librosa-readable audio file."""
    if str(path).lower().endswith('.pt'):
        import torch
        wave = torch.load(path, map_location='cpu', weights_only=False)
        wave = np.asarray(wave, dtype=np.float32).squeeze()
    else:
        wave, _ = librosa.load(path, sr=sr, mono=True, res_type='fft')
        wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim > 1:
        wave = wave.mean(axis=0)
    return wave.astype(np.float32)


def _fit_len(x, n):
    """Trim/pad a 1-D frame array to exactly n samples."""
    x = np.asarray(x)
    if x.shape[0] == n:
        return x
    if x.shape[0] > n:
        return x[:n]
    return np.pad(x, (0, n - x.shape[0]), mode='edge')


def _band_power(S, freqs, lo, hi):
    mask = (freqs >= lo) & (freqs < hi)
    if not mask.any():
        return np.zeros(S.shape[1], dtype=S.dtype)
    return S[mask].sum(axis=0)


# ---------------------------------------------------------------------------
# frame features
# ---------------------------------------------------------------------------

def frame_features(wave, include_hnr=False):
    """Return (frames, times).

    `frames` maps name -> 1-D array of length T (10 ms frames).  `wave` should
    already be peak-normalised: the target is loudness-independent.
    """
    wave = np.asarray(wave, dtype=np.float32).squeeze()
    S = np.abs(librosa.stft(wave, n_fft=N_FFT, hop_length=HOP,
                            center=True)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    T = S.shape[1]
    times = librosa.frames_to_time(np.arange(T), sr=SR, hop_length=HOP)
    eps = 1e-12
    f = {}

    # --- spectral shape -----------------------------------------------------
    mel = librosa.feature.melspectrogram(S=S, sr=SR, n_mels=N_MELS,
                                         fmax=SR / 2)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        f[f'mfcc{i}'] = _fit_len(mfcc[i], T)

    con = librosa.feature.spectral_contrast(S=S, sr=SR, fmin=200.0, n_bands=6)
    for i in range(con.shape[0]):
        f[f'con{i}'] = _fit_len(con[i], T)

    f['cen'] = _fit_len(librosa.feature.spectral_centroid(S=S, sr=SR)[0], T)
    f['flat'] = _fit_len(librosa.feature.spectral_flatness(S=S)[0], T)
    f['roll'] = _fit_len(
        librosa.feature.spectral_rolloff(S=S, sr=SR, roll_percent=0.85)[0], T)
    f['bw'] = _fit_len(librosa.feature.spectral_bandwidth(S=S, sr=SR)[0], T)

    # --- spectral balance (eGeMAPS-style band ratios) -----------------------
    lf = np.log2(np.maximum(freqs, 20.0))
    lp = np.log10(S + eps)
    lfm = lf - lf.mean()
    lpm = lp - lp.mean(axis=0)
    f['tilt'] = _fit_len((lfm[:, None] * lpm).sum(0) / (lfm ** 2).sum(), T)

    a = _band_power(S, freqs, 50, 1000)
    b = _band_power(S, freqs, 1000, 5000)
    c = _band_power(S, freqs, 0, 2000)
    d = _band_power(S, freqs, 2000, 5000)
    hf = _band_power(S, freqs, 2000, 8000)
    tot = _band_power(S, freqs, 0, 8000)
    f['alpha'] = 10 * np.log10(a / (b + eps) + eps)
    f['hamm'] = 10 * np.log10((c + eps) / (d + eps))
    f['hfr'] = 10 * np.log10((hf + eps) / (tot + eps))

    # --- periodicity: CPP from the same STFT --------------------------------
    cep = np.fft.irfft(np.log(S + 1e-10), axis=0)
    quef = np.arange(cep.shape[0]) / SR
    band = (quef >= 1.0 / 500.0) & (quef <= 1.0 / 60.0)
    f['cpp'] = cep[band].max(axis=0) - np.median(cep[band], axis=0)

    # --- temporal -----------------------------------------------------------
    f['zcr'] = _fit_len(librosa.feature.zero_crossing_rate(
        wave, frame_length=N_FFT, hop_length=HOP)[0], T)
    f['rms'] = _fit_len(librosa.feature.rms(
        S=S, frame_length=N_FFT)[0], T)
    Sn = S / (S.sum(axis=0, keepdims=True) + eps)
    flux = np.sqrt((np.diff(Sn, axis=1) ** 2).sum(axis=0))
    f['flux'] = np.concatenate([[0.0], flux])

    # --- F0 / voicing (Praat) ----------------------------------------------
    snd = parselmouth.Sound(wave.astype(np.float64), sampling_frequency=SR)
    pitch = snd.to_pitch_ac(time_step=HOP / SR, pitch_floor=60.0,
                            pitch_ceiling=500.0, very_accurate=False)
    f0_raw = np.asarray(pitch.selected_array['frequency'], dtype=np.float64)
    t_raw = np.asarray(pitch.xs(), dtype=np.float64)
    voiced_raw = (f0_raw > 0).astype(np.float64)
    # interpolate onto the STFT grid for frame-level profiles; clip aggregates
    # use the raw Praat arrays via the extra keys below.
    f['f0_hz'] = np.interp(times, t_raw, f0_raw)
    f['voiced'] = np.interp(times, t_raw, voiced_raw)

    if include_hnr:
        hnr_obj = snd.to_harmonicity_cc(time_step=HOP / SR, minimum_pitch=60.0)
        hnr_raw = np.asarray(hnr_obj.values, dtype=np.float64).squeeze()
        hnr_t = np.asarray(hnr_obj.xs(), dtype=np.float64)
        hnr_raw = np.where(hnr_raw > -200, hnr_raw, np.nan)
        valid = np.isfinite(hnr_raw)
        if valid.sum() >= 2:
            f['hnr'] = np.interp(times, hnr_t[valid], hnr_raw[valid])
        else:
            f['hnr'] = np.full(T, np.nan)
    else:
        f['hnr'] = np.full(T, np.nan)

    # extras used only for clip aggregation / window slicing
    f['_f0_raw'] = f0_raw
    f['_voiced_raw'] = voiced_raw
    f['_pitch_times'] = t_raw
    f['_hnr_raw'] = f['hnr'] if include_hnr else np.array([])
    return f, times



# ---------------------------------------------------------------------------
# clip aggregation
# ---------------------------------------------------------------------------

def clip_features(frames, raw_wave):
    """Aggregate frame features into one scalar dict per clip."""
    out = {}
    for name in _FRAME_COLUMNS + ['hnr']:
        v = np.asarray(frames[name], dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            out[f'{name}_mean'] = np.nan
            out[f'{name}_std'] = np.nan
        else:
            out[f'{name}_mean'] = float(v.mean())
            out[f'{name}_std'] = float(v.std())

    f0 = np.asarray(frames['_f0_raw'], dtype=np.float64)
    voiced = f0[f0 > 0]
    voiced_raw = np.asarray(frames['_voiced_raw'], dtype=np.float64)
    out['voiced_frac'] = float(voiced_raw.mean()) if voiced_raw.size else np.nan
    if voiced.size:
        med = float(np.median(voiced))
        st = 12.0 * np.log2(voiced / med)
        out['f0_median_hz'] = med
        out['log2_f0_median'] = float(np.log2(med))
        out['f0_std_st'] = float(np.std(st))
        q25, q75 = np.percentile(st, [25, 75])
        out['f0_iqr_st'] = float(q75 - q25)
    else:
        out['f0_median_hz'] = np.nan
        out['log2_f0_median'] = np.nan
        out['f0_std_st'] = np.nan
        out['f0_iqr_st'] = np.nan

    hnr = np.asarray(frames['_hnr_raw'], dtype=np.float64)
    hnr = hnr[np.isfinite(hnr)]
    if hnr.size:
        out['hnr_mean'] = float(hnr.mean())
        out['hnr_std'] = float(hnr.std())
    else:
        out['hnr_mean'] = np.nan
        out['hnr_std'] = np.nan

    x = np.asarray(raw_wave, dtype=np.float64)
    out['duration_s'] = float(len(x) / SR)
    out['rms_dbfs'] = float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-10))
    out['peak_dbfs'] = float(20 * np.log10(np.max(np.abs(x)) + 1e-10))
    return out


def extract(wave, tier=1, include_hnr=None, with_frames=False,
            all_features=False):
    """Extract one clip's feature dict for `tier` (0/1/2).

    Loudness independence: features are computed on a peak-normalised copy of
    the wave; the raw level is kept only as `rms_dbfs` / `peak_dbfs` for the
    whisper gate and diagnostics.

    `all_features=True` returns every aggregate column (used by the feature
    cache so the tier can be changed without re-extracting).
    """
    raw = np.asarray(wave, dtype=np.float32).squeeze()
    if raw.ndim > 1:
        raw = raw.mean(axis=0)
    if include_hnr is None:
        include_hnr = tier >= 2
    norm = raw / (np.max(np.abs(raw)) + 1e-9)

    frames, times = frame_features(norm, include_hnr=include_hnr)
    clip = clip_features(frames, raw)
    cols = FEATURE_TIERS[int(tier)]
    out = dict(clip) if all_features else {k: clip.get(k, np.nan) for k in cols}
    if with_frames:
        return out, frames, times
    return out


def extract_path(path, tier=1, include_hnr=None, with_frames=False,
                 all_features=False):
    return extract(load_wave(path), tier=tier, include_hnr=include_hnr,
                   with_frames=with_frames, all_features=all_features)



# ---------------------------------------------------------------------------
# batch CSV
# ---------------------------------------------------------------------------

def _worker(job):
    """Top-level worker (picklable) for --workers."""
    path, tier, include_hnr = job
    try:
        feats = extract_path(path, tier=tier, include_hnr=include_hnr,
                             all_features=True)
        return path, feats, None
    except Exception as exc:  # keep the batch going
        return path, None, f'{type(exc).__name__}: {exc}'


def _row_for(path, label, split, feats):
    from dsp_common import (parse_speaker, parse_corpus, is_singing,
                            is_true_whisper)
    row = {
        'path': path, 'label': label, 'split': split,
        'speaker': parse_speaker(path), 'corpus': parse_corpus(path),
        'singing': int(is_singing(path)), 'whisper': int(is_true_whisper(path)),
    }
    row.update(feats)
    return row


def batch_extract(items, tier=1, include_hnr=None, workers=0, limit=0,
                  verbose=True):
    """items: list of (path, label, split). Returns list of row dicts."""
    if limit:
        items = items[:limit]
    jobs = [(p, tier, include_hnr) for p, _, _ in items]
    results = []
    if workers and workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_worker, jobs, chunksize=8))
    else:
        for i, job in enumerate(jobs, 1):
            results.append(_worker(job))
            if verbose and (i % 100 == 0 or i == len(jobs)):
                print(f'\r{i}/{len(jobs)} clips', end='', flush=True)
        if verbose and jobs:
            print()

    rows = []
    errors = 0
    for (path, feats, err), (_, label, split) in zip(results, items):
        if feats is None:
            errors += 1
            print(f'[skip] {path}: {err}', file=sys.stderr)
            continue
        rows.append(_row_for(path, label, split, feats))
    if verbose and errors:
        print(f'{errors} clip(s) skipped', file=sys.stderr)
    return rows


def write_csv(rows, out_csv):
    if not rows:
        raise SystemExit('no rows to write')
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # stable column order: metadata first, then numeric features
    meta = ['path', 'label', 'split', 'speaker', 'corpus', 'singing', 'whisper']
    feat_cols = [k for k in rows[0] if k not in meta]
    fieldnames = meta + feat_cols
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v)
                        for k, v in row.items()})



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--filelist', nargs='+', default=None,
                        help='one or more path|label filelists')
    parser.add_argument('--split', nargs='+', default=None,
                        help='split name per --filelist (train / val / ...)')
    parser.add_argument('--audio', type=str, default=None,
                        help='extract one audio file and print its features')
    parser.add_argument('--tier', type=int, default=1, choices=[0, 1, 2])
    parser.add_argument('--include-hnr', action='store_true',
                        help='compute Praat HNR (slow; implied by --tier 2)')
    parser.add_argument('--out-csv', type=str, default=None)
    parser.add_argument('--limit', type=int, default=0, help='max clips (0 = all)')
    parser.add_argument('--workers', type=int, default=0,
                        help='parallel processes (0 = sequential)')
    args = parser.parse_args()

    include_hnr = args.include_hnr or args.tier >= 2

    if args.audio:
        feats = extract_path(args.audio, tier=args.tier, include_hnr=include_hnr)
        for k, v in feats.items():
            print(f'{k:20s} {v:.6g}' if isinstance(v, float) else f'{k:20s} {v}')
        return

    if not args.filelist:
        parser.error('pass --filelist or --audio')
    if not args.out_csv:
        parser.error('--out-csv is required for batch extraction')

    from dsp_common import read_filelist
    splits = args.split or ['unknown'] * len(args.filelist)
    if len(splits) != len(args.filelist):
        parser.error('--split must have one entry per --filelist')
    items = []
    for filelist, split in zip(args.filelist, splits):
        items.extend((path, label, split) for path, label in read_filelist(filelist))

    rows = batch_extract(items, tier=args.tier, include_hnr=include_hnr,
                         workers=args.workers, limit=args.limit)
    write_csv(rows, args.out_csv)
    print(f'wrote {args.out_csv} ({len(rows)} clips, tier {args.tier})')


if __name__ == '__main__':
    main()

