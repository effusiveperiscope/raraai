"""Small, dependency-light helpers shared by the DSP intensity pipeline.

This module deliberately imports only numpy/os/re, so `dsp_features.py` can run
without torch, transformers or whisper. Speaker/corpus parsing matches
`pitch_correlation.py` so the same clips get the same group labels.
"""
import os
import re

import numpy as np


# ---------------------------------------------------------------------------
# filelist / metadata
# ---------------------------------------------------------------------------

def read_filelist(path):
    """Read a `path|label` filelist (labels are integers 1-9)."""
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            audio_path, label = line.rsplit('|', 1)
            items.append((audio_path.strip(), int(label)))
    return items


def parse_speaker(path):
    """Best-effort speaker id from the corpus filename (see pitch_correlation.py)."""
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


def is_singing(path):
    return 'singing' in os.path.basename(path).lower()


def is_true_whisper(path):
    """True (unvoiced) whisper clips: `ex0X_whisper_NNNNN`.

    Note: MLP `*_Whispering__*` clips are acted/breathy but voiced and are
    labelled 2-4 in v0.5, so they are deliberately *not* matched here.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    return bool(re.match(r'^ex\d+_whisper', stem, re.IGNORECASE))


def is_whispering(path):
    return 'whisper' in os.path.basename(path).lower()


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def metrics(pred, target):
    """Pearson r, MAE, R^2 (same definition as eval_val.py)."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    r = float(np.corrcoef(pred, target)[0, 1]) if pred.std() > 0 else float('nan')
    mae = float(np.abs(pred - target).mean())
    ss_res = float(((pred - target) ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return r, mae, r2


def demean_by_group(values, groups):
    """Subtract each group's mean (speaker fixed effects)."""
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    out = values.copy()
    for g in np.unique(groups):
        mask = groups == g
        out[mask] -= values[mask].mean(axis=0) if values.ndim > 1 else values[mask].mean()
    return out


def within_speaker_r(pred, target, groups):
    r, _, _ = metrics(demean_by_group(pred, groups), demean_by_group(target, groups))
    return r
