"""Offline conditioning-hint builders for the stage-2 transformer.

This package is a *port* of ``/mnt/data/Code/raraai/20_intensity_annotation``
(see ``hints/README.md`` for provenance and for the one deviation: the shipped
model was re-fitted under scikit-learn 1.9 so it can be loaded here).

Nothing that trains or runs the model imports this package: it only produces
``<base>.cond`` tensors (``build.py``), which the stage-2 dataset reads.  That
is deliberate -- it keeps ``torch``-only training/inference free of
scikit-learn / parselmouth / pandas / transformers-embedding dependencies.

Import note
-----------
The ported modules keep their original top-level filenames (``dsp_features``,
``dsp_intensity``, ``dsp_common``, ``whisper_gate``) and their original
``from dsp_common import ...`` style imports, because the joblib model pickle
refers to ``dsp_intensity.DSPIntensityModel`` and ``whisper_gate.WhisperGate``
*by module path*.  Putting this directory on ``sys.path`` -- which
:func:`ensure_importable` does -- is what makes those names resolve.  Renaming
them into a package path (``hints.dsp_intensity``) would require re-pickling the
model and would break again the next time the directory moves.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def ensure_importable() -> str:
    """Prepend this directory to ``sys.path`` (idempotent). Returns it."""
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    return HERE


__all__ = ["HERE", "ensure_importable"]
