"""Load the ported DSP intensity model.

``DSPIntensityModel.save`` uses ``joblib.dump`` on an object graph that includes
``dsp_intensity.DSPIntensityModel`` and (usually) ``whisper_gate.WhisperGate``,
i.e. the pickle stores *top-level module paths*.  Two consequences:

* the modules must be importable under their original names -- see
  ``hints/__init__.ensure_importable``;
* the pickle is also bound to the scikit-learn version that wrote it.  The
  upstream ``tier1_augmented.pkl`` was written with 1.7.2 and fails on 1.9 with
  ``ModuleNotFoundError: No module named '_loss'``; the copy shipped in
  ``hints/models/dsp_intensity/`` was re-fitted under the repo's environment with
  the same recipe and reproduces the published metrics (see ``hints/README.md``).
  ``load_dsp_model`` turns that failure into an actionable message instead of a
  bare ImportError.
"""

from __future__ import annotations

import os
import sys

# Self-contained so this module works both as ``hints.model_io`` and when
# ``hints/`` is put on sys.path directly (``python hints/build.py``), where the
# ported modules are top-level, exactly like in the upstream project.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

_REBUILD_HINT = (
    "Re-fit the model in this environment:\n"
    "    python hints/dsp_intensity.py --features-csv <csv> --dedupe --tier 1 "
    "--cv group --fit-gate --save hints/models/dsp_intensity/tier1_augmented.pkl\n"
    "(any feature csv with the same 16 tier-1 columns, e.g. the upstream "
    "dsp_features_augmented.csv).")


def load_dsp_model(path: str):
    """Load a ``DSPIntensityModel`` pkl, with a clear error on version skew.

    Args:
        path: Path to a ``.pkl`` written by ``DSPIntensityModel.save``.

    Returns:
        The loaded model (``.tier``, ``.predict(df)``, ``.predict_raw(df)``).

    Raises:
        FileNotFoundError: ``path`` does not exist.
        RuntimeError: the pickle cannot be unpickled in this environment
            (typically a scikit-learn version mismatch).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"DSP intensity model not found: {path}")
    import joblib
    from dsp_intensity import DSPIntensityModel

    try:
        return DSPIntensityModel.load(path)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"cannot unpickle {path}: {exc}.  The pickle is bound to the "
            f"scikit-learn version that wrote it.\n{_REBUILD_HINT}") from exc
