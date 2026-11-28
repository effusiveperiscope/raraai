"""Bridge to the shared stage-1 codec and offline tooling in ``25_transformer``.

This project reuses the *artefacts* of ``25_transformer`` (RVQ codec checkpoints,
the ``.cond`` forced alignments, the source filelists) and its *library* code
(``features.py``, ``hints/``, ``stage2_wer.py``, ``commons.py``,
``speaker_map.py``, ``modeling.generator``).  It does **not** reuse the stage-2
AR model: that is the thing being replaced.

Copying ~5 k lines of shared code into a second tree would drift immediately and
silently, so the older project is appended to ``sys.path`` and imported as-is.
The symlinks next to this file cover the data, which is 400+ GB and must never
be copied.

    from shared import DATA, STAGE1, OLD, build_stage1

Never edit a file through a symlink: those are the old project's files.  If
something shared genuinely needs to change, change it in ``25_transformer`` and
it is picked up here on the next import.

Module names ``features``, ``hints``, ``stage2_wer``, ``commons``, ``utils``,
``speaker_map``, ``horsephonemizer`` and the ``modeling`` package belong to the
old project.  Do not create files with those names here -- the project root is
prepended to ``sys.path`` (so local modules win), which means a local
``utils.py`` would shadow the shared one for every import in this tree.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# fairseq (pulled in by modeling.generator) calls logging.basicConfig(DEBUG) at
# import time; install our handler first so that call becomes a no-op
# (25_transformer/AGENTS.md §6.18).
logging.basicConfig(level=logging.WARNING)

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "25_transformer"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Appended, not inserted: this project's modules win on a name clash, and the
# old project only supplies names we do not define.
if str(OLD) not in sys.path:
    sys.path.append(str(OLD))

# ---- shared artefacts (symlinks created by setup, see AGENTS.md) ---------- #
DATA = ROOT / "data"
FILELISTS_1 = ROOT / "filelists_1"
FILELISTS_2 = ROOT / "filelists_2"
STAGE1 = ROOT / "checkpoints_stage1"
STAGE2 = ROOT / "checkpoints_stage2"      # baselines only
HINTS = ROOT / "hints"
SPEAKERS = ROOT / "speakers.json"

#: Preprocessed corpora, all sharing the same layout:
#: ``<base>.phone_ids`` / ``.spec_codes`` / ``.pitch_codes`` / ``.cond``.
CORPORA = {
    "genshin": DATA / "genshin_plus_extras",
    "genshin_stage1": DATA / "genshin_full_stage1",
    "mlp": DATA / "mlp_stage2",
    "mlp_stage1": DATA / "mlp_stage1",
}

#: Stage-1 config: needed to *build* the frozen Generator for decoding.  Loaded
#: by path from the old project on purpose -- a copy here would drift from the
#: checkpoint it is supposed to describe.
STAGE1_CONFIG = OLD / "configs" / "base.yaml"

#: The current stage-2 baseline to beat, and the WER floor measured with it on
#: the 120-utterance offline matrix (see 25_transformer/AGENTS.md §4).
BASELINE_STAGE2 = STAGE2 / "transformer_hints_v2.4" / "last.ckpt"
BASELINE_WER = 0.1575
BASELINE_FLOOR = 0.0608


def build_stage1(checkpoint: str | Path | None = None, device="cuda:0"):
    """Frozen stage-1 ``Generator`` (the codec decode path) plus its config.

    This is the only part of stage 1 the renderer needs at inference: codes ->
    waveform.  Parameters are frozen and the module is in ``eval``; the RVQ
    quantisers are *not* frozen here because this object is only ever used to
    decode (the code-producing half belongs to preprocessing).
    """
    import torch
    from omegaconf import OmegaConf

    from commons import load_submodule_prefix
    from modeling.generator import Generator

    cfg = OmegaConf.load(STAGE1_CONFIG)
    net = Generator(hp=cfg)
    path = Path(checkpoint) if checkpoint else STAGE1 / "base_v0.7" / "last.ckpt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    load_submodule_prefix(net, "net_g.", state.get("state_dict", state),
                          quiet=True)
    net.eval().to(device)
    for p in net.parameters():
        p.requires_grad_(False)
    return net, cfg


def _self_test() -> int:
    """``python shared.py`` -- prove the bridge resolves before writing code."""
    print(f"project      {ROOT}")
    print(f"shared code  {OLD}")
    bad = 0
    for name, path in (("data", DATA), ("stage-1 ckpts", STAGE1),
                       ("filelists_2", FILELISTS_2), ("hints", HINTS),
                       ("speakers.json", SPEAKERS),
                       ("stage-1 config", STAGE1_CONFIG),
                       ("baseline stage-2", BASELINE_STAGE2)):
        ok = path.exists()
        bad += not ok
        print(f"  {'ok ' if ok else 'MISSING'}  {name:18s} {path}")
    for name, corpus in CORPORA.items():
        ok = corpus.exists()
        bad += not ok
        print(f"  {'ok ' if ok else 'MISSING'}  corpus:{name:11s} {corpus}")
    import commons  # noqa: F401
    import speaker_map  # noqa: F401
    import stage2_wer  # noqa: F401
    print("  ok   imports: commons, speaker_map, stage2_wer")
    import features  # noqa: F401
    import modeling.generator  # noqa: F401
    print("  ok   imports: features, modeling.generator")
    return bad


if __name__ == "__main__":
    sys.exit(1 if _self_test() else 0)
