"""Pre-download the model/tokenizer assets ``25_transformer`` needs at runtime.

This exists so a container is *warm* when it starts: the first phonemization,
pitch extraction or ASR call in ``2_preprocess.py`` / ``test_*.py`` otherwise
blocks on a HuggingFace download, which on a rented vast.ai box is dead time
(and fails outright if the instance has no route to huggingface.co).

What is fetched:

* ``therealvul/svc_helper`` -> ``rvc_rmvpe.pt`` — the RMVPE pitch model built
  by ``features.FeatureExtractor.rmvpe_model`` (stage-1 f0, ~100 MB).
* ``therealvul/g2pen_tokenizer_clean`` — the phoneme tokenizer used for
  ``phone_ids`` (a few MB).
* NLTK's ``averaged_perceptron_tagger_eng`` (+ ``punkt``/``cmudict``) into
  ``$NLTK_DATA``: ``horsephonemizer`` calls ``nltk.download`` at import time and
  ``HorsePhonemizer.tag_words`` needs the tagger for heteronyms.
* With ``--qwen``: the ~2.5 GB of ``Qwen/Qwen3-ASR-0.6B`` +
  ``Qwen/Qwen3-ForcedAligner-0.6B`` weights used by
  ``FeatureExtractor(load_qwen=True)``. Only needed for preprocessing runs that
  actually align audio (``2_preprocess.py`` / ``1_preprocess.py``); stage-2
  training with existing code tensors never touches them.

Everything lands in the HuggingFace cache (``$HF_HOME``) / ``$NLTK_DATA``, so
the same downloads are shared by every process in the container.

Usage::

    python docker/prefetch_models.py            # small assets only
    python docker/prefetch_models.py --qwen     # + ASR / forced aligner
    python docker/prefetch_models.py --nltk-only
"""

from __future__ import annotations

import argparse
import os


# Speech/pitch models. Kept as (repo_id, [filenames]) pairs so a plain
# ``snapshot_download`` can be used for the multi-file repos.
RMVPE_REPO = "therealvul/svc_helper"
RMVPE_FILE = "rvc_rmvpe.pt"
G2P_TOKENIZER_REPO = "therealvul/g2pen_tokenizer_clean"
QWEN_REPOS = ("Qwen/Qwen3-ASR-0.6B", "Qwen/Qwen3-ForcedAligner-0.6B")

# ``horsephonemizer`` imports ``nltk`` and calls
# ``nltk.download('averaged_perceptron_tagger_eng')`` at module import; the
# tagger itself lives in that package, ``punkt``/``cmudict`` are cheap insurance
# for ``nltk.pos_tag`` / g2p_en fallbacks.
NLTK_PACKAGES = (
    "averaged_perceptron_tagger_eng",
    "averaged_perceptron_tagger",
    "punkt",
    "cmudict",
)


def prefetch_nltk() -> None:
    """Download the NLTK data packages into ``$NLTK_DATA``."""
    import nltk

    target = os.environ.get("NLTK_DATA")
    if target:
        os.makedirs(target, exist_ok=True)
        if target not in nltk.data.path:
            nltk.data.path.insert(0, target)
    for package in NLTK_PACKAGES:
        ok = nltk.download(package, download_dir=target, quiet=True)
        print(f"[nltk] {package}: {'ok' if ok else 'FAILED'}")
        if not ok:
            raise SystemExit(
                f"Could not download NLTK package {package!r}; check network "
                "access to raw.githubusercontent.com during the build.")


def prefetch_hf(qwen: bool) -> None:
    """Download the HuggingFace assets (tokenizer, RMVPE, optionally Qwen3)."""
    from huggingface_hub import hf_hub_download, snapshot_download

    print(f"[hf] {G2P_TOKENIZER_REPO}")
    print("   ->", snapshot_download(G2P_TOKENIZER_REPO))

    print(f"[hf] {RMVPE_REPO}/{RMVPE_FILE}")
    print("   ->", hf_hub_download(RMVPE_REPO, RMVPE_FILE))

    if not qwen:
        print("[hf] skipping Qwen3-ASR / forced aligner (pass --qwen)")
        return

    for repo in QWEN_REPOS:
        # snapshot_download (not Qwen3ASRModel.from_pretrained) because the
        # latter wants a CUDA device in device_map and the build has none.
        print(f"[hf] {repo}")
        print("   ->", snapshot_download(repo))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-download model assets for 25_transformer.")
    parser.add_argument(
        "--qwen", action="store_true",
        help="also fetch Qwen3-ASR + Qwen3-ForcedAligner (~2.5 GB)")
    parser.add_argument(
        "--nltk-only", action="store_true",
        help="only download the (tiny) NLTK data packages")
    args = parser.parse_args(argv)

    prefetch_nltk()
    if not args.nltk_only:
        prefetch_hf(qwen=args.qwen)
    print("prefetch done; HF cache:", os.environ.get("HF_HOME", "~/.cache/huggingface"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
