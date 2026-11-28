"""Build the compact plan cache used by ``dataset_plan.py``.

Deriving a plan needs the stage-1 ``.spec`` (769 x 4T floats, ~0.5 MB/clip) for
the silence VAD.  Training must not read that every epoch, so this writes a
few-KB ``.npz`` per clip under ``plans/<corpus>/`` (gitignored).  The cache is
purely derived: delete it and rebuild whenever ``plan.py``'s derivation or
``PLAN_CACHE_VERSION`` changes.

Usage (from ``26_guided_tts``)::

    python scratch/build_plan_cache.py \
        --filelists data/genshin_plus_extras/train_stage2.txt \
                    data/genshin_plus_extras/val_stage2.txt
    # only genshin + mlp, first 1k clips
    python scratch/build_plan_cache.py --corpora genshin_full mlp --limit 1000
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plan import (  # noqa: E402
    PLAN_CACHE_VERSION, PlanError, load_plan, save_plan_cache,
)
from dataset_plan import cache_path, resolve  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filelists", nargs="+", default=[
        "data/genshin_plus_extras/train_stage2.txt"])
    ap.add_argument("--cache-dir", default=str(ROOT / "plans"))
    ap.add_argument("--corpora", nargs="*", default=None,
                    help="substring filter on the clip path, e.g. genshin_full mlp")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--every", type=int, default=2000)
    args = ap.parse_args()

    seen: set[str] = set()
    todo: list[dict] = []
    for fl in args.filelists:
        path = resolve(fl)
        if not path.exists():
            print(f"skip missing filelist {path}")
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 4:
                continue
            phone = resolve(parts[0])
            if str(phone) in seen:
                continue
            seen.add(str(phone))
            if args.corpora and not any(c in str(phone) for c in args.corpora):
                continue
            codes = resolve(parts[1])
            if not (phone.exists() and codes.exists()):
                continue
            if not Path(str(codes)[: -len(".spec_codes")] + ".cond").exists():
                continue
            cache = cache_path(args.cache_dir, phone)
            if cache.exists() and not args.overwrite:
                continue
            todo.append({"phone": phone, "codes": codes, "sid": int(parts[3])})
            if args.limit and len(todo) >= args.limit:
                break
        if args.limit and len(todo) >= args.limit:
            break

    print(f"plan cache: {len(todo)} to build -> {args.cache_dir} "
          f"(version {PLAN_CACHE_VERSION})")
    if not todo:
        return 0

    import torch
    reasons = Counter()
    t0 = time.time()
    built = 0
    for i, it in enumerate(todo, 1):
        try:
            phone_ids = torch.load(it["phone"], map_location="cpu",
                                   weights_only=False).reshape(-1)
            plan = load_plan(it["phone"], speaker=it["sid"])
            save_plan_cache(plan, cache_path(args.cache_dir, it["phone"]))
            built += 1
        except PlanError as exc:
            msg = str(exc)
            if msg.startswith("word run"):
                reasons["degenerate word run"] += 1
            elif "field lengths" in msg:
                reasons["word/phone length mismatch"] += 1
            elif "no word fields" in msg:
                reasons["no word fields"] += 1
            else:
                reasons[msg.split(":")[0][:36]] += 1
        except Exception as exc:  # noqa: BLE001
            reasons[type(exc).__name__] += 1
        if i % args.every == 0 or i == len(todo):
            dt = time.time() - t0
            print(f"  {i}/{len(todo)}  built {built}  {dt:5.1f}s "
                  f"({i / max(dt, 1e-6):.0f}/s)")

    print(f"done: {built} built, {len(todo) - built} skipped")
    if reasons:
        print("  " + "  ".join(f"{k} {v}" for k, v in reasons.most_common()))
    return 0 if built else 1


if __name__ == "__main__":
    raise SystemExit(main())
