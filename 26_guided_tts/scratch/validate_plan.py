"""Validate ``plan.py`` over a corpus subset: does a word-rate plan explain the
code frames we have to render?

This is the load-bearing assumption of the rewrite, and it is testable on a CPU
before any model code exists:

* **partition** -- ``lead + sum(word_dur + pause_after) == len(spec_codes)``
  for every clip (a plan that misses this cannot be trained on without silent
  misalignment);
* **coverage** -- every phone has >= 1 frame and the phone durations sum to the
  word durations;
* **silence** -- how much of each clip the VAD turns into explicit pauses, and
  whether that matches the stage-1 ``.f0`` voicing;
* **pitch** -- the derived contour against the stage-1 ``.f0``: correlation,
  absolute cents, and cents after removing a per-clip offset.  Absolute error
  tests ``register_st``; offset-removed error tests the anchors.

Usage (from ``26_guided_tts``)::

    python scratch/validate_plan.py --limit 200
    python scratch/validate_plan.py --filelist data/mlp_stage2/train_stage2.txt \
        --limit 300 --show 2
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plan import (  # noqa: E402
    PITCH_FPS, F0_REF_HZ, Plan, PlanError, plan_from_cond,
)

DEFAULT_FILELIST = "data/genshin_plus_extras/val_stage2.txt"


def _resolve(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    for base in (ROOT, ROOT.parent / "25_transformer"):
        if (base / p).exists():
            return base / p
    return ROOT / p


def _load(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("f0", "spec", "data", "tensor"):
            if key in obj:
                return obj[key]
        return next(iter(obj.values()))
    return obj


def _siblings(codes: Path) -> tuple[Path | None, Path | None]:
    s = str(codes)
    if "_stage2/" not in s:
        return None, None
    s = s.replace("_stage2/", "_stage1/")
    base = s[: -len(".spec_codes")]
    spec, f0 = Path(base + ".spec"), Path(base + ".f0")
    return (spec if spec.exists() else None, f0 if f0.exists() else None)


def _stats(x) -> dict:
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)),
            "p90": float(np.percentile(a, 90)),
            "mean": float(a.mean())}


def _pitch_report(plan: Plan, f0: np.ndarray) -> dict | None:
    gt = np.asarray(f0, dtype=np.float32).reshape(-1)
    ph = plan.pitch_hz()
    n = min(gt.size, ph.size)
    if n < 20:
        return None
    gt, ph = gt[:n], ph[:n]
    v = (gt > 0) & plan.voiced[:n]
    if int(v.sum()) < 20:
        return None
    st = lambda x: 12.0 * np.log2(np.maximum(x, 1e-3) / F0_REF_HZ)
    a, b = st(gt[v]), st(ph[v])          # semitones
    err = 100.0 * (b - a)                # cents
    off = float(np.median(err))
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-4 else float("nan")
    slope = float(np.polyfit(a, b, 1)[0]) if a.std() > 1e-4 else float("nan")
    voiced_gt = gt > 0
    agree = float((plan.voiced[:n] == voiced_gt).mean())
    reg_true = float(12.0 * np.log2(np.median(gt[voiced_gt]) / F0_REF_HZ))
    # word-level contour: the unit the user actually controls
    wp, wg = [], []
    for w in range(plan.n_words):
        s = int(plan.word_start[w]) * 4
        d = int(plan.word_dur[w]) * 4
        if d <= 0:
            continue
        g = gt[s:s + d]
        vv = g > 0
        pv = plan.voiced[s:s + d]
        if int(vv.sum()) < 4 or not pv.any():
            continue
        wp.append(float(np.median(plan.pitch_st[s:s + d][pv]) + plan.register_st))
        wg.append(float(np.median(st(g[vv]))))
    return {"frames": int(v.sum()), "corr": corr, "slope": slope,
            "abs_c": float(np.median(np.abs(err))),
            "off_c": off,
            "off_removed_c": float(np.median(np.abs(err - off))),
            "voicing_agree": agree,
            "register_true": reg_true,
            "register_plan": plan.register_st,
            "register_c": float((plan.register_st - reg_true) * 100.0),
            "word_plan": wp, "word_gt": wg}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filelist", default=DEFAULT_FILELIST)
    ap.add_argument("--limit", type=int, default=200,
                    help="clips to inspect (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-spec", action="store_true",
                    help="skip .spec/.f0: slots become speech, no VAD")
    ap.add_argument("--show", type=int, default=0,
                    help="print this many example word tables")
    args = ap.parse_args()

    path = _resolve(args.filelist)
    if not path.exists():
        print(f"no such filelist: {path}", file=sys.stderr)
        return 2
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    random.Random(args.seed).shuffle(lines)
    if args.limit:
        lines = lines[: args.limit]

    reasons = Counter()
    clips: list[dict] = []
    errors: list[tuple[str, str]] = []
    shown = 0

    for line in lines:
        parts = line.split("|")
        if len(parts) < 3:
            reasons["bad line"] += 1
            continue
        phone_p, codes_p = _resolve(parts[0]), _resolve(parts[1])
        cond_p = Path(str(codes_p)[: -len(".spec_codes")] + ".cond")
        if not cond_p.exists():
            reasons["no .cond"] += 1
            continue
        if not phone_p.exists() or not codes_p.exists():
            reasons["missing tensor"] += 1
            continue
        try:
            cond = torch.load(cond_p, map_location="cpu", weights_only=False)
            phones = _load(phone_p)
            codes = _load(codes_p)
            n = int(codes.shape[-1]) if int(codes.shape[0]) == 8 else int(codes.shape[0])
        except Exception as exc:  # noqa: BLE001
            reasons["unreadable"] += 1
            errors.append((str(cond_p), f"load: {exc}"))
            continue

        spec = f0 = None
        spec_p, f0_p = _siblings(codes_p)
        if not args.no_spec:
            if spec_p is not None:
                spec = _load(spec_p)
            if f0_p is not None:
                f0 = _load(f0_p)
            if spec is None:
                reasons["no stage-1 spec"] += 1

        try:
            plan = plan_from_cond(cond, phones, n, spec=spec)
        except PlanError as exc:
            msg = str(exc)
            if msg.startswith("word run"):
                key = "degenerate word run"
            elif "field lengths" in msg:
                key = "word/phone length mismatch"
            elif "no word fields" in msg:
                key = "no word fields"
            elif "stale" in msg:
                key = "stale .cond"
            else:
                key = msg.split(":")[0][:40]
            reasons[key] += 1
            errors.append((str(cond_p), msg))
            continue
        except Exception as exc:  # noqa: BLE001
            reasons["plan: other"] += 1
            errors.append((str(cond_p), f"plan: {exc}"))
            continue

        # partition / coverage are already enforced by plan.check(); re-assert
        # here so the validator fails loudly if that ever regresses.
        total = plan.lead_sil + int(plan.word_dur.sum()) + int(plan.pause_after.sum())
        if total != n or int(plan.phone_dur.sum()) != 4 * int(plan.word_dur.sum()):
            reasons["partition mismatch"] += 1
            errors.append((str(cond_p), f"partition {total} != {n}"))
            continue

        row = {
            "n_frames": n,
            "lead": plan.lead_sil,
            "tail": int(plan.pause_after[-1]),
            "pause_frames": int(plan.pause_after.sum()),
            "words": plan.n_words,
            "words_with_pause": int((plan.pause_after >= 2).sum()),
            "grouped": plan.grouped_words,
            "phones": plan.n_phones,
            "word_dur": plan.word_dur.astype(float),
            "has_spec": plan.has_spec,
        }
        if f0 is not None:
            pr = _pitch_report(plan, f0)
            if pr:
                row["pitch"] = pr
        if args.show and shown < args.show:
            shown += 1
            print(f"\n--- {Path(parts[0]).name}  T={n}  "
                  f"lead={plan.lead_sil} group={plan.grouped_words}")
            print(f"{'w':>3} {'start':>6} {'dur':>4} {'pause':>5} {'phones':>7} {'dB':>7}")
            for r in plan.word_table():
                print(f"{r['word']:>3} {r['start_f']:>6} {r['dur_f']:>4} "
                      f"{r['pause_after_f']:>5} {len(r['phones']):>7} {r['db']:>7.2f}")
        clips.append(row)

    print(f"\nfilelist   {path}")
    print(f"clips      {len(clips)} used, {len(lines)} sampled")
    if reasons:
        print("skipped    " + "  ".join(f"{k} {v}" for k, v in reasons.most_common()))
        for where, why in errors[:4]:
            print(f"  e.g. {Path(where).name}: {why}")

    if not clips:
        return 1

    T = sum(c["n_frames"] for c in clips)
    print(f"code frames {T}")
    print(f"has spec   {sum(c['has_spec'] for c in clips)}/{len(clips)}")
    lead = _stats([c["lead"] for c in clips])
    tail = _stats([c["tail"] for c in clips])
    pause_frac = sum(c["pause_frames"] for c in clips) / max(T, 1)
    wd = _stats(np.concatenate([c["word_dur"] for c in clips]))
    words = sum(c["words"] for c in clips)
    wpause = sum(c["words_with_pause"] for c in clips)
    grouped = sum(c["grouped"] for c in clips)
    print(f"\ndurations (code frames, 40 ms)")
    print(f"  word dur       median {wd['median']:.1f}  p10 {wd['p10']:.1f}  p90 {wd['p90']:.1f}")
    print(f"  lead silence   median {lead['median']:.1f}  p90 {lead['p90']:.1f}")
    print(f"  tail silence   (last word's pause) median {tail['median']:.1f}  p90 {tail['p90']:.1f}")
    print(f"  pause frames   {100 * pause_frac:.1f}% of all frames; "
          f"{100 * wpause / max(words, 1):.1f}% of words have a pause >= 2")
    print(f"  grouped words  {grouped} ({100 * grouped / max(words, 1):.1f}%) share a start frame")

    pr = [c["pitch"] for c in clips if "pitch" in c]
    if pr:
        def agg(key):
            return _stats([p[key] for p in pr if np.isfinite(p[key])])
        corr, slope, absc = agg("corr"), agg("slope"), agg("abs_c")
        offrem = _stats([p["off_removed_c"] for p in pr])
        reg = _stats([p["register_c"] for p in pr])
        agree = _stats([p["voicing_agree"] for p in pr])
        print(f"\npitch vs stage-1 .f0 ({len(pr)} clips, voiced frames)")
        print(f"  contour corr   median {corr['median']:.3f}   p10 {corr['p10']:.3f}")
        print(f"  slope          median {slope['median']:.3f}")
        print(f"  |error|        median {absc['median']:.0f} cents (absolute)")
        print(f"  after offset   median {offrem['median']:.0f} cents "
              f"(p90 {offrem['p90']:.0f})")
        print(f"  register error median {reg['median']:+.0f} cents")
        print(f"  voicing agree  median {100 * agree['median']:.1f}%")
        hi = sum(1 for p in pr if p["corr"] > 0.9)
        print(f"  clips corr>0.9 {hi}/{len(pr)}")
        wp = np.concatenate([p["word_plan"] for p in pr if p["word_plan"]])
        wg = np.concatenate([p["word_gt"] for p in pr if p["word_gt"]])
        if wp.size > 3:
            print(f"\nword-level pitch ({wp.size} words, the control unit)")
            print(f"  corr {np.corrcoef(wp, wg)[0, 1]:.3f}  "
                  f"slope {np.polyfit(wg, wp, 1)[0]:.3f}  "
                  f"|err| median {100 * np.median(np.abs(wp - wg)):.0f} cents")

    bad = sum(v for k, v in reasons.items() if k.startswith("partition"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
