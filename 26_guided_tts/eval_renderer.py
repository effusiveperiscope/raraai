"""Decode a trained renderer and measure WER + plan adherence.

This is the only script that can answer the project's actual question.  Code CE
is reported by training but cannot see prosody; here we synthesise audio through
the frozen stage-1 Generator and compare:

* **intelligibility** -- ASR WER against the source transcript, plus an oracle
  decode of the ground-truth codes through the same vocoder (the codec floor);
* **pitch adherence** -- generated f0 vs the plan's contour (correlation and
  cents), and vs the original ``.f0`` (how much micro-prosody was recovered);
* **level adherence** -- generated per-word RMS vs the plan's ``word_db``;
* **duration** -- exact by construction (``len(codes) = plan n_frames``); the
  code tensor is never lengthened or shortened here.

``--register-shift`` and ``--level-shift`` perturb the plan at inference without
changing timing; sweeping them later is the guidance test (a channel that does
not move f0/energy is decoration).

Usage::

    python eval_renderer.py --checkpoint checkpoints/renderer_v1/best.ckpt \
        --num-items 120 --device cuda:0
    # fast check without ASR
    python eval_renderer.py --checkpoint ... --num-items 12 --no-asr
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import shared  # noqa: E402  (adds ../25_transformer to sys.path)
from dataset_plan import PlanDataset, collate_plans  # noqa: E402
from renderer import GuidedRenderer, sample_codes  # noqa: E402
from audio_utils import codes_to_audio, original_wave  # noqa: E402
from plan import F0_REF_HZ  # noqa: E402
from stage2_wer import (  # noqa: E402
    Asr, code_frame_samples, item_basename, load_references, load_sid_map,
    map_sids, pick_indices, resolve_source_filelists, to_16k, wer_diagnostics,
    wer_summary,
)

MODEL_KEYS = ("phone_ids", "phone_mask", "phone_index", "frame_feats", "sid",
              "frame_mask")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def load_model(path: str, device: str) -> tuple[GuidedRenderer, int]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck.get("args", {}) or {}
    model = GuidedRenderer(
        num_speakers=8192,
        width=int(a.get("width", 512)),
        phone_layers=int(a.get("phone_layers", 4)),
        frame_layers=int(a.get("frame_layers", 6)),
        dropout=float(a.get("dropout", 0.1)),
        depth_ar_pitch=bool(a.get("depth_ar_pitch", True)),
        f0_aux=bool(a.get("f0_aux", True)),
    )
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    step = int(ck.get("step", 0) or 0)
    print(f"loaded {path} (step {step})")
    return model, step


def _st(hz: np.ndarray) -> np.ndarray:
    return 12.0 * np.log2(np.maximum(hz, 1e-3) / F0_REF_HZ)


def _f0_compare(a_hz: np.ndarray, b_hz: np.ndarray) -> dict | None:
    m = (a_hz > 0) & (b_hz > 0)
    if int(m.sum()) < 20:
        return None
    a, b = _st(a_hz[m]), _st(b_hz[m])
    err = 100.0 * (b - a)
    off = float(np.median(err))
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-4 else float("nan")
    return {"n": int(m.sum()), "corr": corr,
            "abs_c": float(np.median(np.abs(err))),
            "off_c": off,
            "off_removed_c": float(np.median(np.abs(err - off)))}


def _word_db_from_wave(wave: np.ndarray, starts: np.ndarray, durs: np.ndarray,
                       hop: int) -> np.ndarray:
    out = np.full(len(starts), np.nan, dtype=np.float32)
    for w, (s, d) in enumerate(zip(starts, durs)):
        seg = wave[int(s) * hop: int(s + d) * hop]
        if seg.size:
            rms = float(np.sqrt(np.mean(np.square(seg, dtype=np.float64)) + 1e-12))
            out[w] = 20.0 * np.log10(max(rms, 1e-6))
    return out


def _level_compare(plan_db: np.ndarray, gen_db: np.ndarray) -> dict | None:
    ok = np.isfinite(plan_db) & np.isfinite(gen_db)
    if int(ok.sum()) < 4:
        return None
    a, b = plan_db[ok] - np.median(plan_db[ok]), \
        gen_db[ok] - np.median(gen_db[ok])
    if a.std() < 1e-4 or b.std() < 1e-4:
        return None
    corr = float(np.corrcoef(a, b)[0, 1])
    slope = float(np.polyfit(a, b, 1)[0])
    return {"n": int(ok.sum()), "corr": corr, "slope": slope,
            "mae_db": float(np.mean(np.abs(a - b)))}


def _agg(rows: list[dict], key: str) -> dict:
    vals = [r[key] for r in rows if key in r and np.isfinite(r[key])]
    if not vals:
        return {}
    a = np.asarray(vals, dtype=np.float64)
    return {"n": int(a.size), "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)),
            "p90": float(np.percentile(a, 90))}


def _generated_codes(model, batch, args) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        out = model(**{k: batch[k] for k in MODEL_KEYS},
                    pitch_targets=None, temperature=args.temperature,
                    top_k=args.top_k)
    spec = sample_codes(out["spec_logits"], args.temperature, args.top_k)
    pitch = out["pitch_codes"]
    codes = torch.cat([spec.permute(0, 2, 1), pitch], dim=1)     # (B,10,T)
    return codes, out["pitch_logits"][0]


@torch.no_grad()
def _decode(net_g, codes: torch.Tensor, sids: torch.Tensor):
    """``(B,10,T)`` -> ``(audio list, f0 (B,4T))`` through the frozen decoder."""
    return codes_to_audio(net_g, codes, sids, return_f0=True)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--filelist", default="data/genshin_plus_extras/val_stage2.txt")
    ap.add_argument("--cache-dir", default=str(ROOT / "plans"))
    ap.add_argument("--require-cache", action="store_true")
    ap.add_argument("--num-items", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--register-shift", type=float, default=0.0,
                    help="semitones added to the plan's register at inference")
    ap.add_argument("--level-shift", type=float, default=0.0,
                    help="dB added to every word's plan level")
    ap.add_argument("--oracle", action="store_true",
                    help="also decode the ground-truth codes (codec floor)")
    ap.add_argument("--no-asr", action="store_true")
    ap.add_argument("--save-audio", default=None,
                    help="write gen/oracle/original wavs + manifest.jsonl here")
    ap.add_argument("--audio-logdir", default=None,
                    help="write gen/oracle/original clips to a TensorBoard log dir")
    ap.add_argument("--asr-model", default="Qwen/Qwen3-ASR-0.6B")
    ap.add_argument("--sid-map", default="speaker_map_genshin.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.greedy:
        args.temperature = 0.0
        args.top_k = 0
    torch.manual_seed(args.seed)

    ds = PlanDataset(args.filelist, max_frames=None, train=False,
                     cache_dir=args.cache_dir, require_cache=args.require_cache)
    refs = load_references(resolve_source_filelists(None))
    avail = [i for i in range(len(ds))
             if item_basename(str(ds.items[i]["phone"])) in refs]
    pick = pick_indices(avail, args.num_items)
    sub = Subset(ds, pick)
    loader = DataLoader(sub, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_plans)
    print(f"eval: {len(sub)} items with references of {len(ds)} in the list")

    model, ckpt_step = load_model(args.checkpoint, args.device)
    net_g, cfg = shared.build_stage1(device=args.device)
    sr = int(cfg.data.sampling_rate)
    hop = code_frame_samples(net_g)
    sid_map = load_sid_map(args.sid_map)
    asr = None if args.no_asr else Asr(args.asr_model, args.device, "English")
    writer = None
    if args.audio_logdir:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=args.audio_logdir)
            print(f"tensorboard audio -> {args.audio_logdir} (step {ckpt_step})")
        except Exception as exc:  # noqa: BLE001
            print(f"tensorboard unavailable: {exc}")

    rows: list[dict] = []
    oracle_rows: list[dict] = []
    manifest: list[dict] = []
    pairs: list[tuple[str, str]] = []
    oracle_pairs: list[tuple[str, str]] = []

    for batch in loader:
        paths = batch["paths"]
        batch = {k: (v.to(args.device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        # plan perturbations (guidance test): only the conditioning changes,
        # every duration/phone/segment stays the same
        if args.register_shift:
            batch["frame_feats"][..., 3] += args.register_shift
        if args.level_shift:
            batch["frame_feats"][..., 2] += args.level_shift

        codes, _ = _generated_codes(model, batch, args)
        gt_codes = torch.cat([batch["spec_codes"], batch["pitch_codes"]], dim=1)
        sids = torch.tensor(map_sids(batch["sid"].tolist(), sid_map),
                            dtype=torch.long, device=args.device)

        wavs, f0g = _decode(net_g, codes, sids)
        if args.oracle:
            owavs, of0 = _decode(net_g, gt_codes, sids)

        hyps = asr.transcribe(to_16k(wavs, sr)) if asr else [""] * len(wavs)
        oh = asr.transcribe(to_16k(owavs, sr)) if (asr and args.oracle) else [""] * len(wavs)

        for i, path in enumerate(paths):
            n = int(batch["lengths"][i])
            ref = refs[item_basename(path)]
            pairs.append((ref, hyps[i]))
            plan_hz = F0_REF_HZ * np.exp2(
                (float(batch["register_st"][i]) + args.register_shift
                 + batch["pitch_st"][i].cpu().numpy()) / 12.0)
            plan_v = batch["voiced"][i].cpu().numpy()
            plan_hz = np.where(plan_v, plan_hz, 0.0)[: 4 * n]
            gen = f0g[i, : 4 * n]
            gt = batch["f0"][i].cpu().numpy()[: 4 * n]

            wdb = batch["word_db"][i].cpu().numpy() + args.level_shift
            wst = batch["word_start"][i].cpu().numpy()
            wdu = batch["word_dur"][i].cpu().numpy()
            gen_db = _word_db_from_wave(wavs[i], wst, wdu, hop)

            row = {"index": int(batch["index"][i]), "path": path,
                   "n_frames": n, "hyp": hyps[i], "ref": ref,
                   "plan_vs_gen": _f0_compare(plan_hz, gen),
                   "gt_vs_gen": _f0_compare(gt, gen),
                   "level": _level_compare(wdb, gen_db)}
            rows.append(row)

            if args.save_audio or args.audio_logdir:
                import soundfile as sf

                stem = Path(item_basename(path)).stem
                tag = f"{int(batch['index'][i]):05d}_{stem}"
                orig = original_wave(path)

                if args.save_audio:
                    d = Path(args.save_audio)
                    d.mkdir(parents=True, exist_ok=True)

                    def _save(name, wav):
                        w = np.clip(np.nan_to_num(np.asarray(wav, dtype=np.float32)),
                                    -1.0, 1.0)
                        sf.write(str(d / name), w, sr, subtype="PCM_16")

                    _save(f"{tag}_gen.wav", wavs[i])
                    if args.oracle:
                        _save(f"{tag}_oracle.wav", owavs[i])
                    if orig is not None:
                        _save(f"{tag}_orig.wav", orig)

                if writer is not None:
                    def _log(name, wav):
                        w = np.clip(np.nan_to_num(np.asarray(wav, dtype=np.float32)),
                                    -1.0, 1.0)
                        writer.add_audio(name, torch.from_numpy(w).unsqueeze(0),
                                         global_step=ckpt_step, sample_rate=sr)

                    _log(f"gen/{tag}", wavs[i])
                    if args.oracle:
                        _log(f"oracle/{tag}", owavs[i])
                    if orig is not None:
                        _log(f"orig/{tag}", orig)

                manifest.append({"tag": tag, "ref": ref, "hyp": hyps[i],
                                 "path": path, "n_frames": n,
                                 "f0_plan_corr": (row["plan_vs_gen"] or {}).get("corr"),
                                 "f0_gt_corr": (row["gt_vs_gen"] or {}).get("corr"),
                                 "level_corr": (row["level"] or {}).get("corr")})
            if asr and args.oracle:
                oracle_pairs.append((ref, oh[i]))
            if args.oracle:
                oracle_rows.append({
                    "index": int(batch["index"][i]),
                    "gt_vs_oracle": _f0_compare(gt, of0[i, : 4 * n]),
                    "level_oracle": _level_compare(
                        wdb, _word_db_from_wave(owavs[i], wst, wdu, hop)),
                })

    # ---- report ---------------------------------------------------------- #
    print(f"\nitems {len(rows)}")
    if asr:
        s = wer_summary(pairs)
        print(f"WER generated {100 * s['wer']:.2f}%  "
              f"S{int(s['sub'])} D{int(s['del'])} I{int(s['ins'])}")
        if oracle_pairs:
            o = wer_summary(oracle_pairs)
            print(f"WER oracle    {100 * o['wer']:.2f}%  (codec floor)")

    for key, label in (("plan_vs_gen", "plan f0 vs generated"),
                       ("gt_vs_gen", "true f0 vs generated")):
        vals = [r[key] for r in rows if r[key] and "corr" in r[key]]
        if vals:
            print(f"\n{label} ({len(vals)} clips)")
            for m in ("corr", "abs_c", "off_removed_c"):
                a = _agg(vals, m)
                if a:
                    print(f"  {m:14s} median {a['median']:.3f}  "
                          f"p10 {a['p10']:.3f}  p90 {a['p90']:.3f}")
    regs = [r["plan_vs_gen"]["off_c"] for r in rows
            if r["plan_vs_gen"] and "off_c" in r["plan_vs_gen"]]
    if regs:
        print(f"  pitch offset   median {np.median(regs):+.0f} cents")

    lv = [r["level"] for r in rows if r["level"]]
    if lv:
        print(f"\nword level, plan vs generated ({len(lv)} clips)")
        for m in ("corr", "slope", "mae_db"):
            a = _agg(lv, m)
            if a:
                print(f"  {m:14s} median {a['median']:.3f}  "
                      f"p10 {a['p10']:.3f}  p90 {a['p90']:.3f}")

    if oracle_rows:
        vals = [r["gt_vs_oracle"] for r in oracle_rows
                if r["gt_vs_oracle"] and "corr" in r["gt_vs_oracle"]]
        if vals:
            print(f"\ncodec ceiling, true f0 vs oracle decode ({len(vals)} clips)")
            for m in ("corr", "abs_c", "off_removed_c"):
                a = _agg(vals, m)
                if a:
                    print(f"  {m:14s} median {a['median']:.3f}")

    if args.save_audio:
        d = Path(args.save_audio)
        (d / "manifest.jsonl").write_text(
            "\n".join(json.dumps(m) for m in manifest) + "\n")
        print(f"\naudio {len(manifest)} items -> {d}")

    if writer is not None:
        writer.flush()
        writer.close()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "checkpoint": args.checkpoint, "args": vars(args),
            "items": len(rows), "rows": rows,
            "oracle_rows": oracle_rows,
        }, indent=1) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
