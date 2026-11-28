"""Train the non-autoregressive plan renderer.

    python scratch/build_plan_cache.py \
        --filelists data/genshin_plus_extras/train_stage2.txt \
                    data/genshin_plus_extras/val_stage2.txt
    python train_renderer.py --epochs 40 --batch-size 16

The first command matters: without the cache the dataset has to read the
stage-1 ``.spec`` (0.5 MB/clip) for every plan.  ``--require-cache`` keeps a
run from silently falling back to that and crashing on a degenerate alignment.

What this optimises: masked code CE for the 8 spec and 2 pitch books, plus an
optional f0 L1 in semitones.  CE is a *training* signal only -- guidance is
judged later on plan adherence and WER, which this script does not compute (the
decode/eval path is deliberately separate so it can use the frozen stage-1
Generator and the ASR).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import shared
from dataset_plan import PlanDataset, collate_plans
from renderer import GuidedRenderer, renderer_loss, sample_codes
from audio_utils import codes_to_audio, original_wave, zq_to_audio
from stage2_wer import code_frame_samples, item_basename, load_sid_map, map_sids

ROOT = Path(__file__).resolve().parent
MODEL_KEYS = ("phone_ids", "phone_mask", "phone_index", "frame_feats", "sid",
              "frame_mask")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-filelist",
                    default="data/genshin_plus_extras/train_stage2.txt")
    ap.add_argument("--val-filelist",
                    default="data/genshin_plus_extras/val_stage2.txt")
    ap.add_argument("--cache-dir", default=str(ROOT / "plans"))
    ap.add_argument("--require-cache", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "renderer_v0"))
    ap.add_argument("--resume", default=None)

    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                    help="bf16 autocast on CUDA")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--max-frames", type=int, default=400,
                    help="word-aligned training window (code frames)")
    ap.add_argument("--val-max-frames", type=int, default=None,
                    help="0/None = full clip for validation")
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="stop after this many optimiser steps (smoke test)")

    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--phone-layers", type=int, default=4)
    ap.add_argument("--frame-layers", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--depth-ar-pitch", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--f0-aux", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--phone-aux", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="predict the plan's phone id per code frame")
    ap.add_argument("--phone-aux-weight", type=float, default=0.5)
    ap.add_argument("--zq-aux", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="continuous zq regression head (primary spec loss)")
    ap.add_argument("--zq-weight", type=float, default=1.0)
    ap.add_argument("--zq-dim", type=int, default=1024)
    ap.add_argument("--phone-attn-layers", type=int, default=1)
    ap.add_argument("--frame-attn-layers", type=int, default=2)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--spec-weight", type=float, default=1.0)
    ap.add_argument("--pitch-weight", type=float, default=1.0)
    ap.add_argument("--f0-weight", type=float, default=0.05)

    ap.add_argument("--audio-every", type=int, default=5,
                    help="log generated/oracle/original audio to TensorBoard every N epochs (0 = off)")
    ap.add_argument("--audio-items", type=int, default=2)
    ap.add_argument("--audio-temperature", type=float, default=0.9)
    ap.add_argument("--audio-top-k", type=int, default=20)
    ap.add_argument("--sid-map", default="speaker_map_genshin.json")

    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=1,
                    help="validate every N epochs")
    return ap


def make_loader(args, train: bool) -> DataLoader:
    ds = PlanDataset(
        args.train_filelist if train else args.val_filelist,
        max_frames=args.max_frames if train else args.val_max_frames,
        train=train,
        cache_dir=args.cache_dir,
        require_cache=args.require_cache,
        limit=args.limit_train if train else args.limit_val,
        seed=args.seed,
    )
    return DataLoader(
        ds, batch_size=args.batch_size if train else max(1, args.batch_size),
        shuffle=train, drop_last=train and len(ds) > 0,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=collate_plans,
        persistent_workers=args.num_workers > 0)


def to_device(batch: dict, device: str) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def zq_target(codes: torch.Tensor, codebooks) -> torch.Tensor:
    """``(B,8,T)`` spec codes -> ``(B,T,1024)`` sum of codebook embeddings."""
    z = None
    for q, emb in enumerate(codebooks):
        e = torch.nn.functional.embedding(codes[:, q, :], emb)
        z = e if z is None else z + e
    return z


def forward_loss(model, batch: dict, args, codebooks=None) -> tuple[dict, dict]:
    out = model(**{k: batch[k] for k in MODEL_KEYS},
                pitch_targets=batch["pitch_codes"])
    zqt = zq_target(batch["spec_codes"], codebooks) if (
        args.zq_aux and codebooks is not None) else None
    losses = renderer_loss(out, batch, spec_weight=args.spec_weight,
                           pitch_weight=args.pitch_weight,
                           f0_weight=args.f0_weight,
                           phone_weight=args.phone_aux_weight if args.phone_aux else 0.0,
                           zq_target=zqt,
                           zq_weight=args.zq_weight if args.zq_aux else 0.0)
    return out, losses


@torch.no_grad()
def evaluate(model, loader, args, device: str, codebooks=None) -> dict:
    model.eval()
    totals = {"val_loss": 0.0, "val_spec_ce": 0.0, "val_pitch_ce": 0.0,
              "val_f0_l1": 0.0, "val_spec_acc": 0.0, "val_pitch_acc": 0.0,
              "val_zq_l1": 0.0}
    n = 0
    for batch in loader:
        batch = to_device(batch, device)
        out, losses = forward_loss(model, batch, args, codebooks)
        mask = batch["frame_mask"]
        b = batch["batch_size"] if "batch_size" in batch else int(mask.shape[0])
        spec_ok = (out["spec_logits"].argmax(-1)
                   == batch["spec_codes"].permute(0, 2, 1)).float()
        spec_acc = float((spec_ok * mask.unsqueeze(-1)).sum()
                         / (mask.sum() * 8).clamp_min(1))
        p0 = out["pitch_logits"][0].argmax(-1)
        p_acc = float(((p0 == batch["pitch_codes"][:, 0]).float()
                       * mask).sum() / mask.sum().clamp_min(1))
        if "phone_logits" in out and "frame_phone" in batch:
            pm = batch["frame_phone"] >= 0
            if bool(pm.any()):
                phone_acc = float((out["phone_logits"].argmax(-1) ==
                                   batch["frame_phone"])[pm].float().mean())
                totals["val_phone_acc"] = totals.get("val_phone_acc", 0.0) + phone_acc * b
        totals["val_loss"] += float(losses["loss"]) * b
        totals["val_spec_ce"] += float(losses["spec_ce"]) * b
        totals["val_pitch_ce"] += float(losses["pitch_ce"]) * b
        totals["val_f0_l1"] += float(losses.get("f0_l1", 0.0)) * b
        totals["val_zq_l1"] += float(losses.get("zq_l1", 0.0)) * b
        totals["val_spec_acc"] += spec_acc * b
        totals["val_pitch_acc"] += p_acc * b
        n += b
    model.train()
    return {k: v / max(n, 1) for k, v in totals.items()} | {"val_items": n}


@torch.no_grad()
def log_audio(model, batch: dict, net_g, sr: int, sid_map, writer,
              step: int, args) -> None:
    """Log a few val clips to TensorBoard: generated, oracle, original."""
    model.eval()
    out = model(**{k: batch[k] for k in MODEL_KEYS},
                pitch_targets=None, temperature=args.audio_temperature,
                top_k=args.audio_top_k)
    gt = torch.cat([batch["spec_codes"], batch["pitch_codes"]], dim=1)
    sids = torch.tensor(map_sids(batch["sid"].tolist(), sid_map),
                        dtype=torch.long, device=batch["sid"].device)
    lengths = batch["lengths"].tolist()
    if "zq_pred" in out:
        wavs = zq_to_audio(net_g, out["zq_pred"], out["pitch_codes"], sids,
                           lengths=lengths)
    else:
        spec = sample_codes(out["spec_logits"], args.audio_temperature,
                            args.audio_top_k)
        codes = torch.cat([spec.permute(0, 2, 1), out["pitch_codes"]], dim=1)
        wavs = codes_to_audio(net_g, codes, sids, lengths=lengths)
    owavs = codes_to_audio(net_g, gt, sids, lengths=lengths)
    for i in range(min(args.audio_items, len(wavs))):
        name = Path(item_basename(batch["paths"][i])).stem
        tag = f"{int(batch['index'][i]):05d}_{name}"
        for kind, wav in (("gen", wavs[i]), ("oracle", owavs[i])):
            w = torch.from_numpy(wav.astype("float32")).unsqueeze(0)
            writer.add_audio(f"audio/{kind}/{tag}", w, global_step=step,
                             sample_rate=sr)
        orig = original_wave(batch["paths"][i])
        if orig is not None:
            writer.add_audio(f"audio/orig/{tag}",
                             torch.from_numpy(orig.astype("float32")).unsqueeze(0),
                             global_step=step, sample_rate=sr)
    model.train()


def save_ckpt(path: Path, model, opt, sched, epoch, step, best, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict() if opt else None,
        "scheduler": sched.state_dict() if sched else None,
        "epoch": epoch, "step": step, "best": best,
        "args": vars(args),
    }, tmp)
    tmp.replace(path)


def main() -> int:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    device = args.device
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    train_loader = make_loader(args, True)
    val_loader = make_loader(args, False)
    audio_batch, net_g, audio_sr, sid_map = None, None, 48000, {}
    codebooks = None
    if args.zq_aux or (args.audio_every and len(val_loader) > 0):
        net_g, gen_cfg = shared.build_stage1(device=device)
        audio_sr = int(gen_cfg.data.sampling_rate)
        if args.zq_aux:
            codebooks = [l.codebook.detach()
                         for l in net_g.codec.quantizer.codebook.layers]
    if args.audio_every and len(val_loader) > 0:
        audio_batch = to_device(next(iter(val_loader)), device)
        sid_map = load_sid_map(args.sid_map)
        print(f"audio logging: {args.audio_items} clips every "
              f"{args.audio_every} epoch(s) at {audio_sr} Hz")
    if len(train_loader) == 0:
        print("no training items -- build the plan cache or drop --require-cache")
        return 2
    print(f"train {len(train_loader.dataset)} items "
          f"({len(train_loader)} batches), val {len(val_loader.dataset)} items")

    model = GuidedRenderer(
        num_speakers=8192, width=args.width,
        phone_layers=args.phone_layers, frame_layers=args.frame_layers,
        dropout=args.dropout, depth_ar_pitch=args.depth_ar_pitch,
        f0_aux=args.f0_aux, phone_attn_layers=args.phone_attn_layers,
        frame_attn_layers=args.frame_attn_layers, nhead=args.nhead,
        phone_aux=args.phone_aux, zq_aux=args.zq_aux,
        zq_dim=args.zq_dim).to(device)
    print(f"model {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.98))
    steps_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum))
    total = max(1, args.epochs * steps_per_epoch)
    warm = min(args.warmup, total)

    def sched_fn(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        p = min(1.0, (step - warm) / max(1, total - warm))
        return 0.5 * (1.0 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, sched_fn)
    use_amp = bool(args.amp and device.startswith("cuda"))

    start_epoch, step, best = 0, 0, float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        if ck.get("optimizer"):
            opt.load_state_dict(ck["optimizer"])
        if ck.get("scheduler"):
            sched.load_state_dict(ck["scheduler"])
        start_epoch = int(ck.get("epoch", 0))
        step = int(ck.get("step", 0))
        best = float(ck.get("best", best))
        print(f"resumed {args.resume} at epoch {start_epoch}, step {step}")

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(ROOT / "logs" / out.name))
    except Exception:  # noqa: BLE001
        writer = None

    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    stop = False
    for epoch in range(start_epoch, args.epochs):
        for it, batch in enumerate(train_loader):
            batch = to_device(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=use_amp):
                _, losses = forward_loss(model, batch, args, codebooks)
            loss = losses["loss"] / args.grad_accum
            loss.backward()
            if (it + 1) % args.grad_accum == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   args.grad_clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0:
                    dt = time.time() - t0
                    f0v = float(losses["f0_l1"].detach()) if "f0_l1" in losses else 0.0
                    print(f"ep {epoch:02d} step {step:5d} "
                          f"loss {losses['loss'].detach().item():.4f} "
                          f"spec {losses['spec_ce'].detach().item():.4f} "
                          f"pitch {losses['pitch_ce'].detach().item():.4f} "
                          f"f0 {f0v:.2f} "
                          + (f"zq {losses['zq_l1'].detach().item():.3f} "
                             if "zq_l1" in losses else "")
                          + (f"phone {losses['phone_ce'].detach().item():.3f} "
                             if "phone_ce" in losses else "") +
                          f"lr {sched.get_last_lr()[0]:.2e} "
                          f"{step / max(dt, 1e-6):.2f} it/s")
                    if writer is not None:
                        for k, v in losses.items():
                            writer.add_scalar(
                                f"train/{k}",
                                float(v.detach()) if torch.is_tensor(v) else float(v),
                                step)
                        writer.add_scalar("train/lr", sched.get_last_lr()[0], step)
                if args.max_steps and step >= args.max_steps:
                    stop = True
                    break
        if (epoch + 1 - start_epoch) % max(1, args.val_every) == 0:
            metrics = evaluate(model, val_loader, args, device, codebooks)
            print(f"  val ep {epoch:02d} " + "  ".join(
                f"{k} {v:.4f}" for k, v in metrics.items()))
            if writer is not None:
                for k, v in metrics.items():
                    writer.add_scalar(k.replace("val_", "val/"), v, step)
                if (args.audio_every and audio_batch is not None and net_g is not None
                        and (epoch + 1) % args.audio_every == 0):
                    log_audio(model, audio_batch, net_g, audio_sr, sid_map,
                              writer, step, args)
            score = metrics["val_loss"]
            if score < best:
                best = score
                save_ckpt(out / "best.ckpt", model, opt, sched, epoch + 1,
                          step, best, args)
                print(f"  saved best.ckpt (val {best:.4f})")
        save_ckpt(out / "last.ckpt", model, opt, sched, epoch + 1, step,
                  best, args)
        if stop:
            break

    if writer is not None:
        writer.close()
    print(f"done: {step} steps in {time.time() - t0:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
