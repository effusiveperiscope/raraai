"""Sanity-check a stage-2 ``CodeTransformer`` checkpoint.

The stage-2 objective is easy to improve *without* using the phonemes at all:
the decoder only has to model acoustic continuity (the previous code frame),
which is a strong, always-available signal.  A run can therefore show a
falling loss and rising accuracy while the text conditioning channel is
completely dead -- the model is then an unconditional code LM conditioned on
the speaker, and no amount of extra data or parameters will fix it.

This script measures that directly, without listening to audio:

* **teacher-forced CE with shuffled / random phones.**  If ``CE(shuffled
  phones) - CE(real phones)`` is ~0 the decoder ignores the encoder memory.
* **encoder memory diagnostics.**  A pre-LN residual stream that grows
  unchecked ends up with one or two huge dimensions; the final LayerNorm then
  divides by a per-position std dominated by those dimensions, and every
  informative dimension is crushed to a fraction of its size.  The memory
  still *contains* the phoneme information, it is just scaled into the noise
  floor of the cross-attention keys.
* **cross-attention entropy.**  Uniform attention (entropy / log N ~= 1)
  means the decoder averages the memory positions instead of aligning to the
  current phoneme.  The script also reports the same statistic with the
  memory per-dimension standardised, i.e. what the attention *would* do if
  the scale collapse were removed.

Usage::

    python test_stage2_conditioning.py \
        --checkpoint checkpoints_stage2/transformer_v0.4/last.ckpt

Runs on CPU by default so it can be used while a training job owns the GPU;
pass ``--device cuda`` for a faster (but competing) run.
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import commons
from dataset_codes import CodeCollator, CodeDataset
from modeling.transformer import CodeTransformer


def load_model(checkpoint: str, config, device: str) -> CodeTransformer:
    """Build ``CodeTransformer`` from `config` and load `checkpoint`."""
    net = CodeTransformer(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(k.startswith("net.") for k in state):
        commons.load_submodule_prefix(net, "net.", state)
    else:
        commons.load_state_dict_mismatch(net, state)
    net.to(device)
    net.eval()
    return net


def memory_stats(net: CodeTransformer, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Scale / spread statistics of the phoneme encoder output.

    Args:
        net: Model in eval mode.
        batch: Collated batch (see ``dataset_codes.CodeCollator``).

    Returns:
        Dict with the mean per-position norm of the memory, the mean
        per-dimension std *over time* (how much the memory varies between
        phoneme positions — ~1.0 by construction when
        ``transformer.memory_norm == "temporal"``), the mean per-position norm
        of the residual stream before the final normalisation, and the largest
        per-dimension mean of that residual (a runaway residual dimension shows
        up here) with the normalisation gain applied to it.
    """
    # Capture the residual stream after the last encoder layer (i.e. the input
    # of the final normalisation, where a runaway dimension lives).
    blocks = net.encoder.encoder.layers
    residual: List[torch.Tensor] = []

    def hook(module, args, output, *extra):
        residual.append(output.detach() if torch.is_tensor(output)
                        else output[0].detach())

    handle = blocks[-1].register_forward_hook(hook)
    try:
        with torch.no_grad():
            memory, valid = net.encode(
                batch["phone_ids"], batch["phone_lengths"], batch["sid"]
            )
    finally:
        handle.remove()

    with_valid = valid.unsqueeze(-1).to(memory.dtype)
    per_pos_norm = memory.norm(dim=-1)
    n_valid = with_valid.sum(dim=1).clamp_min(1)
    norm = float((per_pos_norm * with_valid.squeeze(-1)).sum() / n_valid.sum())

    def _dim_stats(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-dimension mean / std over the valid positions of each sample."""
        means, stds = [], []
        for b in range(x.shape[0]):
            m = x[b][valid[b]]
            if m.shape[0] == 0:
                continue
            means.append(m.mean(0))
            stds.append(m.std(0))
        if not means:  # degenerate batch
            return torch.zeros(x.shape[-1]), torch.zeros(x.shape[-1])
        return torch.stack(means).mean(0), torch.stack(stds).mean(0)

    # Spread of the (post-LayerNorm) memory over time: how distinguishable two
    # phoneme positions are once they reach the cross-attention.
    _, mem_std = _dim_stats(memory)
    spread = float(mem_std.mean())

    # Runaway dimension of the pre-normalisation residual stream.
    if residual:
        res_mean, _ = _dim_stats(residual[0])
        outlier_dim = int(res_mean.abs().argmax())
        outlier_mean = float(res_mean[outlier_dim])
        if getattr(net.encoder, "memory_gain", None) is not None:
            outlier_weight = float(net.encoder.memory_gain[outlier_dim].detach())
        elif net.encoder.encoder.norm is not None:
            outlier_weight = float(net.encoder.encoder.norm.weight[outlier_dim].detach())
        else:
            outlier_weight = float("nan")
        x = residual[0]
        mask = valid.unsqueeze(-1).to(x.dtype)
        residual_norm = float((x.norm(dim=-1) * mask.squeeze(-1)).sum() / mask.sum())
    else:
        outlier_dim, outlier_mean, outlier_weight, residual_norm = -1, 0.0, 0.0, 0.0

    return {
        "norm": norm,
        "spread": spread,
        "residual_norm": residual_norm,
        "outlier_dim": outlier_dim,
        "outlier_mean": outlier_mean,
        "outlier_ln_weight": outlier_weight,
        "hidden_size": memory.shape[-1],
    }


def _cross_attention_entropy(
    module: torch.nn.MultiheadAttention,
    query: torch.Tensor,
    memory: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor],
    standardize_keys: bool = False,
) -> Tuple[float, float]:
    """Normalised attention entropy and peak weight of one decoder layer.

    Args:
        module: The layer's ``multihead_attn`` (source of the projections).
        query: ``norm2`` output fed to the attention, ``(B, L, E)``.
        memory: Encoder memory used as key/value, ``(B, N, E)``.
        key_padding_mask: ``(B, N)`` bool, True == padded.
        standardize_keys: Per-dimension standardise the memory keys first.
            This is the *counterfactual* "what would the attention look like
            if the memory scale collapse were removed"; it does not change the
            values, so the decode output is unaffected by the flag.

    Returns:
        ``(entropy / log(n_valid), max attention weight)``.
    """
    emb = module.embed_dim
    heads = module.num_heads
    head_dim = emb // heads
    weight, bias = module.in_proj_weight.detach(), module.in_proj_bias
    if bias is not None:
        bias = bias.detach()

    q = F.linear(query, weight[:emb], None if bias is None else bias[:emb])
    q = q.view(query.shape[0], query.shape[1], heads, head_dim).transpose(1, 2)

    keys = memory
    if standardize_keys:
        mu = memory.mean(dim=1, keepdim=True)
        sd = memory.std(dim=1, keepdim=True).clamp_min(1e-6)
        keys = (memory - mu) / sd
    k = F.linear(keys, weight[emb:2 * emb], None if bias is None else bias[emb:2 * emb])
    k = k.view(keys.shape[0], keys.shape[1], heads, head_dim).transpose(1, 2)

    att = (q @ k.transpose(-1, -2)) / math.sqrt(head_dim)
    if key_padding_mask is not None:
        att = att.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
    att = att.softmax(dim=-1)

    entropy = -(att.clamp_min(1e-9).log() * att).sum(-1)  # (B, H, L)
    if key_padding_mask is not None:
        n_valid = (~key_padding_mask).sum(-1).float().clamp_min(1)
    else:
        n_valid = att.new_full((att.shape[0],), float(att.shape[-1]))
    entropy = entropy / n_valid[:, None, None].log()
    return float(entropy.mean()), float(att.max())


def cross_attention_stats(
    net: CodeTransformer, batch: Dict[str, torch.Tensor]
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, torch.Tensor]]:
    """Cross-attention entropy per decoder layer, as-trained vs counterfactual.

    Runs one teacher-forced forward pass with hooks on every decoder layer's
    cross-attention and reports, per layer, the attention entropy normalised
    by ``log(n_valid)`` (1.0 == perfectly uniform attention over the phoneme
    sequence) plus the same statistic with per-dimension standardised keys.

    Args:
        net: Model in eval mode.
        batch: Collated batch.

    Returns:
        ``(stats, loss)`` where ``stats`` maps ``layer{i}`` to
        ``{"entropy", "entropy_standardised", "max"}`` and ``loss`` is the
        dict returned by ``CodeTransformer.compute_batch_loss``.
    """
    captured: List[Tuple[int, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = []

    def make_hook(layer_idx: int):
        def hook(module, args, kwargs, *extra):
            captured.append((layer_idx, args[0].detach(), args[1].detach(),
                             kwargs.get("key_padding_mask")))
        return hook

    handles = [
        layer.multihead_attn.register_forward_hook(make_hook(idx), with_kwargs=True)
        for idx, layer in enumerate(net.decoder.decoder.layers)
    ]
    try:
        with torch.no_grad():
            loss = net.compute_batch_loss(batch, epoch=None)
    finally:
        for handle in handles:
            handle.remove()

    stats: Dict[str, Dict[str, float]] = {}
    for idx, query, memory, key_mask in captured:
        entropy, peak = _cross_attention_entropy(
            net.decoder.decoder.layers[idx].multihead_attn, query, memory, key_mask)
        entropy_std, _ = _cross_attention_entropy(
            net.decoder.decoder.layers[idx].multihead_attn, query, memory,
            key_mask, standardize_keys=True)
        stats[f"layer{idx}"] = {
            "entropy": entropy,
            "entropy_standardised": entropy_std,
            "max": peak,
        }
    return stats, loss


def _phone_variants(
    batch: Dict[str, torch.Tensor], generator: torch.Generator
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Build the (phone_ids, phone_lengths) variants used for the CE probe."""
    phones, lengths = batch["phone_ids"], batch["phone_lengths"]
    variants = {"real": (phones, lengths)}
    if phones.shape[0] > 1:
        variants["shuffled"] = (phones.roll(1, 0), lengths.roll(1, 0))
    variants["random"] = (
        torch.randint(1, 1024, phones.shape, generator=generator),
        lengths,
    )
    return variants


def conditioning_report(
    net: CodeTransformer,
    batches: List[Dict[str, torch.Tensor]],
    generator: torch.Generator,
) -> Dict[str, Dict[str, float]]:
    """Teacher-forced CE (and accuracy) for real / shuffled / random phonemes.

    Args:
        net: Model in eval mode.
        batches: Collated batches.
        generator: CPU generator for the random-phoneme variant.

    Returns:
        ``{variant: {"loss", "spec_acc", "pitch_acc", "code_losses"}}``
        averaged over `batches`.
    """
    rows: Dict[str, List[Dict[str, float]]] = {}
    with torch.no_grad():
        for batch in batches:
            for name, (phones, lengths) in _phone_variants(batch, generator).items():
                out = net.compute_batch_loss(
                    {**batch, "phone_ids": phones, "phone_lengths": lengths},
                    epoch=None,
                )
                rows.setdefault(name, []).append({
                    "loss": float(out["loss"]),
                    "spec_acc": float(out["spec_acc"]),
                    "pitch_acc": float(out["pitch_acc"]),
                    "code_losses": [float(v) for v in out["code_losses"]],
                })
    report: Dict[str, Dict[str, float]] = {}
    for name, values in rows.items():
        n = len(values)
        report[name] = {
            "loss": sum(v["loss"] for v in values) / n,
            "spec_acc": sum(v["spec_acc"] for v in values) / n,
            "pitch_acc": sum(v["pitch_acc"] for v in values) / n,
            "code_losses": [
                sum(v["code_losses"][k] for v in values) / n
                for k in range(len(values[0]["code_losses"]))
            ],
        }
    return report


def verdict(report: Dict[str, Dict[str, float]]) -> str:
    """Short human-readable judgement of the phoneme-conditioning delta."""
    real = report["real"]["loss"]
    shuffled = report.get("shuffled", report["real"])["loss"]
    delta = shuffled - real
    if delta < 0.01:
        return (f"DEAD: wrong phonemes change the loss by only {delta:+.4f} nats "
                "(< 0.01). The decoder ignores the encoder memory, so the run "
                "is an unconditional code LM conditioned on the speaker only.")
    if delta < 0.05:
        return (f"WEAK: wrong phonemes cost only {delta:+.4f} nats (< 0.05). "
                "Text conditioning exists but carries little information.")
    return f"HEALTHY: wrong phonemes cost {delta:+.4f} nats."


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose whether a stage-2 checkpoint actually uses the "
                    "phoneme conditioning.")
    parser.add_argument("--checkpoint",
                        default="checkpoints_stage2/transformer_v0.4/last.ckpt")
    parser.add_argument("--config", default="configs/transformer.yaml")
    parser.add_argument("--filelist",
                        default="data/genshin_full_stage2/val_stage2.txt")
    parser.add_argument("--batches", type=int, default=4,
                        help="number of validation batches to average over")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_frames", type=int, default=128,
                        help="code frames per item (0 = whole clip)")
    parser.add_argument("--max_phone_length", type=int, default=512)
    parser.add_argument("--device", default="cpu",
                        help="cpu (default, leaves the GPU to training) or cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    config = OmegaConf.load(args.config)
    net = load_model(args.checkpoint, config, args.device)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"checkpoint: {args.checkpoint}  epoch={state.get('epoch')}  "
          f"global_step={state.get('global_step')}")

    dataset = CodeDataset(
        args.filelist,
        is_train=False,
        max_frames=args.max_frames or None,
        max_phone_length=args.max_phone_length,
    )
    collator = CodeCollator(config=config)
    total = min(len(dataset), args.batches * args.batch_size)
    batches = [
        collator([dataset[i] for i in range(start, min(start + args.batch_size, total))])
        for start in range(0, total, args.batch_size)
    ]
    n_clips = sum(b["phone_ids"].shape[0] for b in batches)
    n_frames = sum(int(b["code_lengths"].sum()) for b in batches)
    print(f"clips: {n_clips}  code frames: {n_frames}  ({n_frames / 25.0:.1f} s)")

    # ------------------------------------------------------------- memory ---
    stats = [memory_stats(net, b) for b in batches]
    norm = sum(s["norm"] for s in stats) / len(stats)
    spread = sum(s["spread"] for s in stats) / len(stats)
    res_norm = sum(s["residual_norm"] for s in stats) / len(stats)
    worst = max(stats, key=lambda s: abs(s["outlier_mean"]))
    hidden = worst["hidden_size"]
    print("\nencoder memory")
    print(f"  residual |x| / pos    : {res_norm:8.3f}   "
          f"(pre-final-LayerNorm; ~{math.sqrt(hidden):.0f} is a healthy scale)")
    print(f"  |memory| per position : {norm:8.3f}   "
          f"(LayerNorm output is ~{math.sqrt(hidden):.1f} at init)")
    print(f"  spread over time      : {spread:8.4f}   "
          f"(per-dimension std over positions; ~1.0 is healthy, "
          f"<0.05 means the memory is near-constant)")
    print(f"  largest dim           : #{worst['outlier_dim']} "
          f"mean {worst['outlier_mean']:+.1f}  "
          f"norm gain {worst['outlier_ln_weight']:.4f}")

    # ----------------------------------------------------- cross-attention ---
    attn, _ = cross_attention_stats(net, batches[0])
    entropies = [v["entropy"] for v in attn.values()]
    entropies_std = [v["entropy_standardised"] for v in attn.values()]
    print("\ncross-attention (entropy / log n_phones; 1.0 = uniform)")
    for name, value in attn.items():
        print(f"  {name:8s} as-trained {value['entropy']:.3f}   "
              f"standardised keys {value['entropy_standardised']:.3f}   "
              f"peak weight {value['max']:.2f}")
    print(f"  mean     as-trained {sum(entropies) / len(entropies):.3f}   "
          f"standardised {sum(entropies_std) / len(entropies_std):.3f}")

    # ---------------------------------------------------------- text probe --
    report = conditioning_report(net, batches, generator)
    print("\nteacher-forced loss by phoneme variant")
    for name in ("real", "shuffled", "random"):
        if name not in report:
            continue
        r = report[name]
        delta = "" if name == "real" else f"  ({r['loss'] - report['real']['loss']:+.4f})"
        print(f"  {name:9s} loss {r['loss']:.4f}{delta}   "
              f"spec_acc {r['spec_acc']:.4f}  pitch_acc {r['pitch_acc']:.4f}")
        print("            per codebook: "
              + " ".join(f"{v:.2f}" for v in r["code_losses"]))

    print("\nper codebook loss (real / shuffled)")
    other = report.get("shuffled", report["real"])["code_losses"]
    for k, (a, b) in enumerate(zip(report["real"]["code_losses"], other)):
        name = f"spec{k}" if k < net.spec_codebook_num else f"pitch{k}"
        print(f"  {name:7s} {a:.3f} / {b:.3f}  delta {b - a:+.4f}")

    print("\nverdict")
    print("  " + verdict(report))
    print("  Reference: these codebooks have a marginal entropy of ~3.9-4.0 "
          "nats (spec) and ~4.1/4.0 (pitch), and a first-order 'copy the "
          "previous frame' model already reaches ~3.3 nats on spec0. A real "
          "loss well below that (and a clear gap to the shuffled-phone loss) "
          "is the bar for text conditioning to be doing any work.")


if __name__ == "__main__":
    main()


