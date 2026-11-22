"""LoRA adapters for stage-2 fine-tuning (``--finetune``).

The problem this exists for: a full stage-2 fine-tune of ``CodeTransformer``
(136 M parameters) on a small corpus overfits within a few thousand steps --
measured on ``transformer_mlp_v1`` (41 869 clips, 25 speakers), train CE fell
2.35 -> 1.96 while val CE turned around at the val minimum (2.25 at ~2.6 k
steps) and rose to 2.36 by 15.7 k, with ``val_acc`` 0.334 -> 0.317.  Freezing
everything except a low-rank update keeps the pretrained model (and the whole
code space) fixed and restricts the fine-tune to a ~2 % subspace.

What is adapted
---------------

``LoRALinear`` replaces an existing ``nn.Linear`` in the module tree, adopting
its weight/bias parameters (frozen) and adding the low-rank pair:

* encoder + decoder FFN (``linear1`` / ``linear2``), i.e. every block,
* the decoder self-attention ``out_proj`` and its fused qkv projection
  (``in_proj_weight``) -- qkv only on ``RotaryMultiheadAttention``, which
  computes its own projections, so the delta is three lines there.

The stock ``nn.MultiheadAttention`` used by the *encoder* self-attention and the
*decoder cross-attention* is deliberately not adapted.  It never calls its
``out_proj`` as a module: ``F.multi_head_attention_forward`` receives
``self.out_proj.weight`` / ``.bias`` as plain tensors and its fused qkv weight
the same way, so a ``LoRALinear`` wrapper there would be inert in eval **and**
training (measured: zero difference on the module output), and an adapter that
a merge cannot fold into a weight is a silent-correctness trap.  Adapting one
would mean reimplementing ``nn.MultiheadAttention.forward`` (the encoder layer
also has a fused eval path that bypasses ``self_attn`` entirely).  If a run
underfits without them, that is the change to make -- not this one.

``lora_B`` is zero-initialised, so an injected model is *exactly* its source
function: ``--transfer_from`` reproduces the base checkpoint bit-for-bit and
the adapters have to earn their effect.  ``merge_lora`` folds the deltas back
into the base weights and returns a state dict with no adapter keys, which turns
a fine-tuned run into an ordinary checkpoint that every existing reader
understands (``scratch/merge_lora.py``).

Contract (see AGENTS.md §6.23): enabling this changes the parameter *set*, so a
plain checkpoint cannot be ``--resume_from``'d into a LoRA run (Lightning's
strict load would miss every ``lora_*`` key).  Use ``--transfer_from`` with a
fresh ``exp_name``; the adapters start at zero.  The reverse direction (a LoRA
checkpoint into a model without adapters) is fatal, because dropping trained
adapters silently evaluates the *base* model.
"""

from __future__ import annotations

import math
from typing import Dict, Iterator, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# What ``finetune.targets`` accepts.  ``decoder_attn`` / ``decoder_qkv`` need
# ``transformer.rope: true`` (they adapt the rotary self-attention); injection
# raises if a requested target matches no module, so a config cannot silently
# train fewer adapters than it names.  See the module docstring for why the
# stock attention modules are not adapted.
TARGETS = ("encoder_ffn", "decoder_ffn", "decoder_attn", "decoder_qkv")
DEFAULT_TARGETS = TARGETS

DEFAULTS = {
    "rank": 16,
    "alpha": 32.0,
    "dropout": 0.05,
    "train_spk_emb": True,
    "train_memory_norm": False,
    "lr": 5.0e-4,
    "weight_decay": 0.0,
}


def _get(cfg, key, default):
    """Tolerant read from a dict / OmegaConf node / namespace."""
    if cfg is None:
        return default
    try:
        if key in cfg:
            v = cfg[key]
            if v is not None:
                return v
    except Exception:
        pass
    v = getattr(cfg, key, None)
    return default if v is None else v


def resolve_config(cfg) -> Dict:
    """Normalise a ``finetune`` config block into plain Python values."""
    targets = _get(cfg, "targets", None)
    if targets is None:
        targets = list(DEFAULT_TARGETS)
    else:
        targets = [str(t) for t in targets]
        unknown = [t for t in targets if t not in TARGETS]
        if unknown:
            raise ValueError(
                f"finetune.targets has unknown entr{'y' if len(unknown) == 1 else 'ies'} "
                f"{unknown}; valid: {list(TARGETS)}")
    rank = int(_get(cfg, "rank", DEFAULTS["rank"]))
    if rank <= 0:
        raise ValueError(f"finetune.rank must be > 0, got {rank}")
    alpha = float(_get(cfg, "alpha", DEFAULTS["alpha"]))
    dropout = float(_get(cfg, "dropout", DEFAULTS["dropout"]))
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"finetune.dropout must be in [0, 1), got {dropout}")
    return {
        "targets": targets,
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "train_spk_emb": bool(_get(cfg, "train_spk_emb",
                                   DEFAULTS["train_spk_emb"])),
        "train_memory_norm": bool(_get(cfg, "train_memory_norm",
                                       DEFAULTS["train_memory_norm"])),
        "lr": float(_get(cfg, "lr", DEFAULTS["lr"])),
        "weight_decay": float(_get(cfg, "weight_decay",
                                   DEFAULTS["weight_decay"])),
        "cosine_anneal_end": _get(cfg, "cosine_anneal_end", None),
    }


class LoRALinear(nn.Module):
    """Frozen ``nn.Linear`` weights plus a low-rank update ``scale * B @ A``.

    The original ``weight`` / ``bias`` tensors are *adopted* (same objects,
    ``requires_grad`` off) instead of being kept in a child module, so
    ``state_dict()`` keys are **unchanged** (``...linear1.weight``, not
    ``...linear1.base.weight``) and exactly ``lora_A`` / ``lora_B`` are added.
    That is what lets ``--transfer_from`` load a plain checkpoint into a LoRA
    model, and turns "merge" into "fold into ``weight`` and drop the two adapter
    keys" -- no module-tree surgery, no key renaming, in either direction.

    ``lora_B`` is zero, i.e. the wrapper is the identity at injection time.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float,
                 dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)}")
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.weight = base.weight
        self.register_parameter("bias", base.bias)
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        delta = F.linear(F.linear(self.lora_dropout(x), self.lora_A),
                         self.lora_B)
        return out + delta * self.scaling

    def merged_weight(self) -> torch.Tensor:
        return self.weight + self.scaling * (self.lora_B @ self.lora_A)

    def merge_into_base(self) -> None:
        with torch.no_grad():
            self.weight.copy_(self.merged_weight())
            self.lora_B.zero_()

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={self.rank}, alpha={self.alpha:g}, "
                f"scaling={self.scaling:g}")


# ---------------------------------------------------------------------- #
# qkv adapter for RotaryMultiheadAttention
# ---------------------------------------------------------------------- #
def attach_qkv_lora(attn: nn.Module, rank: int, alpha: float,
                    dropout: float = 0.0) -> None:
    """Add ``lora_qkv_A``/``lora_qkv_B`` to a rotary attention module.

    Split out of :func:`inject_lora` so the three lines that read it in
    ``RotaryMultiheadAttention.forward`` are the only coupling between the
    adapter and the model.
    """
    embed_dim = int(attn.embed_dim)
    attn.lora_qkv_A = nn.Parameter(torch.empty(int(rank), embed_dim))
    attn.lora_qkv_B = nn.Parameter(torch.zeros(3 * embed_dim, int(rank)))
    attn.lora_qkv_scaling = float(alpha) / float(rank)
    attn.lora_qkv_dropout = nn.Dropout(float(dropout))
    nn.init.kaiming_uniform_(attn.lora_qkv_A, a=math.sqrt(5))
    attn.in_proj_weight.requires_grad_(False)
    if attn.in_proj_bias is not None:
        attn.in_proj_bias.requires_grad_(False)


def qkv_delta(attn: nn.Module, query: torch.Tensor, key: torch.Tensor,
              value: torch.Tensor):
    """Return ``(dq, dk, dv)`` for a module carrying ``lora_qkv_A``.

    Shared by the rotary forward.  Self-attention passes the same tensor three
    times, so the (rank x E) projection is computed once in that case.
    """
    scale = attn.lora_qkv_scaling
    A, B = attn.lora_qkv_A, attn.lora_qkv_B
    drop = attn.lora_qkv_dropout
    E = int(attn.embed_dim)
    if key is query and value is query:
        a = F.linear(drop(query), A)
        return (F.linear(a, B[:E]) * scale,
                F.linear(a, B[E:2 * E]) * scale,
                F.linear(a, B[2 * E:]) * scale)
    return (F.linear(F.linear(drop(query), A), B[:E]) * scale,
            F.linear(F.linear(drop(key), A), B[E:2 * E]) * scale,
            F.linear(F.linear(drop(value), A), B[2 * E:]) * scale)


def merge_qkv(attn: nn.Module) -> None:
    """Fold ``lora_qkv_*`` into ``in_proj_weight`` and neutralise the adapter."""
    with torch.no_grad():
        attn.in_proj_weight.add_(
            attn.lora_qkv_scaling * (attn.lora_qkv_B @ attn.lora_qkv_A))
        attn.lora_qkv_B.zero_()


# ---------------------------------------------------------------------- #
# Injection / freezing / merging
# ---------------------------------------------------------------------- #
def _wrap(module: nn.Module, attr: str, cfg: Dict) -> None:
    base = getattr(module, attr)
    if isinstance(base, LoRALinear):
        return
    setattr(module, attr, LoRALinear(base, cfg["rank"], cfg["alpha"],
                                     cfg["dropout"]))


def inject_lora(net: nn.Module, cfg) -> Dict:
    """Wrap the configured linear layers of ``net`` in place.

    Called from ``CodeTransformer.__init__`` when ``finetune.enabled`` is set,
    i.e. *before* any checkpoint is loaded, so a LoRA run and every reader
    (training, REPL, eval scripts, probes) agree on the parameter names.  Returns
    the resolved config plus a count of wrapped modules.
    """
    fcfg = resolve_config(cfg)
    targets = set(fcfg["targets"])
    counts = {name: 0 for name in TARGETS}

    # ``nn.TransformerEncoderLayer`` has a fused **eval** fast path
    # (``torch._transformer_encoder_layer_fwd``) that reads ``linear1.weight`` /
    # ``linear2.weight`` as plain tensors and never calls the FFN modules, so
    # the encoder adapters would train and then silently do nothing at
    # inference (measured: exactly that).  It is gated on
    # ``torch.backends.mha.get_fastpath_enabled()``, so switch the fast path off
    # for the process.  It only exists in eval -- training was never using it --
    # and the cost is irrelevant next to the correctness of "trained ==
    # evaluated".  (Setting ``self_attn._qkv_same_embed_dim = False`` would also
    # disable it, but on a *stock* ``nn.MultiheadAttention`` that flag selects
    # the use_separate_proj_weight branch, which asserts because the separate
    # weights are None; only the rotary subclass overrides forward.)
    if torch.backends.mha.get_fastpath_enabled():
        torch.backends.mha.set_fastpath_enabled(False)
        print("LoRA: disabled torch's fused MHA/encoder fast path (it bypasses "
              "the adapter modules outside training)")

    encoder = getattr(net, "encoder", None)
    if encoder is not None and hasattr(encoder, "encoder"):
        for layer in encoder.encoder.layers:
            if "encoder_ffn" in targets:
                _wrap(layer, "linear1", fcfg)
                _wrap(layer, "linear2", fcfg)
                counts["encoder_ffn"] += 2

    decoder = getattr(net, "decoder", None)
    if decoder is not None and hasattr(decoder, "decoder"):
        for layer in decoder.decoder.layers:
            if "decoder_ffn" in targets:
                _wrap(layer, "linear1", fcfg)
                _wrap(layer, "linear2", fcfg)
                counts["decoder_ffn"] += 2
            # Both attention targets only exist on the rotary self-attention
            # (see the module docstring); detect it by the rotary embedding it
            # owns rather than importing the class (circular import).
            rotary = getattr(layer.self_attn, "rope", None) is not None
            if rotary and "decoder_attn" in targets:
                _wrap(layer.self_attn, "out_proj", fcfg)
                counts["decoder_attn"] += 1
            if (rotary and "decoder_qkv" in targets
                    and getattr(layer.self_attn, "lora_qkv_A", None) is None):
                attach_qkv_lora(layer.self_attn, fcfg["rank"], fcfg["alpha"],
                                fcfg["dropout"])
                counts["decoder_qkv"] += 1

    empty = sorted(name for name in targets if counts.get(name, 0) == 0)
    if empty:
        raise ValueError(
            f"finetune.targets requested {empty}, which matched no module.  "
            f"decoder_attn/decoder_qkv adapt the rotary decoder self-attention, "
            f"so they need transformer.rope: true; encoder_attn/decoder "
            f"cross-attention adapters are not implemented (see "
            f"modeling/lora.py).")

    n_wrapped = sum(counts.values())
    net.lora_enabled = n_wrapped > 0
    net.lora_config = dict(fcfg)
    return {"config": fcfg, "modules": n_wrapped, "counts": counts}


def lora_parameters(net: nn.Module) -> Iterator[nn.Parameter]:
    for module in net.modules():
        if isinstance(module, LoRALinear):
            yield module.lora_A
            yield module.lora_B
        a = getattr(module, "lora_qkv_A", None)
        if isinstance(a, nn.Parameter):
            yield a
            yield module.lora_qkv_B


def lora_param_names(net: nn.Module) -> List[str]:
    names = []
    for name, module in net.named_modules():
        if isinstance(module, LoRALinear):
            names.extend([f"{name}.lora_A", f"{name}.lora_B"])
        if getattr(module, "lora_qkv_A", None) is not None:
            names.extend([f"{name}.lora_qkv_A", f"{name}.lora_qkv_B"])
    return names


def is_lora_param(name: str) -> bool:
    return ".lora_" in name


def freeze_except_adapters(net: nn.Module, cfg=None) -> Dict[str, float]:
    """Freeze every parameter, then re-enable the adapters (+ options).

    Returns the trainable/total parameter counts of each group, for the startup
    print.  The optimizer split (``TrainModule._split_params``) calls this only
    once, from ``TrainModule.__init__``; nothing else in the model is movable.
    """
    fcfg = resolve_config(cfg if cfg is not None else
                          getattr(net, "lora_config", None))
    total = sum(p.numel() for p in net.parameters())
    for p in net.parameters():
        p.requires_grad_(False)
    lora_n = 0
    for p in lora_parameters(net):
        p.requires_grad_(True)
        lora_n += p.numel()

    spk_n = 0
    if fcfg["train_spk_emb"]:
        for holder in (getattr(net, "encoder", None), getattr(net, "decoder", None)):
            emb = getattr(holder, "spk_emb", None)
            if emb is not None:
                emb.weight.requires_grad_(True)
                spk_n += emb.weight.numel()

    mem_n = 0
    if fcfg["train_memory_norm"]:
        enc = getattr(net, "encoder", None)
        for name in ("memory_gain", "memory_bias"):
            p = getattr(enc, name, None)
            if isinstance(p, nn.Parameter):
                p.requires_grad_(True)
                mem_n += p.numel()

    return {"lora": lora_n, "spk_emb": spk_n, "memory_norm": mem_n,
            "trainable": lora_n + spk_n + mem_n, "total": total}


def merge_lora(net: nn.Module) -> Dict[str, torch.Tensor]:
    """Fold every adapter into its base weight and return a *plain* state dict.

    The returned mapping is exactly what a model built with
    ``finetune.enabled: false`` expects: every original key, no ``lora_*`` and
    no ``lora_marker``.  The model in memory stays adapted (with a zeroed
    ``lora_B``, i.e. unchanged function), so the caller can compare before/after
    -- which ``scratch/merge_lora.py`` does before writing anything.
    """
    with torch.no_grad():
        for module in net.modules():
            if isinstance(module, LoRALinear):
                module.merge_into_base()
            elif getattr(module, "lora_qkv_A", None) is not None:
                merge_qkv(module)
    state = {k: v.detach().clone()
             for k, v in net.state_dict().items() if not is_lora_key(k)}
    return state


def is_lora_key(name: str) -> bool:
    return (".lora_" in name or name == "lora_marker"
            or name.endswith(".lora_marker"))


# ---------------------------------------------------------------------- #
# Checkpoint helpers
# ---------------------------------------------------------------------- #
def uses_lora(state_dict) -> bool:
    """Whether a checkpoint carries LoRA adapters (raw or ``net.``-prefixed)."""
    return any(is_lora_key(k) for k in state_dict)


def lora_keys(state_dict) -> List[str]:
    return [k for k in state_dict if is_lora_key(k)]


def check_lora_layout(state_dict, net: nn.Module, source: str,
                      strict: bool = True) -> None:
    """Refuse to silently drop / rewrite trained adapters.

    Called from ``2_training.check_special_layout`` (and therefore from both the
    ``--resume_from`` and ``--transfer_from`` paths), with the checkpoint's
    state dict as Lightning writes it (``net.`` prefix).  It lives here rather
    than in the training script so the CPU test can exercise it without
    importing ``2_training`` (whose ``ultimate_xc`` import installs a pdb
    excepthook).

    * checkpoint has adapters, model does not -> fatal, always (every base key
      would still match, so the result would silently be the base model);
    * model has adapters, checkpoint does not -> the intended warm start
      (``lora_B`` is zero) under ``--transfer_from``, fatal under
      ``--resume_from`` (a strict load cannot absorb the new keys);
    * both have them -> ``rank``/``alpha``/``targets`` must match: rank changes
      shapes (silently skipped on load), alpha changes only the scale (would
      load completely and score a different function), and a target the
      checkpoint has but the model does not is a dropped trained adapter.
    """
    has_lora = uses_lora(state_dict)
    model_lora = bool(getattr(net, "lora_enabled", False))
    if has_lora and not model_lora:
        raise SystemExit(
            f"ERROR: {source} carries trained LoRA adapters (lora_A/lora_B), but "
            f"this model was built with finetune.enabled: false.  Set "
            f"finetune.enabled: true with the same rank/alpha/targets as the "
            f"run (otherwise the adapters are dropped and the result is the "
            f"base model), or fold them into a plain checkpoint with "
            f"scratch/merge_lora.py.")
    if model_lora and not has_lora:
        message = (f"{source} has no LoRA adapters, but this model was built "
                   f"with finetune.enabled: true.  Every adapter starts at "
                   f"zero, so the base weights transfer and this is exactly the "
                   f"intended warm start.")
        if strict:
            raise SystemExit(
                f"ERROR: {message}  Use --transfer_from {source} instead of "
                f"--resume_from, or start a fresh exp_name.")
        print(f"NOTE: {message}")
        return
    if not (has_lora and model_lora):
        return

    # Lightning writes ``net.``-prefixed keys; a raw ``model.state_dict()`` does
    # not.  Compare in the checkpoint's key space so both callers work.
    prefix = "net." if any(k.startswith("net.") for k in state_dict) else ""
    marker = state_dict.get(f"{prefix}lora_marker")
    want = net._buffers.get("lora_marker")
    if marker is not None and want is not None and marker.numel() >= 2:
        if (int(marker[0]) != int(want[0])
                or abs(float(marker[1]) - float(want[1])) > 1e-6):
            raise SystemExit(
                f"ERROR: {source} was trained with LoRA rank={int(marker[0])} "
                f"alpha={float(marker[1]):g}, but this model is built with "
                f"rank={net.lora_config['rank']} alpha="
                f"{net.lora_config['alpha']:g}.  Rank changes the adapter "
                f"shapes; alpha only changes the scale (it would load and "
                f"silently rescale the fine-tune).  Match finetune.rank / "
                f"finetune.alpha to the checkpoint.")
    model_sd = {f"{prefix}{k}": v for k, v in net.state_dict().items()}
    dropped = [k for k in state_dict
               if ".lora_" in k and "lora_marker" not in k
               and k not in model_sd]
    if dropped:
        raise SystemExit(
            f"ERROR: {source} has {len(dropped)} adapter tensor(s) this model "
            f"does not have (e.g. {dropped[0]}).  finetune.targets must match "
            f"the run that wrote the checkpoint, otherwise those trained "
            f"adapters are dropped and the affected modules run with their base "
            f"weights only.")
    bad = [k for k in state_dict
           if ".lora_" in k and k in model_sd
           and state_dict[k].shape != model_sd[k].shape]
    if bad:
        raise SystemExit(
            f"ERROR: {source}: adapter shapes disagree with this model "
            f"(e.g. {bad[0]}).  finetune.rank must match the run that wrote the "
            f"checkpoint.")
