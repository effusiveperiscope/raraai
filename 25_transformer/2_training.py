"""Stage-2 training for the discrete-code transformer.

Mirrors the layout of ``1_training.py`` (Lightning module + ``train``
function + checkpoint/resume handling), but trains ``modeling.transformer
.CodeTransformer`` instead of the VITS-style Generator/Discriminator pair.

All model-side computations (delay pattern, shifted targets, masked
cross-entropy, sampling) live in ``modeling/transformer.py``; this file only
owns the optimisation loop, data loading, logging and checkpointing.
"""

import itertools
import logging
import os

# Claim the root logger's handler slot *before* the heavy imports.
#
# `modeling/generator.py` pulls in the vendored NSF decoder
# (`modeling/rvc/my_nsf.py` -> `svc_helper.svc.rvc.lib.infer_pack`), and
# `svc_helper.svc.rvc.modules.vc.utils` does `from fairseq import checkpoint_utils`
# (`vc/pipeline.py` imports faiss too).  fairseq's
# `models/wav2vec/wav2vec2_classification.py` calls
# `logging.basicConfig(level=logging.DEBUG)` **at module level**, which installs
# a root StreamHandler formatted `%(levelname)s:%(name)s:%(message)s` and turns
# the root logger to DEBUG -- that is where `DEBUG:numba.core.byteflow:bytecode
# dump:` and `INFO:faiss.loader:` in the log come from (measured: 47 such lines
# for one njit call, 0 with this guard).
#
# `logging.basicConfig` returns immediately once the root logger has a handler,
# so doing it here means fairseq's call can change neither the handler nor the
# level.  A bare `logging.getLogger().setLevel(WARNING)` is NOT enough: this
# script imports the Generator *lazily* (`load_generator()` runs at the first
# validation sample / OOD probe, long after that line), so DEBUG came back
# afterwards (measured root level 10 -> 30 with the guard).
# The post-import reset below is kept as cheap insurance against a library that
# calls `logging.getLogger().setLevel(DEBUG)` directly (hydra does that).
logging.basicConfig(level=logging.WARNING)

import librosa
import matplotlib
matplotlib.use("Agg")  # training is headless; must precede the pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from commons import (load_state_dict_mismatch, load_submodule_prefix,
                     matplotlib_to_tensorboard)
from dataset_codes import (CodeCollator, CodeDataset,
                           LengthBucketBatchSampler)
from modeling.lora import (check_lora_layout, freeze_except_adapters,
                           lora_parameters, resolve_config as resolve_finetune)
from modeling.transformer import CodeTransformer, expand_state_dict_with_special

import ultimate_xc
logging.getLogger().setLevel(logging.WARNING)


# Held-out text used by the out-of-distribution sampling probe (see
# ``TrainModule._log_ood_generation``). These sentences do not occur in the
# training filelists, so they exercise the phoneme encoder on unseen wording.
# Override with ``train.ood_texts`` in configs/transformer.yaml.
DEFAULT_OOD_TEXTS = [
    "The quick brown fox jumps over the lazy dog while the orchestra "
    "rehearses a brand new symphony in the courtyard.",
]


def load_generator(checkpoint_path, config, device):
    from modeling.generator import Generator

    net_g = Generator(hp=config)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(k.startswith("net_g.") for k in state.keys()):
        load_submodule_prefix(net_g, "net_g.", state)
    else:
        load_state_dict_mismatch(net_g, state)
    net_g.to(device)
    net_g.eval()
    return net_g


# ---------------------------------------------------------------------- #
# Visualisation helpers
#
# Module level so they can be exercised from a REPL without a Trainer.  Each
# returns a matplotlib Figure; the caller converts it with
# ``commons.matplotlib_to_tensorboard`` and closes it.
# ---------------------------------------------------------------------- #
def plot_code_comparison_figure(gt, pred, pad_id, title, row_labels):
    """Two stacked codebook heatmaps: target (top) vs generated (bottom).

    Args:
        gt, pred: (K, T) integer code tensors for the *same* speaker + text.
        pad_id: code index used for padding (rendered blank).
        title: figure title.
        row_labels: per-row y labels, e.g. ``["book 0", ...]``.
    """
    panels = []
    for codes in (gt, pred):
        arr = codes.detach().float().cpu().numpy()
        # pad ids are shared inside a spec/pitch group, so one mask suffices
        panels.append(np.ma.masked_array(arr, mask=(arr == float(pad_id))))

    k = panels[0].shape[0]
    fig, axes = plt.subplots(2, 1, figsize=(10, 0.38 * k * 2 + 1.9),
                             sharex=True)
    im = None
    for ax, arr, name in zip(axes, panels, ("target", "generated")):
        im = ax.imshow(arr, aspect="auto", interpolation="nearest",
                       cmap="viridis", vmin=0, vmax=max(int(pad_id) - 1, 1))
        ax.set_yticks(np.arange(k))
        ax.set_yticklabels(row_labels, fontsize=7)
        ax.set_title(f"{title} — {name}", fontsize=9)
    axes[-1].set_xlabel("code frame")
    # Dedicated colourbar axes: `fig.colorbar(ax=axes)` overlaps the wide
    # heatmaps after tight_layout.
    fig.subplots_adjust(left=0.08, right=0.88, top=0.92, bottom=0.09,
                        hspace=0.28)
    fig.colorbar(im, cax=fig.add_axes([0.90, 0.15, 0.02, 0.7]))
    return fig


def plot_generated_codes_figure(codes, pad_id, title, row_labels):
    """One codebook heatmap for a generated sequence.

    Used by the OOD probe, where there is no target code stream to put next to
    the prediction (``plot_code_comparison_figure`` needs both).

    Args:
        codes: (K, T) integer code tensor.
        pad_id: code index used for padding (rendered blank).
        title: figure title.
        row_labels: per-row y labels, e.g. ``["book 0", ...]``.
    """
    arr = codes.detach().float().cpu().numpy()
    arr = np.ma.masked_array(arr, mask=(arr == float(pad_id)))

    k = arr.shape[0]
    fig, ax = plt.subplots(figsize=(10, 0.38 * k + 1.4))
    im = ax.imshow(arr, aspect="auto", interpolation="nearest", cmap="viridis",
                   vmin=0, vmax=max(int(pad_id) - 1, 1))
    ax.set_yticks(np.arange(k))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("code frame")
    fig.subplots_adjust(left=0.08, right=0.88, top=0.90, bottom=0.12)
    fig.colorbar(im, cax=fig.add_axes([0.90, 0.15, 0.02, 0.7]))
    return fig


def plot_mel_figure(audio, sampling_rate, hop_length, title, n_mels=80,
                    n_fft=2048):
    """Single log-mel spectrogram (OOD samples have no target to compare)."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size < n_fft:
        audio = np.pad(audio, (0, n_fft - audio.size))

    mel = librosa.feature.melspectrogram(
        y=audio, sr=sampling_rate, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, power=2.0)
    db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 2.6))
    im = ax.imshow(db, aspect="auto", origin="lower", cmap="magma",
                   vmin=-80, vmax=0)
    ax.set_xlabel("frame")
    ax.set_ylabel("mel bin")
    ax.set_title(title, fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.88, top=0.88, bottom=0.16)
    fig.colorbar(im, cax=fig.add_axes([0.90, 0.15, 0.02, 0.7]))
    return fig


def plot_pitch_curves_figure(curves, hop_length, sampling_rate, title):
    """Overlay one or more f0 contours decoded from pitch codes.

    Args:
        curves: list of ``(label, f0)`` where f0 is a (T,) tensor in Hz at the
            audio frame rate (``sampling_rate / hop_length`` = 100 Hz here)
            and 0 marks unvoiced frames.
        hop_length, sampling_rate: used to build a seconds time axis.
    """
    prepared = []
    for label, f0 in curves:
        arr = f0.detach().float().cpu().reshape(-1).numpy()
        arr = np.where(arr > 0, arr, np.nan)  # gaps at unvoiced frames
        prepared.append((label, arr))

    n = max(arr.size for _, arr in prepared)
    if n == 0 or not any(np.isfinite(arr).any() for _, arr in prepared):
        return None

    t = np.arange(n) * hop_length / sampling_rate
    fig, ax = plt.subplots(figsize=(10, 3))
    for label, arr in prepared:
        ax.plot(t[: arr.size], arr, linewidth=1.1, label=label)

    # Pin the y-range to the reference (first curve) so a diverging
    # generation cannot squash the contour we care about.
    finite = prepared[0][1][np.isfinite(prepared[0][1])]
    if finite.size == 0:
        finite = np.concatenate(
            [arr[np.isfinite(arr)] for _, arr in prepared] or [np.array([])])
    if finite.size:
        ax.set_ylim(0.0, max(float(finite.max()) * 1.25, 1.0))

    ax.set_xlim(0, max(t[-1], 1e-3))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("f0 (Hz)")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    return fig


def plot_mel_comparison_figure(audio_ref, audio_pred, sampling_rate,
                               hop_length, title, n_mels=80, n_fft=2048):
    """Two stacked log-mel spectrograms (target-code vs generated-code audio).

    Both panels share one colour scale (referenced to the max over their
    concatenation) so loudness differences stay visible instead of being
    normalised away per panel.
    """
    audio_ref = np.asarray(audio_ref, dtype=np.float32).reshape(-1)
    audio_pred = np.asarray(audio_pred, dtype=np.float32).reshape(-1)
    if audio_ref.size < n_fft:
        audio_ref = np.pad(audio_ref, (0, n_fft - audio_ref.size))
    if audio_pred.size < n_fft:
        audio_pred = np.pad(audio_pred, (0, n_fft - audio_pred.size))

    kwargs = dict(sr=sampling_rate, n_fft=n_fft, hop_length=hop_length,
                  n_mels=n_mels, power=2.0)
    mel_ref = librosa.feature.melspectrogram(y=audio_ref, **kwargs)
    mel_pred = librosa.feature.melspectrogram(y=audio_pred, **kwargs)
    mel = np.concatenate([mel_ref, mel_pred], axis=1)
    db = librosa.power_to_db(mel, ref=np.max)
    db_ref, db_pred = db[:, : mel_ref.shape[1]], db[:, mel_ref.shape[1]:]

    fig, axes = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True, sharey=True)
    im = None
    for ax, spec, name in zip(axes, (db_ref, db_pred),
                              ("target codes", "generated codes")):
        im = ax.imshow(spec, aspect="auto", origin="lower", cmap="magma",
                       vmin=-80, vmax=0)
        ax.set_title(f"{title} — {name}", fontsize=9)
    axes[-1].set_xlabel("frame")
    axes[0].set_ylabel("mel bin")
    fig.subplots_adjust(left=0.08, right=0.88, top=0.92, bottom=0.11,
                        hspace=0.28)
    fig.colorbar(im, cax=fig.add_axes([0.90, 0.15, 0.02, 0.7]))
    return fig


def _optimizer_list(optims):
    """Normalise Lightning's ``optimizers()`` / ``lr_schedulers()`` return value.

    Lightning returns a single object when ``configure_optimizers`` supplied
    one, and a list otherwise.  Stage 2 has two optimizers normally and one in
    fine-tune mode, so every consumer has to handle both shapes (indexing a bare
    optimizer would silently index its *parameter groups*).
    """
    if optims is None:
        return []
    if isinstance(optims, (list, tuple)):
        return list(optims)
    return [optims]


class _LayoutGuard:
    """Mixin: survive a checkpoint whose optimizer *layout* changed.

    Stage 2 has one parameter group per optimizer, and which parameters land in
    it depends on ``train.muon_include_embeddings`` / ``train.muon_include_
    heads``.  ``torch.optim.Optimizer.load_state_dict`` refuses such a
    checkpoint (``ValueError: loaded state dict contains a parameter group that
    doesn't match the size of optimizer's group``) — and if the *count* happened
    to match while the ordering shifted, it would silently attach the restored
    ``exp_avg``/``exp_avg_sq``/``momentum_buffer`` to the wrong tensors, since
    the state is matched by position.

    Neither outcome is what we want when only the split changed: the *weights*
    are still perfectly valid, only the per-parameter moments are unusable.  So
    on a mismatch this copies the group hyperparameters (lr, betas, eps,
    weight_decay, momentum, ns_steps, ...) from the checkpoint, skips the
    per-parameter state, and says so.  Training then continues with a fresh
    momentum/moment estimate instead of dying or drifting silently.
    """

    def load_state_dict(self, state_dict):
        saved = [len(g["params"]) for g in state_dict.get("param_groups", [])]
        current = [len(g["params"]) for g in self.param_groups]
        if saved == current:
            return super().load_state_dict(state_dict)
        for group, saved_group in zip(self.param_groups,
                                      state_dict.get("param_groups", [])):
            for key, value in saved_group.items():
                if key != "params":
                    group[key] = value
        self.state.clear()
        print(
            f"WARNING: {type(self).__name__} layout changed since the "
            f"checkpoint was written (parameters per group: checkpoint "
            f"{saved}, now {current}). The per-parameter optimizer state "
            f"cannot be mapped onto the new layout and was skipped; the "
            f"(already loaded) weights and the group hyperparameters are kept, "
            f"so the run continues with a fresh momentum/moment estimate.")
        return None


class LayoutCheckedAdamW(_LayoutGuard, torch.optim.AdamW):
    pass


class LayoutCheckedMuon(_LayoutGuard, torch.optim.Muon):
    pass


class TrainModule(pl.LightningModule):
    def __init__(self, net: CodeTransformer, config: OmegaConf):
        super().__init__()
        self.net = net
        self.config = config
        # The frozen stage-1 Generator used for validation-time audio lives in
        # a plain list, NOT in `self.net_g = <Generator>`: `nn.Module.__setattr__`
        # registers *any* `nn.Module` assigned to an attribute as a child
        # module, which would put its 497 tensors into `state_dict()` of every
        # stage-2 checkpoint and -- the real problem -- into Lightning's strict
        # `load_state_dict` contract. See the `net_g` property below.
        self._net_g_holder = [None]

        # Gradient clipping (applied in _clip_and_log_grads, i.e. before every
        # optimizer step). 0 disables clipping; the norm is still logged. See
        # configs/transformer.yaml for the knobs.
        self.grad_clip_val = float(
            config.train.get("grad_clip_val", 0.0) or 0.0)
        self.grad_clip_algorithm = str(
            config.train.get("grad_clip_algorithm", "norm") or "norm")
        if self.grad_clip_val < 0:
            raise ValueError(
                f"train.grad_clip_val must be >= 0, got {self.grad_clip_val}")
        if self.grad_clip_val > 0 and self.grad_clip_algorithm not in ("norm",
                                                                      "value"):
            raise ValueError(
                "train.grad_clip_algorithm must be 'norm' or 'value', got "
                f"{self.grad_clip_algorithm!r}")
        if self.grad_clip_val > 0:
            print(f"Gradient clipping: {self.grad_clip_algorithm} "
                  f"max={self.grad_clip_val}")

        # Counterfactual hint margin (v1.1).  The clip-level channels are almost
        # loss-irrelevant under plain CE (their level moves the memory by ~2 % of
        # its norm and the table never grows -- HINTS_V1_PLAN.md §5.6), so CE
        # alone trains neither the *encoding* nor the model's *sensitivity* to
        # it.  On `counterfactual_rate` of the steps this scores the same batch
        # twice, once with every sample's level replaced by a different one, and
        # adds
        #     relu(margin - (CE_shifted - CE_true))
        # i.e. "changing the level must hurt by at least `margin` nats".  That is
        # the cheapest objective that rewards *dependence* on the hint; it is the
        # plan's §4.1 fallback, and it is what plain augmentation cannot do (with
        # a fixed target a perturbed hint is best ignored, so augmenting alone
        # would train invariance instead of control).
        self.cf_margin = float(
            config.train.get("counterfactual_margin", 0.0) or 0.0)
        self.cf_weight = float(
            config.train.get("counterfactual_weight", 1.0) or 1.0)
        self.cf_rate = float(
            config.train.get("counterfactual_rate", 0.0) or 0.0)
        self.cf_channels = [c.strip() for c in str(
            config.train.get("counterfactual_channels", "intensity,register")
            or "").split(",") if c.strip()]
        for name in self.cf_channels:
            if name not in ("intensity", "register"):
                raise ValueError(
                    f"train.counterfactual_channels: unknown channel {name!r}; "
                    f"only the clip-level ones (intensity, register) have a "
                    f"level to shift")
        if (self.cf_margin > 0) != (self.cf_rate > 0):
            raise ValueError(
                "train.counterfactual_margin and train.counterfactual_rate must "
                f"both be > 0 (got margin={self.cf_margin}, rate={self.cf_rate})")
        if self.cf_margin > 0:
            if not net.hints:
                raise ValueError(
                    "train.counterfactual_margin needs transformer.hints: true")
            print(f"Counterfactual hint margin: {self.cf_margin} nats on "
                  f"{100 * self.cf_rate:.0f}% of steps, channels "
                  f"{self.cf_channels} (weight {self.cf_weight})")

        # ---- fine-tuning mode (--finetune, LoRA) -------------------------- #
        # The adapters themselves are injected by ``CodeTransformer.__init__``
        # (so every reader of the same config sees the same parameter names);
        # this freezes everything else and owns the optimizer.  See
        # modeling/lora.py for the rationale and the checkpoint contract.
        ft_raw = config.get("finetune", None)
        self.finetune_enabled = bool(getattr(net, "lora_enabled", False))
        self.finetune_cfg = None
        if self.finetune_enabled:
            self.finetune_cfg = resolve_finetune(
                ft_raw if ft_raw is not None else net.lora_config)
            counts = freeze_except_adapters(net, ft_raw)
            if counts["trainable"] <= 0:
                raise ValueError(
                    "finetune is enabled but freezes every parameter: no "
                    "adapter was injected.  Check finetune.targets "
                    "(decoder_qkv needs transformer.rope: true) and that the "
                    "config the model was built from is the same one.")
            ft = self.finetune_cfg
            print(f"Finetune: LoRA rank={ft['rank']} alpha={ft['alpha']:g} "
                  f"dropout={ft['dropout']:g} on "
                  f"{','.join(ft['targets'])}; "
                  f"trainable {counts['trainable']/1e6:.2f} M / "
                  f"{counts['total']/1e6:.1f} M "
                  f"({100.0 * counts['trainable'] / max(counts['total'], 1):.2f} %: "
                  f"lora {counts['lora']/1e6:.2f} M, "
                  f"spk_emb {counts['spk_emb']/1e6:.2f} M, "
                  f"memory_norm {counts['memory_norm']} params) "
                  f"@ lr={ft['lr']:g}")

        # Learning-rate schedule end.  ``CosineAnnealingLR`` is *periodic* with
        # period ``2 * T_max``: past ``cosine_anneal_end`` the LR does not stay
        # at ``eta_min``, it climbs back towards its base value (reaching it at
        # ``2 * cosine_anneal_end``).  Nothing in Lightning stops the run there,
        # so a config whose ``max_steps`` is larger than its
        # ``cosine_anneal_end`` silently turns the tail of the run into an
        # un-anneal -- the exact opposite of what the key name promises.
        # ``_check_anneal_end`` warns once, at train start (a resumed run can
        # already be past it) and at the step it is crossed.
        self.anneal_end = int(
            config.train.get("cosine_anneal_end", 1000000) or 0)
        if self.finetune_cfg and self.finetune_cfg["cosine_anneal_end"]:
            # A fine-tune is short, so its own budget defaults to the *global*
            # one but can be shortened without touching the base config.
            self.anneal_end = int(self.finetune_cfg["cosine_anneal_end"])
        self._anneal_warned = False

        # Muon's Newton-Schulz iteration count (see configure_optimizers). Its
        # cost is per-step and batch-independent, so this is the cheapest perf
        # knob stage 2 has: measured ~1.26 TMAC at ns_steps=5 on this model.
        ns_steps = config.train.get("muon_ns_steps", 5)
        self.ns_steps = 5 if ns_steps is None else int(ns_steps)
        if not 1 <= self.ns_steps <= 99:
            raise ValueError(
                f"train.muon_ns_steps must be in [1, 99], got {self.ns_steps}")
        # Per-epoch spike statistics (only meaningful with clipping enabled,
        # where the norm is computed on every step anyway).
        self._grad_norm_max = None
        self._grad_clipped_steps = 0.0
        self._grad_steps = 0.0

        # `configure_optimizers` returns two optimizers (Muon for the 2-D
        # matrices, AdamW for everything else) and PyTorch Lightning >= 2.0
        # rejects that with automatic optimization ("Training with multiple
        # optimizers is only supported with manual optimization").  We
        # therefore own zero_grad / backward / clip / step, see training_step.
        # Fine-tune mode (`--finetune`) returns a *single* AdamW but keeps the
        # manual loop, so there is one code path and no "which mode am I in?"
        # branch in the step accounting.
        # NB: Lightning does not call `on_before_optimizer_step` in manual
        # optimization, which is why the clipping moved into
        # `_clip_and_log_grads` (called explicitly before stepping).
        self.automatic_optimization = False

        # Lightning forbids `Trainer(accumulate_grad_batches > 1)` with manual
        # optimization, so the knob must not be silently ignored if someone
        # adds it to the config.
        accum = int(config.train.get("accumulate_grad_batches", 1) or 1)
        if accum > 1:
            raise ValueError(
                "train.accumulate_grad_batches > 1 is not supported: Lightning "
                "disallows automatic gradient accumulation with manual "
                "optimization, which the Muon+AdamW optimizer pair requires.")

        # Text-conditioning diagnostic (v0.4.1): every epoch, compare the
        # teacher-forced CE on the first `text_delta_batches` validation
        # batches against the same batches with the phonemes rotated across
        # samples. The difference is `val_text_delta`; ~0 means the decoder is
        # ignoring the encoder memory (a text-blind, speaker-only code LM),
        # which is invisible in the loss because acoustic continuity alone is a
        # strong prior. 0 disables the probe.
        self.text_delta_batches = int(
            config.train.get("text_delta_batches", 0) or 0)
        self._reset_text_delta()

        # Hint-oracle probe (v0.8): every epoch, re-score the first
        # `hint_probe_batches` validation batches with every conditioning hint
        # dropped and log both `val_loss_nohint` and the gap.  The hints come
        # from the target audio, so this is the metric that says whether the
        # model is *using* the oracle or leaning on it. 0 disables the probe.
        self.hint_probe_batches = int(
            config.train.get("hint_probe_batches", 2) or 0)
        self._reset_hint_probe()
        self._hints_reported = False

        # Which hint state *generation* runs in (TensorBoard samples, the FIM
        # infill probe, the speaker probe, the OOD probe and the printed
        # lengths).  This is not cosmetic: `None` used to be passed to
        # `generate`, which drops the hint vectors entirely -- a *different*
        # function from the level-0 "not provided" state the model is trained
        # on (+0.051 nats, 52 % memory shift on transformer_v0.9).  Modes:
        #   cond -> the item's real `.cond` hints (val batches carry them)
        #   null -> the trained null state (every channel at level 0)
        #   omit -> no hint vectors at all (pre-Phase-0 behaviour; ablation)
        # OOD text has no `.cond` (there is no target audio), so it always
        # falls back to `null`/`omit`.
        self.sample_hints = str(config.train.get("sample_hints", "null")
                                or "null").lower()
        if self.sample_hints not in ("cond", "null", "zero", "omit"):
            raise ValueError(
                f"train.sample_hints must be cond | null | omit, got "
                f"{self.sample_hints!r}")
        if self.sample_hints == "zero":       # alias
            self.sample_hints = "null"

        # Per-channel hint liveness probe (Phase 1).  `val_hint_gap` drops every
        # channel at once, so one live channel hides four dead ones; this one
        # re-scores the first `hint_channel_probe` validation batches with each
        # channel dropped *separately* and logs
        # `val_hint_channel_delta/{int,reg,sem,pitch,slot,all,omit}`.  `all` is
        # the trained null state, `omit` the pre-Phase-0 inference path, so
        # `all - omit` is that bug's cost on this checkpoint.  Seven extra
        # forwards per probed batch; 0 disables it.
        self.hint_channel_probe = int(
            config.train.get("hint_channel_probe", 0) or 0)
        self._reset_hint_channel_probe()

        # Guidance weight for the *generated* audio in TensorBoard (Phase 3).
        # `None` uses `transformer.hint_guidance` (off by default); the sweep
        # belongs in `scratch/hint_guidance_sweep.py`, not in a training run.
        self.sample_hint_guidance = config.train.get(
            "sample_hint_guidance", None)

        # End-to-end WER probe (v2.4).  Every `wer_interval` epochs, decode a few
        # val utterances through the frozen stage-1 Generator + ASR and log
        # `wer/generated` (plain AR), `wer/fim` (middle third regenerated by
        # `infill`, spliced back) and `wer/reference` (ground-truth codes, the
        # vocoder+ASR floor).  The implementation lives in `stage2_wer.py`, the
        # same module `scratch/eval_stage2_wer.py` uses, so an offline number and
        # a training-time number are directly comparable.  0 disables it (and
        # costs nothing: the ASR weights are only loaded on the first probe).
        self.wer_interval = int(config.train.get("wer_interval", 0) or 0)
        self.wer_items = int(config.train.get("wer_items", 8) or 8)
        self.wer_fim = bool(config.train.get("wer_fim", True))
        self.wer_reference = bool(config.train.get("wer_reference", True))
        self.wer_hints = str(config.train.get("wer_hints", "null")
                             or "null").lower()
        self._wer_state = None
        self._wer_failed = False
        self._wer_floor = None

        # Stage-2 -> stage-1 speaker-id map applied ONLY when the frozen
        # Generator decodes codes (see AGENTS.md §6.1).  A run trained on a
        # stage-2 filelist whose ids differ from the stage-1 filelist's (the
        # shipped combined corpus does: Nahida is 1777 vs 1563) conditions the
        # transformer on the id it learned but renders a *different* voice for
        # that id.  `train.speaker_map` fixes this at data time and needs a
        # retrain; this map fixes an existing checkpoint's audio without one.
        self.stage1_speaker_map = config.train.get("stage1_speaker_map", None)
        if self.stage1_speaker_map and config.train.get("speaker_map", None):
            print("WARNING: both train.speaker_map and train.stage1_speaker_map "
                  "are set. The dataset has already canonicalised the ids, so "
                  "re-mapping them at decode would be wrong -- disabling "
                  "stage1_speaker_map.")
            self.stage1_speaker_map = None
        self._gen_sid_map_cache = None



        # FIM diagnostics (v0.6-fim2): every epoch, teacher-force a *synthetic*
        # prefix/suffix split on the first `fim_probe_batches` validation batches
        # and compare the middle-only CE against three control arms (phonemes
        # rotated, suffix taken from another sample, region / length tags
        # zeroed).  Validation items are never FIM (`dataset_codes` gates it to
        # training), so the split is made here instead of by the dataset -- that
        # is also what makes the arms comparable: identical spans, identical
        # targets, one input changed.  `fim_middle_code_match` from the sampling
        # pass cannot answer this (free-running code match is at chance for a
        # healthy model), so these deltas are the "is FIM working?" signal.
        # 0 disables the probe.
        self.fim_probe_batches = int(
            config.train.get("fim_probe_batches", 0) or 0)
        self._reset_fim_probe()

        # Out-of-distribution text probe (see `_log_ood_generation`). The
        # phoneme ids are built lazily, on first use, and cached: the first
        # sampling pass pays a one-off CPU phonemization cost, later epochs
        # reuse the ids. `_ood_phone_ids` holds `[(text, ids), ...]`; an empty
        # list means "tried and unavailable" (never retried, never fatal).
        self._ood_phone_ids = None

    # ------------------------------------------------------------------ #
    # Frozen stage-1 Generator (validation-time audio only)
    #
    # Why this is a property over a plain list instead of `self.net_g`:
    # `setup()` attaches the generator only when `train.generator_checkpoint`
    # exists *and* loads on the current host, so a checkpoint written on a box
    # that had the stage-1 run used to carry `net_g.*` tensors that a box
    # without it could not satisfy:
    #
    #     RuntimeError: Error(s) in loading state_dict for TrainModule:
    #         Missing key(s) in state_dict: "net_g.enc_q.pre.weight", ...
    #
    # It is a frozen, inference-only copy: the only authoritative source is the
    # stage-1 checkpoint path, never the stage-2 file. Keeping it out of the
    # module tree leaves `state_dict()` at the 325 `net.*` keys this module
    # actually trains (and keeps stage-2 checkpoints ~134 MB smaller).
    # ------------------------------------------------------------------ #
    @property
    def net_g(self):
        """The frozen stage-1 ``Generator``, or ``None`` when unavailable."""
        return self._net_g_holder[0]

    def load_state_dict(self, state_dict, strict=True):
        """Load a checkpoint, discarding legacy ``net_g.*`` tensors.

        Checkpoints written before the frozen generator was moved out of the
        module tree contain it as a submodule (497 tensors, ~134 MB), so a
        strict load into today's ``TrainModule`` would fail with
        ``Unexpected key(s) in state_dict: "net_g.enc_q.pre.weight", ...``.
        Those weights are not ours to restore -- the generator is rebuilt from
        ``train.generator_checkpoint`` in ``setup()`` -- so they are dropped
        here rather than forcing every existing run through a checkpoint
        rewrite. Lightning calls this via ``strategy.load_model_state_dict``
        (``lightning_module.load_state_dict(ckpt["state_dict"], strict=...)``),
        so both ``--resume_from`` and auto-resume go through it.
        """
        stale = [k for k in state_dict if k.startswith("net_g.")]
        if stale:
            state_dict = {k: v for k, v in state_dict.items()
                          if not k.startswith("net_g.")}
            print(f"NOTE: ignored {len(stale)} stale 'net_g.*' tensor(s) in the "
                  f"checkpoint -- the frozen stage-1 Generator used to be saved "
                  f"as a TrainModule submodule. It is loaded from "
                  f"train.generator_checkpoint instead.")
        return super().load_state_dict(state_dict, strict=strict)

    def _reset_text_delta(self):
        """Clear the per-epoch accumulators of the text-conditioning probe."""
        self._text_delta_sum = 0.0
        self._text_delta_spec_sum = 0.0
        self._text_delta_pitch_sum = 0.0
        self._text_delta_count = 0

    def _reset_hint_probe(self):
        """Clear the per-epoch accumulators of the hint-oracle probe."""
        self._hint_nohint_sum = 0.0
        self._hint_gap_sum = 0.0
        self._hint_nohint_count = 0

    def _hints_off(self, batch):
        """``batch`` with every hint channel dropped to its "absent" state.

        Level 0 is exactly what ``dataset_codes`` emits for a clip whose hints
        were dropped (or never built), i.e. the state the model has been trained
        to handle -- not an out-of-distribution input.
        """
        if "hint_int" not in batch:
            return None
        off = dict(batch)
        off["hint_int"] = torch.zeros_like(batch["hint_int"])
        for key in ("hint_reg", "hint_pitch", "hint_slot",
                    # Hints V2 anchors: state 0 is the trained absent path.
                    "hint_pitch_val", "hint_pitch_pos", "hint_pitch_n",
                    "hint_pitch_state", "hint_phone_frac",
                    # V2.1 word states: the mask is the absent state.
                    "hint_word", "hint_word_valid"):
            if batch.get(key) is not None:
                off[key] = torch.zeros_like(batch[key])
        return off

    def _reset_hint_channel_probe(self):
        """Clear the per-epoch accumulators of the per-channel hint probe."""
        self._hint_channel_sum = {}
        self._hint_channel_count = 0

    def _accumulate_hint_channel(self, batch):
        """Per-channel CE deltas for one validation batch (see ``_log_*``)."""
        deltas = self.net.hint_channel_deltas(batch)
        if not deltas:
            return
        self._hint_channel_count += 1
        for key, value in deltas.items():
            self._hint_channel_sum[key] = (self._hint_channel_sum.get(key, 0.0)
                                           + float(value))

    def _log_hint_channel_probe(self):
        """Log ``val_hint_channel_delta/*`` and print the per-channel table."""
        if self.hint_channel_probe <= 0 or not self._hint_channel_count:
            return
        n = float(self._hint_channel_count)
        base = self._hint_channel_sum.get("base", 0.0) / n
        self._add_scalar("val_hint_channel_base", torch.tensor(base))
        parts = []
        for name in (list(self.net.HINT_CHANNEL_KEYS)
                     + ["all", "omit"]):
            if name not in self._hint_channel_sum:
                continue
            delta = self._hint_channel_sum[name] / n
            self._add_scalar(f"val_hint_channel_delta/{name}",
                             torch.tensor(delta))
            parts.append(f"{name} {delta:+.4f}")
        print(f"val_hint_channel_delta (nats, CE[channel dropped] - CE[all "
              f"hints], base {base:.4f}): " + ", ".join(parts))

    def _sample_hint_arg(self, batch=None, mode=None):
        """``(hints argument, log tag)`` for one **generation** call.

        ``None`` is not a neutral value here: ``generate`` resolves it to the
        trained *null state* (every channel absent), whereas ``"omit"`` drops
        the vectors entirely.  Both are reachable, but they are different
        functions and the difference is visible in the output (+0.051 nats and
        a 52 % encoder-memory shift on transformer_v0.9), so every sample log
        carries the state as a tag and as ``sample/hints_mode``.

        Tag values: ``"_hint"`` (real ``.cond``), ``"_nohint"`` (trained null
        state), ``"_omit"`` (no vectors), ``""`` (model has no hint tables, so
        the question does not arise -- tag names stay as they were pre-v0.8).
        """
        mode = self.sample_hints if mode is None else mode
        if not getattr(self.net, "hints", False):
            return None, ""
        if mode == "cond" and batch is not None:
            return batch, "_hint"
        if mode == "omit":
            return "omit", "_omit"
        return "null", "_nohint"

    def _hint_mode_id(self, mode=None) -> int:
        """0 = omit, 1 = null state, 2 = real ``.cond`` (for the TB scalar)."""
        mode = self.sample_hints if mode is None else mode
        return {"omit": 0, "null": 1, "cond": 2}.get(mode, 1)

    def _accumulate_hint_gap(self, batch, out):

        """``CE(no hints) - CE(with hints)`` for one validation batch.

        The hints are derived from the *target audio*, so at train time they are
        an oracle; ``val_loss`` therefore cannot tell "the model got better"
        from "the model reads the answer".  This probe re-runs the exact same
        batch with every channel dropped and reports both the no-hint loss and
        the gap.  Read them together with ``val_loss``:

        * gap ~ 0                       -> the hints are ignored (dropout too
                                           aggressive, or the tables are dead);
        * gap small and flat            -> the model uses them as a nudge;
        * gap growing while
          ``val_loss_nohint`` rises     -> it is trading unconditional quality
                                           for hint-following (the VC failure
                                           mode).
        """
        off = self._hints_off(batch)
        if off is None:
            return
        nohint = self.net.compute_batch_loss(off, epoch=None, diagnostics=False)
        self._hint_nohint_sum += float(nohint["loss"])
        self._hint_gap_sum += float(nohint["loss"] - out["loss"])
        self._hint_nohint_count += 1

    def _log_hint_probe(self):
        """Log ``val_loss_nohint`` and ``val_hint_gap`` for the epoch."""
        if self.hint_probe_batches <= 0 or not self._hint_nohint_count:
            return
        n = float(self._hint_nohint_count)
        nohint = self._hint_nohint_sum / n
        gap = self._hint_gap_sum / n
        self._add_scalar("val_loss_nohint", torch.tensor(nohint))
        self._add_scalar("val_hint_gap", torch.tensor(gap))
        print(f"val_loss_nohint = {nohint:.4f}, val_hint_gap = {gap:+.4f} nats "
              f"(CE with every hint dropped - CE with them)")

    def _reset_fim_probe(self):
        """Clear the per-epoch accumulators of the FIM diagnostics probe."""
        # Base (correct inputs) middle CE / accuracy plus the three control arms.
        self._fim_base = 0.0
        self._fim_text = 0.0
        self._fim_suffix = 0.0
        self._fim_region = 0.0
        self._fim_length = 0.0
        self._fim_acc = 0.0
        self._fim_acc_bins = [0.0, 0.0, 0.0]
        # Seam diagnostics: ``fim_seam_acc`` (last N middle targets) and the
        # lookahead CE/acc (the transition, only on a ``fim_lookahead`` model).
        self._fim_seam_acc = 0.0
        self._fim_seam_count = 0
        self._fim_la_loss = 0.0
        self._fim_la_acc = 0.0
        self._fim_la_count = 0
        self._fim_length_count = 0
        self._fim_probe_count = 0

    def setup(self, stage=None):
        hp = self.config
        val_filelist = hp.train.val_filelist
        self.test_dataset = CodeDataset(
            val_filelist,
            is_train=False,
            max_frames=hp.train.get("max_frames", None),
            max_phone_length=hp.train.get("max_phone_length", None),
            speaker_map=hp.train.get("speaker_map", None),
        )
        self.test_collator = CodeCollator(config=hp)
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=1, shuffle=False,
            collate_fn=self.test_collator, num_workers=0)

        ckpt = hp.train.get("generator_checkpoint", None)
        if ckpt is not None and os.path.exists(ckpt):
            try:
                # Held outside the module tree on purpose (see the `net_g`
                # property): assigning a Generator to a plain attribute would
                # register it as a submodule and write it into every checkpoint.
                self._net_g_holder[0] = load_generator(ckpt, hp, self.device)
                for p in self.net_g.parameters():
                    p.requires_grad = False
                print(f"Loaded stage-1 Generator for sampling: {ckpt}")
            except Exception as e:  # sampling is optional
                print(f"Could not load Generator ({ckpt}): {e}")
                self._net_g_holder[0] = None
        elif ckpt is not None:
            # Worth saying out loud: this used to be the difference between a
            # resumable and an unresumable run (see `load_state_dict`). It is
            # harmless now -- audio/infill logging is simply off.
            print(f"NOTE: train.generator_checkpoint does not exist "
                  f"({ckpt}); sampling still works, audio decode is off.")
        self.hp = hp

    # ------------------------------------------------------------------ #
    # Parameter split / optimizer knobs
    # ------------------------------------------------------------------ #
    def _split_params(self):
        """Split parameters into ``(muon_params, adamw_params)``.

        Muon orthogonalises the update matrix of every 2-D parameter, which is
        the right thing for the hidden *linear maps* of a transformer and the
        wrong thing for two other kinds of 2-D tensor:

        * ``nn.Embedding`` tables (the phoneme / speaker / codebook
          embeddings).  Only the rows of the ids that occur in the batch get a
          gradient, and Newton-Schulz costs ~26 GMAC per iteration batch for
          the 4096-row speaker tables -- a per-step cost that is almost
          entirely wasted, because the update is a linear combination of the
          *rows* of the gradient.
        * the per-codebook output ``heads`` (66/130 x 768).  These are the CE
          projection, i.e. a loss head, and a handful of rows: reference Muon
          excludes output heads for a reason.

        ``torch.optim.Muon._adjust_lr`` also scales the LR by
        ``sqrt(max(1, rows/cols))`` -- 2.3x for a 4096x768 embedding table
        versus 1.0 for the square attention matrices -- so routing them to
        AdamW also removes that accidental LR bump.

        Both exemptions are configurable (``train.muon_include_embeddings`` /
        ``train.muon_include_heads``, both default ``false``) for reproducing
        older runs.

        Note: changing this split changes the *layout* of the optimizer state,
        so a ``--resume_from`` of an older checkpoint keeps the weights but
        discards the momentum/Adam moments (``_LayoutGuard`` says so loudly).
        """
        include_emb = bool(
            self.config.train.get("muon_include_embeddings", False))
        include_heads = bool(
            self.config.train.get("muon_include_heads", False))
        skip = set()
        if not include_emb:
            for m in self.net.modules():
                if isinstance(m, torch.nn.Embedding):
                    skip.add(id(m.weight))
        if not include_heads:
            heads = getattr(getattr(self.net, "decoder", None), "heads", None)
            if heads is not None:
                for m in heads.modules():
                    if isinstance(m, torch.nn.Linear):
                        skip.add(id(m.weight))

        muon_params, adamw_params = [], []
        for p in self.net.parameters():
            if p.ndim == 2 and id(p) not in skip:
                muon_params.append(p)
            else:
                adamw_params.append(p)
        return muon_params, adamw_params

    def configure_optimizers(self):
        """Muon on the 2-D matrices, AdamW on everything else (biases, norms).

        Two optimizers on disjoint parameter sets, both fed by the same loss,
        so the module runs with ``automatic_optimization = False`` (see
        ``__init__``).  The optimizers/schedulers are returned in a fixed order
        and `training_step` unpacks them by position.

        ``train.muon_ns_steps`` controls Muon's Newton-Schulz iteration count.
        Its cost is *fixed per step* -- it does not scale with the batch -- and
        measured on this model it is ~1.26 TMAC, i.e. ~75 % of the
        forward+backward FLOPs, delivered as 5 iterations x 3 matmuls x 119
        parameters in a per-parameter Python loop (``torch.optim.Muon`` has no
        ``foreach`` path).  Dropping to 3 steps removes 40 % of that.

        In fine-tune mode (``--finetune``) every hidden 2-D matrix is frozen, so
        Muon has nothing to orthogonalise and the run uses a single AdamW over
        the adapters (plus the unfrozen speaker rows) at ``finetune.lr``.  See
        ``_configure_finetune_optimizer``.
        """
        if self.finetune_enabled:
            return self._configure_finetune_optimizer()
        muon_params, adamw_params = self._split_params()
        if not muon_params or not adamw_params:
            raise ValueError(
                "Expected both 2-D (Muon) and non-2-D (AdamW) parameters, got "
                f"{len(muon_params)} and {len(adamw_params)}")
        weight_decay = float(self.config.train.get("weight_decay", 0.0) or 0.0)
        # Muon normalises its update onto the orthogonal group, so it needs a
        # much larger step than Adam(W): the usual recipe is ~0.02 for the
        # hidden layers, i.e. the AdamW LR scaled by ~200.
        muon_lr_mult = float(self.config.train.get("muon_lr_mult", 200.0))
        optim_adamw = LayoutCheckedAdamW(
            adamw_params,
            lr=self.config.train.lr,
            betas=tuple(self.config.train.betas),
            eps=self.config.train.eps,
            weight_decay=weight_decay,
        )
        optim_muon = LayoutCheckedMuon(
            muon_params,
            lr=self.config.train.lr * muon_lr_mult,
            weight_decay=weight_decay,
            ns_steps=self.ns_steps,
        )
        sched_adamw = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optim_adamw,
            T_max=self.config.train.get("cosine_anneal_end", 1000000),
            eta_min=1e-6,
        )
        sched_muon = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optim_muon,
            T_max=self.config.train.get("cosine_anneal_end", 1000000),
            eta_min=1e-6 * muon_lr_mult,
        )
        n_muon = sum(p.numel() for p in muon_params)
        n_adamw = sum(p.numel() for p in adamw_params)
        print(f"Optimizers: Muon {len(muon_params)} 2-D tensors "
              f"({n_muon/1e6:.1f} M) @ lr="
              f"{optim_muon.param_groups[0]['lr']:g}, ns_steps={self.ns_steps}; "
              f"AdamW {len(adamw_params)} tensors ({n_adamw/1e6:.1f} M) @ "
              f"lr={optim_adamw.param_groups[0]['lr']:g} "
              f"(weight_decay={weight_decay:g})")
        return [optim_adamw, optim_muon], [sched_adamw, sched_muon]

    def _configure_finetune_optimizer(self):
        """One AdamW over the LoRA adapters (``--finetune``).

        Muon is deliberately not used: it is defined for the hidden linear maps,
        which are all frozen here, and its ``_adjust_lr`` scaling
        (``sqrt(rows/cols)``) is meaningless for a rank-16 A/B pair.  The
        adapters are small (a few M parameters), so the step cost of this mode
        is essentially the frozen base model's forward+backward.

        ``finetune.lr`` is a *separate* knob from ``train.lr`` (the base run's
        schedule is not the right operating point for fresh low-rank weights:
        measured full-fine-tunes of this model use 5e-6, while LoRA wants ~1e-3).
        """
        ft = self.finetune_cfg
        params = [p for p in self.net.parameters() if p.requires_grad]
        if not params:
            raise ValueError(
                "finetune is enabled but no parameter has requires_grad=True; "
                "check finetune.targets and transformer.rope (decoder_qkv only "
                "exists on the rotary attention).")
        n_params = sum(p.numel() for p in params)
        n_lora = sum(p.numel() for p in lora_parameters(self.net))
        anneal_end = int(
            ft["cosine_anneal_end"]
            or self.config.train.get("cosine_anneal_end", 1000000))
        optim = LayoutCheckedAdamW(
            params,
            lr=ft["lr"],
            betas=tuple(self.config.train.betas),
            eps=float(self.config.train.eps),
            weight_decay=ft["weight_decay"],
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optim, T_max=anneal_end, eta_min=ft["lr"] * 0.01)
        print(f"Optimizers: AdamW (finetune) {len(params)} tensors "
              f"({n_params/1e6:.2f} M, of which {n_lora/1e6:.2f} M adapters) "
              f"@ lr={ft['lr']:g} (weight_decay={ft['weight_decay']:g}), "
              f"cosine anneal over {anneal_end} steps")
        return [optim], [sched]

    def _apply_muon_settings(self):
        """Re-apply ``train.muon_ns_steps`` after a checkpoint restore.

        ``Optimizer.load_state_dict`` restores ``param_groups`` wholesale, and
        ``ns_steps`` lives in the Muon group's dict — so a ``--resume_from``
        would otherwise silently keep the value the checkpoint was trained
        with, no matter what the config says.  Called from ``on_train_start``,
        which runs after Lightning has restored the optimizers.
        """
        optims = _optimizer_list(self.optimizers())
        if len(optims) < 2:      # fine-tune mode has a single AdamW
            return
        muon = getattr(optims[1], "optimizer", optims[1])  # unwrap LightningOptimizer
        changed = False
        for group in muon.param_groups:
            if group.get("ns_steps") != self.ns_steps:
                group["ns_steps"] = self.ns_steps
                changed = True
        if changed:
            print(f"Muon: ns_steps forced to {self.ns_steps} "
                  f"(the checkpoint's value was overridden)")

    def _maybe_compile_decoder(self):
        """Optionally ``torch.compile`` the codebook decoder stack.

        ``train.compile_decoder`` (default ``false``).  Same pattern as stage
        1's ``compile_dec``:

        * the compile target is ``DelayCodebookDecoder._forward_impl`` with
          ``use_mask_cache=False``, **not** the module — wrapping the module
          returns an ``OptimizedModule`` whose ``_orig_mod`` child renames every
          decoder key in ``state_dict()`` and breaks checkpoint loading;
        * ``dynamic=True`` because validation/sampling batches have a different
          column count (``L = T + K - 1``, and val items are not cropped to
          ``max_frames``);
        * the warm-up call compiles up front and turns a compile failure into a
          printed fallback to eager instead of dying at step 0.  Its RNG side
          effects are undone, so on/off differ only in speed.
        """
        if not bool(self.config.train.get("compile_decoder", False)):
            return
        max_frames = int(self.config.train.get("max_frames", 256))
        K = self.net.codebook_num
        H = self.net.hidden_size
        L = max_frames + K - 1
        codes = torch.zeros(1, K, L, dtype=torch.long, device=self.device)
        memory = torch.zeros(1, 4, H, device=self.device)
        memory_mask = torch.ones(1, 4, dtype=torch.bool, device=self.device)
        sid = torch.zeros(1, dtype=torch.long, device=self.device)
        cpu_rng = torch.get_rng_state()
        cuda_rng = (torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None)
        impl = self.net.decoder._forward_impl

        def compiled_forward(codes, memory, memory_mask=None, sid=None,
                             region=None, length=None, hint_vec=None):
            # Explicit signature (not a ``lambda``) because ``code_loss`` and
            # ``infill`` call the decoder as ``decoder(inputs, memory,
            # memory_mask, sid=sid, region=region, length=length,
            # hint_vec=...)`` and ``torch.compile`` rejects unknown keyword
            # arguments.  Keep the positional order of ``_forward_impl``.
            return impl(codes, memory, memory_mask, sid, region, length,
                        hint_vec, False)

        try:
            compiled = torch.compile(compiled_forward, dynamic=True)
            compiled(codes, memory, memory_mask, sid, None, None)
            self.net.decoder.forward = compiled
            print("torch.compile: decoder (DelayCodebookDecoder._forward_impl)")
        except Exception as exc:
            print(f"torch.compile failed for the decoder, keeping eager: {exc}")
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

    def on_train_start(self):
        self._apply_muon_settings()
        self._check_anneal_end()
        self._sample_and_log()
        self._maybe_compile_decoder()

    def _check_anneal_end(self, step=None, lr=None):
        """Warn once when the run is past ``train.cosine_anneal_end``.

        ``torch.optim.lr_scheduler.CosineAnnealingLR`` wraps around: with
        ``T_max = cosine_anneal_end`` the LR goes ``base -> eta_min`` over the
        first ``T_max`` steps and then ``eta_min -> base`` over the next
        ``T_max``.  A run that keeps going therefore *raises* its learning rate
        again -- typically to the full ``train.lr`` at step
        ``2 * cosine_anneal_end`` -- while the key name implies it is pinned at
        the end of the anneal.  Two ways this happens in practice:

        * ``train.max_steps`` is larger than ``cosine_anneal_end`` (the shipped
          stage-2 config has 1 M vs 300 k), so nothing stops the run at the
          minimum and the tail re-warms,
        * a ``--resume_from`` of a checkpoint whose step is already past it.

        The warning is printed once per run (``self._anneal_warned``) and says
        where the LR is heading, so "the loss changed direction" is attributable
        without digging through the checkpoint's ``lr_schedulers``.
        """
        if self._anneal_warned or self.anneal_end <= 0:
            return
        step = self.global_step if step is None else int(step)
        if step < self.anneal_end:
            return
        self._anneal_warned = True
        if lr is None:
            optims = _optimizer_list(self.optimizers())
            adamw = getattr(optims[0], "optimizer", optims[0])
            lr = float(adamw.param_groups[0]["lr"])
        base_lr = float(self.config.train.get("lr", lr))
        print(
            f"WARNING: global_step {step} is past train.cosine_anneal_end "
            f"({self.anneal_end}). CosineAnnealingLR is periodic, so the LR is "
            f"now *rising* again (currently {lr:.3e}, back to the base "
            f"{base_lr:.3e} at step {2 * self.anneal_end}) -- the run is no "
            f"longer annealing. Set train.max_steps to cosine_anneal_end (or "
            f"raise cosine_anneal_end) if that is not what you want.")

    def on_train_epoch_end(self):
        self._log_grad_epoch_stats()
        self._log_hint_coverage()
        self._sample_and_log()
        self._wer_probe()
        return super().on_train_epoch_end()

    def _log_hint_coverage(self):
        """Print **once** how many training clips had a usable ``.cond``.

        The dataset degrades a missing/stale ``.cond`` to "no hints" instead of
        failing, so a corpus that was only partly built would otherwise train
        quietly with a fraction of the items hinted.  This runs at the end of the
        first epoch, i.e. after every item has been read once.
        """
        if self._hints_reported:
            return
        loader = getattr(self.trainer, "train_dataloader", None)
        ds = getattr(loader, "dataset", None)
        if ds is None:
            return
        self._hints_reported = True
        miss = int(getattr(ds, "n_cond_missing", 0))
        stale = int(getattr(ds, "n_cond_stale", 0))
        fallback = int(getattr(ds, "n_word_crop_fallback", 0))
        if fallback:
            print(f"Word crop: {fallback} long clip(s) had no usable word "
                  f"timings and kept the full transcript")
        if not getattr(ds, "hints", False):
            return
        print(f"Hints: {miss} clip(s) without a .cond, {stale} stale "
              f"(n_phone mismatch or unreadable .cond)")
        if miss or stale:
            print("       -> missing hints only cost conditioning quality (those "
                  "clips train unconditionally), but build them before the next "
                  "run: python hints/build.py --filelist <train filelist> "
                  "--wave-dir <stage1 dir> --source-filelist <transcripts> "
                  "--hints intensity,register,semantic")

    def _log_grad_epoch_stats(self):
        """Log the worst gradient norm seen this epoch and how often clipping hit.

        ``train_grad_norm`` is only sampled every ``log_interval`` steps, so a
        single spiking step can slip between two log points; these two scalars
        are computed on *every* step (when clipping is enabled) and make such
        spikes — and whether ``grad_clip_val`` is doing anything — visible.
        """
        if (self.logger is None or self.grad_clip_val <= 0
                or not self._grad_steps):
            return
        self._add_scalar("train_grad_norm_max", self._grad_norm_max)
        frac = self._grad_clipped_steps / self._grad_steps  # tensor once clipped
        self._add_scalar("train_grad_clip_frac", torch.as_tensor(frac))

    # ------------------------------------------------------------------ #
    # Optional sampling / audio + visual logging
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _sample_and_log(self):
        """Decode a few val items and log codes/audio/pitch to TensorBoard."""
        interval = int(self.config.train.get("sample_interval", 1) or 0)
        if interval <= 0 or self.current_epoch % interval:
            return
        if self.logger is None or len(self.test_dataset) == 0:
            return
        hint_desc = {"cond": "real .cond", "null": "trained null state",
                     "omit": "OMITTED (out of distribution)"}[self.sample_hints]
        if not getattr(self.net, "hints", False):
            hint_desc = "n/a (model has no hint tables)"
        print(f"=== Sampling (hints: {hint_desc}) ===")
        if self.logger is not None:
            self._add_scalar("sample/hints_mode", torch.tensor(
                float(self._hint_mode_id())))
        self.net.eval()
        n = int(self.config.train.get("sample_count", 2))
        frames = int(self.config.train.get("sample_frames", 128))
        temperature = float(self.config.train.get("sample_temperature", 1.0))
        top_k = self.config.train.get("sample_top_k", None)
        top_k = int(top_k) if top_k else None
        probe_n = int(self.config.train.get("speaker_probe", 0))

        # Speakers available in the val set, used for the conditioning probe.
        candidates = []
        if probe_n > 0:
            for s in self.test_dataset.speaker_ids():
                if s not in candidates:
                    candidates.append(s)

        # Speaker for the OOD probe: the first sampled val item's voice, so the
        # generated speech is in a voice the model has actually seen.
        ood_sid = None

        for j, batch in enumerate(itertools.islice(self.test_dataloader, n)):
            batch = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            code_len = int(batch["code_lengths"][0].item())
            gen_frames = max(1, min(frames, code_len))
            hint_arg, hint_tag = self._sample_hint_arg(batch)

            codes, gen_lens = self.net.generate(
                batch["phone_ids"], batch["phone_lengths"], batch["sid"],
                max_frames=gen_frames, temperature=temperature, top_k=top_k,
                return_lengths=True, hints=hint_arg,
                hint_guidance=self.sample_hint_guidance,
            )  # (B, K, n) with n <= gen_frames when EOS stopped it early

            n_gen = int(codes.shape[-1])
            if self.net.eos_token:
                # How many frames the model decided to produce (max over the
                # batch): with EOS this is the "did it learn to stop?" signal.
                self._add_scalar(f"sample{hint_tag}/gen_frames_{j}",
                                 gen_lens.max())

            # Ground truth for the same item (spec rows, then pitch rows).
            gt = torch.cat([batch["spec_codes"][0], batch["pitch_codes"][0]],
                           dim=0)[:, :n_gen]

            self._log_code_maps(j, gt, codes[0], hint_tag)

            if self.net_g is not None:
                self._log_audio(j, gt, codes[0], batch["sid"], hint_tag)

            if self.net.fim:
                # Regenerate a fixed-length middle span of the *ground-truth*
                # item and log target-vs-regenerated codes/audio.  This is the
                # direct "does infill work?" view (the free-running samples above
                # are plain AR generation).
                try:
                    self._log_infill(j, batch, temperature, top_k,
                                     hint_arg, hint_tag)
                except Exception as e:
                    print(f"FIM infill probe failed: {e}")

            if probe_n > 0 and len(candidates) > 1:
                own = int(batch["sid"][0].item())
                probe = [own] + [s for s in candidates if s != own][:probe_n]
                self._log_speaker_probe(j, batch, probe, gen_frames,
                                        temperature, top_k,
                                        hint_arg, hint_tag)


            if ood_sid is None:
                ood_sid = batch["sid"][:1]

        # Held-out text, generated *longer* than any val item: unlike the
        # samples above (capped at `code_len`), this has to keep decoding past
        # the length the model ever saw, which is where autoregressive
        # degradation (repetition, drift) shows up first.
        if ood_sid is not None:
            try:
                self._log_ood_generation(ood_sid, temperature, top_k)
            except Exception as e:  # sampling must never break training
                print(f"OOD generation failed: {e}")

        if self.logger is not None:
            self.logger.experiment.flush()

    # ------------------------------------------------------------------ #
    # Visualisation loggers (each is a no-op without a TensorBoard logger)
    # ------------------------------------------------------------------ #
    def _add_scalar(self, tag, value):
        if self.logger is None or not torch.is_tensor(value):
            return
        if value.numel() == 0:
            return
        self.logger.experiment.add_scalar(
            tag, float(value.detach().float().mean()), self.global_step)

    def _add_image(self, tag, fig):
        if fig is None:
            return
        if self.logger is not None:
            self.logger.experiment.add_image(
                tag, matplotlib_to_tensorboard(fig),
                global_step=self.global_step)
        plt.close(fig)

    def _add_audio(self, tag, audio, sample_rate):
        if self.logger is None:
            return
        audio = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1),
                        -1.0, 1.0)
        if audio.size == 0:
            return
        self.logger.experiment.add_audio(
            tag, audio, global_step=self.global_step, sample_rate=sample_rate)

    @staticmethod
    def _hint_desc(tag: str) -> str:
        """Human-readable hint state for figure titles (``""`` = no tables)."""
        return {"_hint": "with hints", "_nohint": "null hints",
                "_omit": "hints omitted"}.get(tag, "")

    def _log_code_maps(self, j, gt, pred, tag: str = ""):
        """Target-vs-generated codebook heatmaps, one figure per group.

        ``tag`` names the hint state that produced ``pred`` (see
        ``_sample_hint_arg``) and is part of both the image and the scalar tag,
        so "which state was this sample generated in?" is visible in
        TensorBoard instead of being a matter of what the code passed.
        """
        n_spec = self.net.spec_codebook_num
        desc = self._hint_desc(tag)
        groups = (
            ("spec", gt[:n_spec], pred[:n_spec], self.net.vocab_sizes[0]),
            ("pitch", gt[n_spec:], pred[n_spec:], self.net.vocab_sizes[-1]),
        )
        for name, g, p, pad in groups:
            if g.shape[0] == 0:
                continue
            labels = [f"book {i}" for i in range(g.shape[0])]
            title = f"{name} codes @ step {self.global_step}"
            if desc:
                title = f"{name} codes ({desc}) @ step {self.global_step}"
            self._add_image(
                f"codes{tag}/{name}_{j}",
                plot_code_comparison_figure(g, p, pad, title, labels))
            # Free-running agreement on real (non-pad) target frames: unlike
            # teacher-forced accuracy this is the number that decides whether
            # the generated audio is intelligible.
            valid = g != int(pad)
            if bool(valid.any()):
                self._add_scalar(
                    f"sample{tag}/{name}_code_match_frac_{j}",
                    (g == p).float()[valid].mean())


    def _gen_sid_map(self):
        """Stage-2 -> stage-1 id map (``train.stage1_speaker_map``), lazily."""
        if self._gen_sid_map_cache is None:
            from stage2_wer import load_sid_map

            self._gen_sid_map_cache = load_sid_map(self.stage1_speaker_map)
            if self._gen_sid_map_cache:
                print(f"Stage-1 decode speaker map: "
                      f"{len(self._gen_sid_map_cache)} id(s) from "
                      f"{self.stage1_speaker_map} (the transformer keeps its "
                      f"own ids; only infer_from_codes is remapped)")
        return self._gen_sid_map_cache

    def _gen_sid(self, sid):
        """Map a stage-2 speaker id to the stage-1 id the Generator renders."""
        mapping = self._gen_sid_map()
        if not mapping:
            return sid
        flat = [mapping.get(int(s), int(s)) for s in sid.reshape(-1)]
        return torch.tensor(flat, dtype=sid.dtype, device=sid.device).view(
            sid.shape)

    def _decode_codes(self, codes, sid):
        """(K, T) codes -> (waveform, f0 in Hz (T,)) through stage 1."""
        n_spec = self.net.spec_codebook_num
        audio, f0 = self.net_g.infer_from_codes(
            codes[:n_spec].unsqueeze(1), codes[n_spec:].unsqueeze(1),
            self._gen_sid(sid), return_pitch=True)
        return audio[0].squeeze(0).float().cpu().numpy(), f0[0]

    def _log_audio(self, j, gt, pred, sid, tag: str = ""):
        try:
            audio_gt, f0_gt = self._decode_codes(gt, sid)
            audio_pred, f0_pred = self._decode_codes(pred, sid)
        except Exception as e:  # sampling is optional
            print(f"Sample synthesis failed: {e}")
            return
        sr = int(self.config.data.sampling_rate)
        hop = int(self.config.data.hop_length)
        desc = self._hint_desc(tag)
        self._add_audio(f"audio{tag}/target_{j}", audio_gt, sr)
        self._add_audio(f"audio{tag}/generated_{j}", audio_pred, sr)
        pitch_title = f"f0 decoded from codes @ step {self.global_step}"
        mel_title = f"log-mel @ step {self.global_step}"
        if desc:
            pitch_title = f"f0 ({desc}) @ step {self.global_step}"
            mel_title = f"log-mel ({desc}) @ step {self.global_step}"
        self._add_image(
            f"pitch{tag}/{j}",
            plot_pitch_curves_figure(
                [("target codes", f0_gt), ("generated codes", f0_pred)],
                hop, sr, pitch_title))
        self._add_image(
            f"mel{tag}/{j}",
            plot_mel_comparison_figure(audio_gt, audio_pred, sr, hop,
                                       mel_title))


    def _log_infill(self, j, batch, temperature, top_k, hint_arg="null",
                    tag: str = ""):
        """Regenerate a fixed-length middle span of a ground-truth item.

        Picks ``prefix = T//3`` and ``middle = T//3`` frames (leaving a suffix),
        blanks the middle, calls :meth:`CodeTransformer.infill` with the known
        prefix/suffix, and logs target-vs-regenerated codes (and audio when a
        stage-1 Generator is loaded).  ``sample{tag}/fim_middle_code_match_{j}``
        is the free-running agreement of the regenerated middle with the
        original; it is a *sample* number (voice sampling variance means it is
        never 1.0 for a good model), but a value near chance means the middle is
        ignored.  ``hint_arg`` is resolved by ``_sample_hint_arg`` and ``tag``
        names that state in the log.
        """
        T = int(batch["code_lengths"][0].item())
        if T < 6:
            return
        p = max(1, T // 3)
        m = max(2, T // 3)
        if p + m >= T:
            m = T - p - 1
        if m < 2:
            return
        n_spec = self.net.spec_codebook_num
        codes = torch.cat([batch["spec_codes"][0], batch["pitch_codes"][0]],
                          dim=0)[:, :T]                    # (K, T)
        prefix = codes[:, :p].unsqueeze(0)
        suffix = codes[:, p + m:T].unsqueeze(0)
        middle = self.net.infill(
            batch["phone_ids"][:1], batch["phone_lengths"][:1], batch["sid"][:1],
            prefix, suffix, m, temperature=temperature, top_k=top_k,
            hints=hint_arg, hint_guidance=self.sample_hint_guidance,
        )                                                       # (1, K, m)

        regen = torch.cat([prefix, middle, suffix], dim=2)[0]  # (K, T)
        self._log_code_maps(f"fim{j}", codes, regen, tag)
        match = (middle[0, :, :m] == codes[:, p:p + m]).float().mean()
        self._add_scalar(f"sample{tag}/fim_middle_code_match_{j}", match)
        self._add_scalar(f"sample{tag}/fim_spec_code_match_{j}",
                         (middle[0, :n_spec, :m] == codes[:n_spec, p:p + m])
                         .float().mean())
        if self.net.fim_lookahead > 0:
            # The seam number: teacher-forced CE of the real suffix start given
            # the regenerated span.  Unlike `fim_middle_code_match_*` (free-
            # running, at chance even for a healthy model) this is a readable
            # "does the join land?" signal, and the same quantity
            # `infill(best_of=N)` ranks candidates by.
            seam_ce = self.net.infill_seam_score(
                batch["phone_ids"][:1], batch["phone_lengths"][:1],
                batch["sid"][:1], prefix, middle, suffix, hints=hint_arg,
                hint_guidance=self.sample_hint_guidance)
            self._add_scalar(f"sample{tag}/fim_seam_ce_{j}", seam_ce.mean())
        if self.net_g is not None:
            self._log_audio(f"fim{j}", codes, regen, batch["sid"][:1], tag)

    # ------------------------------------------------------------------ #
    # End-to-end WER probe (shared with scratch/eval_stage2_wer.py)
    # ------------------------------------------------------------------ #
    def _wer_setup(self):
        """Build the WER eval inputs once: dataset, references, items, ASR.

        Deliberately separate from ``self.test_dataset``: that one crops to
        ``train.max_frames`` and is used for the loss, while WER must score the
        whole utterance the reference transcript describes.  The hint channels
        are built when the model has them (so ``wer_hints: cond`` is possible),
        and the ASR weights are loaded here, i.e. only on the first probe.
        """
        import stage2_wer as W

        config = self.config
        sources = W.resolve_source_filelists(
            config.train.get("wer_source_filelists", None))
        refs_raw = W.load_references(sources)
        refs = W.apply_wer_reference_audit(
            dict(refs_raw), config.train.get("wer_reference_audit", None),
            mode=str(config.train.get("wer_reference_audit_mode", "drop")
                     or "drop"))
        dataset = W.build_wer_dataset(config)
        pool = [i for i in range(len(dataset))
                if W.item_basename(dataset.items[i]["spec_codes"]) in refs_raw]
        picked, items, texts = W.select_items(
            dataset, refs, self.wer_items, avail=pool, require_ref=False)
        device = self.device
        asr = W.Asr(str(config.train.get("wer_asr_model",
                                         "Qwen/Qwen3-ASR-0.6B")),
                    device, str(config.train.get("wer_language", "English")))
        hint_fn = None
        if self.wer_hints == "cond":
            if not getattr(dataset, "hints", False):
                print("WER: wer_hints=cond but the dataset built without hint "
                      "channels; falling back to the trained null state.")
            else:
                from dataset_codes import CodeCollator
                collator = CodeCollator(config=config)

                def hint_fn(chunk):
                    return {k: (v.to(device) if torch.is_tensor(v) else v)
                            for k, v in collator(chunk).items()}
        elif self.wer_hints not in ("null", "zero"):
            print(f"WER: unknown wer_hints={self.wer_hints!r}; using null.")
            self.wer_hints = "null"
        print(f"WER probe: {len(items)} val items, hints={self.wer_hints}, "
              f"asr={self.config.train.get('wer_asr_model', 'Qwen/Qwen3-ASR-0.6B')}")
        return {"items": items, "texts": texts, "picked": picked,
                "dataset": dataset, "asr": asr, "hint_fn": hint_fn,
                "sid_map": self._gen_sid_map(),
                "sample_rate": int(config.data.sampling_rate)}

    @torch.no_grad()
    def _wer_probe(self):
        """Log ``wer/{generated,fim,reference}`` every ``train.wer_interval``."""
        if self.wer_interval <= 0 or self.current_epoch % self.wer_interval:
            return
        if self._wer_failed:
            return
        if self.net_g is None:
            print("WER: skipped -- no stage-1 Generator (train.generator_checkpoint).")
            self._wer_failed = True
            return
        try:
            if self._wer_state is None:
                self._wer_state = self._wer_setup()
        except Exception as e:  # a probe must never break training
            print(f"WER: setup failed ({e}); disabling the probe for this run.")
            self._wer_failed = True
            return

        import stage2_wer as W

        st = self._wer_state
        items, texts = st["items"], st["texts"]
        sample_rate = st["sample_rate"]
        cfg = self.config.train
        temperature = cfg.get("wer_temperature", None)
        if temperature is None:
            temperature = cfg.get("sample_temperature", 0.9)
        temperature = float(temperature)
        top_k = cfg.get("wer_top_k", None)
        if top_k is None:
            top_k = cfg.get("sample_top_k", None)
        top_k = int(top_k) if top_k else None
        max_frames = int(cfg.get("wer_max_frames", 1000) or 1000)
        batch_size = int(cfg.get("wer_batch_size", 4) or 4)
        ignore_eos = bool(cfg.get("wer_ignore_eos", False))
        seed = int(cfg.get("wer_seed", 0) or 0)
        phone_pad = int(self.config.transformer.get("phone_pad_id", 0) or 0)

        self.net.eval()
        results = {}
        # Keep the probe from perturbing the training RNG stream (FIM spans,
        # hint dropout, ... all draw from it at the next step).
        devices = [self.device] if self.device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=devices):
                if self.wer_reference and self._wer_floor is None:
                    torch.manual_seed(seed)
                    ref_wavs = W.synthesize_reference(self.net_g, items,
                                                      self.device,
                                                      sid_map=st["sid_map"])
                    hyps = st["asr"].transcribe(W.to_16k(ref_wavs, sample_rate))
                    self._wer_floor = W.wer_summary(list(zip(texts, hyps)))
                if self.wer_reference and self._wer_floor is not None:
                    results["reference"] = self._wer_floor

                torch.manual_seed(seed)
                wavs = W.synthesize_generated(
                    self.net, self.net_g, items, phone_pad, self.device,
                    batch_size=batch_size, max_frames=max_frames,
                    temperature=temperature, top_k=top_k,
                    stop_on_eos=not ignore_eos, hint_arg_fn=st["hint_fn"],
                    sid_map=st["sid_map"])
                hyps = st["asr"].transcribe(W.to_16k(wavs, sample_rate))
                results["generated"] = W.wer_summary(list(zip(texts, hyps)))

                if self.wer_fim and self.net.fim:
                    torch.manual_seed(seed)
                    fim_wavs, _ = W.synthesize_fim(
                        self.net, self.net_g, items, phone_pad, self.device,
                        temperature=temperature, top_k=top_k,
                        hint_arg_fn=st["hint_fn"], sid_map=st["sid_map"])
                    keep = [i for i, w in enumerate(fim_wavs) if w is not None]
                    if keep:
                        hyps = st["asr"].transcribe(W.to_16k(
                            [fim_wavs[i] for i in keep], sample_rate))
                        results["fim"] = W.wer_summary(list(zip(
                            [texts[i] for i in keep], hyps)))
        except Exception as e:  # ASR OOM etc. must not kill a long run
            print(f"WER: probe failed ({e}); disabling it for this run.")
            self._wer_failed = True
            return

        for arm, s in results.items():
            self._add_scalar(f"wer/{arm}", torch.tensor(s["wer"]))
            self._add_scalar(f"wer/{arm}_sub", torch.tensor(float(s["sub"])))
            self._add_scalar(f"wer/{arm}_del", torch.tensor(float(s["del"])))
            self._add_scalar(f"wer/{arm}_ins", torch.tensor(float(s["ins"])))
        floor = results.get("reference")
        parts = []
        for arm in ("generated", "fim", "reference"):
            if arm in results:
                s = results[arm]
                text = f"{arm} {s['wer'] * 100:.2f}%"
                if floor is not None and arm != "reference":
                    text += f" (gap {(s['wer'] - floor['wer']) * 100:+.2f}pp)"
                parts.append(text)
        print(f"wer @ step {self.global_step}: " + " | ".join(parts))

    def _log_speaker_probe(self, j, batch, sids, gen_frames, temperature,
                           top_k, hint_arg="null", tag: str = ""):
        """Generate the same phonemes for several speakers.

        Logs (a) how much of the generated code stream changes when only
        ``sid`` changes and (b) the decoded f0 contours / audio, which makes
        speaker conditioning (and speaker bleed) directly visible.  ``hint_arg``
        / ``tag`` are the hint state, as in ``_log_code_maps``.
        """
        n_spec = self.net.spec_codebook_num
        b = len(sids)
        phone_ids = batch["phone_ids"].repeat(b, 1)
        phone_lengths = batch["phone_lengths"].repeat(b)
        sid_t = torch.tensor(sids, dtype=torch.long, device=self.device)

        codes, lens = self.net.generate(phone_ids, phone_lengths, sid_t,
                                        max_frames=gen_frames,
                                        temperature=temperature, top_k=top_k,
                                        return_lengths=True,
                                        hints=hint_arg,
                                        hint_guidance=self.sample_hint_guidance)

        if self.net.eos_token:
            # Trim to the shortest stream: past its own end each row would be
            # the repeat-filled last frame (see CodeTransformer.generate), which
            # would show up as a spurious *difference* between speakers.
            codes = codes[:, :, : int(lens.min())]
        diff = (codes[1:] != codes[:1]).float()  # (B-1, K, T)
        if self.net.eos_token:
            # Only meaningful with EOS: otherwise every stream is `gen_frames`.
            self._add_scalar(f"probe{tag}/gen_frames", lens.max())
        self._add_scalar(f"probe{tag}/spec_code_change_frac",
                         diff[:, :n_spec].mean())
        self._add_scalar(f"probe{tag}/pitch_code_change_frac",
                         diff[:, n_spec:].mean())


        if self.net_g is None:
            return
        try:
            audio, f0 = self.net_g.infer_from_codes(
                codes[:, :n_spec].permute(1, 0, 2),
                codes[:, n_spec:].permute(1, 0, 2),
                self._gen_sid(sid_t), return_pitch=True)
        except Exception as e:
            print(f"Speaker probe synthesis failed: {e}")
            return

        sr = int(self.config.data.sampling_rate)
        hop = int(self.config.data.hop_length)
        for i, s in enumerate(sids):
            self._add_audio(f"probe{tag}/audio_sid{s}_{j}",
                            audio[i].squeeze(0).float().cpu().numpy(), sr)
        desc = self._hint_desc(tag)
        title = f"same phonemes, different speaker @ step {self.global_step}"
        if desc:
            title = (f"same phonemes, different speaker, {desc} @ step "
                     f"{self.global_step}")
        self._add_image(
            f"probe{tag}/pitch_{j}",
            plot_pitch_curves_figure(
                [(f"sid {s}", f0[i]) for i, s in enumerate(sids)],
                hop, sr, title))

        # Mean |delta f0| vs the item's own speaker, over frames voiced in
        # both contours: near-zero => the model is ignoring `sid`.
        deltas = []
        for i in range(1, len(sids)):
            both = (f0[0] > 0) & (f0[i] > 0)
            if bool(both.any()):
                deltas.append((f0[i][both] - f0[0][both]).abs().mean())
        if deltas:
            self._add_scalar(f"probe{tag}/f0_delta_hz",
                             torch.stack(deltas).mean())

    # ------------------------------------------------------------------ #
    # Out-of-distribution text probe
    # ------------------------------------------------------------------ #
    def _ood_texts(self):
        """Texts used by the OOD probe (``train.ood_texts`` or the default)."""
        texts = self.config.train.get("ood_texts", None)
        if texts is None:
            return list(DEFAULT_OOD_TEXTS)
        return [str(t) for t in texts if str(t).strip()]

    def _build_ood_phone_ids(self, texts):
        """Phonemize the OOD texts once and cache the ids.

        Uses ``FeatureExtractor.phonemize_text`` -- the same path as
        ``2_preprocess.py --phones_from transcript`` -- with
        ``load_qwen=False``, so this never pulls the ASR / forced-aligner
        weights into the training process. A failure (or an empty
        phonemization) is cached as an empty list: the probe is optional and
        must not be retried every epoch.

        Returns:
            ``[(text, ids), ...]``; empty when nothing could be phonemized.
        """
        if self._ood_phone_ids is not None:
            return self._ood_phone_ids
        try:
            from features import FeatureExtractor

            # CPU + no Qwen: phonemize_text needs only g2p_en and the g2pen
            # tokenizer, which keeps the GPU free for training.
            extractor = FeatureExtractor(self.config, device="cpu",
                                         load_qwen=False)
            items = []
            for text in texts:
                ids = [int(i) for i in extractor.phonemize_text(text)]
                if ids:
                    items.append((text, ids))
                else:
                    print(f"OOD text produced no phonemes, skipped: {text!r}")
            del extractor
        except Exception as e:  # optional probe: never break training
            print(f"Could not phonemize OOD text: {e}")
            items = []
        self._ood_phone_ids = items
        return items

    def _log_generated_code_maps(self, prefix, pred):
        """Generated-only codebook heatmaps (one figure per spec/pitch group)."""
        n_spec = self.net.spec_codebook_num
        groups = (
            ("spec", pred[:n_spec], self.net.vocab_sizes[0]),
            ("pitch", pred[n_spec:], self.net.vocab_sizes[-1]),
        )
        for name, p, pad in groups:
            if p.shape[0] == 0:
                continue
            labels = [f"book {i}" for i in range(p.shape[0])]
            self._add_image(
                f"{prefix}/codes_{name}",
                plot_generated_codes_figure(
                    p, pad, f"OOD {name} codes @ step {self.global_step}",
                    labels))

    @torch.no_grad()
    def _log_ood_generation(self, sid, temperature, top_k):
        """Autoregressively generate held-out text, longer than any val item.

        The val samples above are capped at the item's own ``code_len``, so
        they never test what the decoder does once it runs past the length it
        saw.  This probe phonemizes text that is not in the training data and
        decodes ``train.ood_frames`` frames (default 4x ``sample_frames``),
        which forces it to keep going well beyond that point -- where
        autoregressive failure modes (repetition, spectral/pitch drift) appear.

        There is no ground truth for unseen text, so only the generated side is
        logged: ``ood/text_*`` (the transcript), ``ood/codes_{spec,pitch}``
        heatmaps and -- with a stage-1 Generator -- ``ood/audio_*``,
        ``ood/pitch_*`` and ``ood/mel_*``.

        Args:
            sid: (1,) tensor with the speaker id (usually the first sampled
                val item's, or ``train.ood_sid`` when that is set).
            temperature, top_k: decoding knobs, shared with the val samples.
        """
        limit = int(self.config.train.get("ood_samples", 1) or 0)
        if limit <= 0:
            return
        items = self._build_ood_phone_ids(self._ood_texts())
        if not items:
            return

        # `generate` builds an (max_frames + K - 1)-long sequence, so the
        # positional encoding is the hard upper bound.
        max_positions = int(self.config.transformer.get("max_positions", 4096))
        frames = int(self.config.train.get("ood_frames", 0) or 0)
        if frames <= 0:
            frames = 4 * int(self.config.train.get("sample_frames", 128))
        frames = max(1, min(frames, max_positions - self.net.codebook_num + 1))

        sid_cfg = self.config.train.get("ood_sid", None)
        if sid_cfg is not None:
            sid = torch.tensor([int(sid_cfg)], dtype=torch.long,
                               device=self.device)

        # Held-out text has no `.cond` (there is no target audio to derive one
        # from), so `cond` cannot apply here: fall back to the trained null
        # state, or `omit` when that is what the run asked for.
        hint_arg, hint_tag = self._sample_hint_arg(None)
        print(f"OOD generation (hints: {self._hint_desc(hint_tag) or 'none'})")

        for i, (text, ids) in enumerate(items[:limit]):
            phone_ids = torch.tensor(ids, dtype=torch.long,
                                     device=self.device).unsqueeze(0)
            phone_lengths = torch.tensor([len(ids)], dtype=torch.long,
                                         device=self.device)
            codes, lens = self.net.generate(
                phone_ids, phone_lengths, sid, max_frames=frames,
                temperature=temperature, top_k=top_k, return_lengths=True,
                hints=hint_arg,
                hint_guidance=self.sample_hint_guidance,
            )  # (1, K, n)

            codes = codes[0]  # (K, n)
            n_gen = int(codes.shape[-1])
            if self.net.eos_token:
                # The whole point of the probe is to run past the trained
                # length; with EOS the model may instead decide to stop, so log
                # the length it chose (`frames` is only the upper bound).
                self._add_scalar(f"ood{hint_tag}/gen_frames_{i}", lens.max())

            if self.logger is not None:
                self.logger.experiment.add_text(
                    f"ood{hint_tag}/text_{i}", text,
                    global_step=self.global_step)
            print(f"OOD sample {i}: {len(ids)} phonemes -> {n_gen} code "
                  f"frames: {text}")
            self._log_generated_code_maps(f"ood{hint_tag}/{i}", codes)

            if self.net_g is None:
                continue
            try:
                audio, f0 = self._decode_codes(codes, sid)
            except Exception as e:  # sampling is optional
                print(f"OOD synthesis failed: {e}")
                continue
            sr = int(self.config.data.sampling_rate)
            hop = int(self.config.data.hop_length)
            self._add_audio(f"ood{hint_tag}/audio_{i}", audio, sr)
            self._add_image(
                f"ood{hint_tag}/pitch_{i}",
                plot_pitch_curves_figure(
                    [("generated", f0)], hop, sr,
                    f"OOD f0 @ step {self.global_step}"))
            self._add_image(
                f"ood{hint_tag}/mel_{i}",
                plot_mel_figure(
                    audio, sr, hop,
                    f"OOD log-mel @ step {self.global_step}"))


    # ------------------------------------------------------------------ #
    # Train / val
    # ------------------------------------------------------------------ #
    def _forward_loss(self, batch, epoch=None, diagnostics=True):
        """Batch -> loss dict, with ``epoch`` selecting the codebook weights.

        Train passes ``self.current_epoch`` so the curriculum applies.  Val
        passes ``None`` (the full configured weights) on purpose: val has to
        stay a fixed yardstick, otherwise ``val_loss`` silently changes
        definition midway through the warmup and early stopping /
        checkpointing compare incomparable numbers.

        ``diagnostics=False`` skips the per-codebook accuracy / usage-entropy
        diagnostics (see ``CodeTransformer.code_loss``): same loss, less GPU
        work.  Train only asks for them on the steps whose values are actually
        logged; val always does, because those are aggregated over the epoch.
        """
        batch = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        return self.net.compute_batch_loss(batch, epoch=epoch,
                                           diagnostics=diagnostics)

    def _counterfactual_batch(self, batch):
        """Same batch, every sample's clip-level level replaced by a *different*
        one (uniform over the remaining levels, wrapping).

        The shift is chosen in ``1..n-1``, so the new level always differs from
        the true one -- a no-op draw would make the margin free.
        """
        out = dict(batch)
        for name in self.cf_channels:
            key = {"intensity": "hint_int", "register": "hint_reg"}[name]
            cur = batch.get(key)
            if cur is None:
                continue
            n = (self.net.n_intensity_levels if key == "hint_int"
                 else self.net.n_register_levels)
            if n < 2:
                continue
            cur = cur.long()
            shift = torch.randint(1, n, cur.shape, device=cur.device)
            out[key] = 1 + torch.remainder(cur - 1 + shift, n)
        return out

    def on_train_epoch_start(self):
        """Log the effective per-codebook loss weights once per epoch.

        Makes ``transformer.codebook_weight_warmup_epochs`` visible in
        TensorBoard: the curves start uniform and reach the configured
        hierarchy at the end of the warmup.  Also resets the per-epoch
        gradient-spike counters.
        """
        self._grad_norm_max = None
        self._grad_clipped_steps = 0.0
        self._grad_steps = 0.0
        if self.logger is None:
            return
        n_spec = self.net.spec_codebook_num
        weights = self.net.loss_weights(self.current_epoch)
        for i, w in enumerate(weights.detach().float().cpu().tolist()):
            group = "spec" if i < n_spec else "pitch"
            self._add_scalar(f"train_loss_weight/{group}{i}",
                             torch.tensor(float(w)))

    def _log_step_metrics(self, prefix, out, on_step):
        """Log the loss dict produced by ``CodeTransformer.code_loss``.

        Train logs per step (so curves track the actual optimisation), val
        logs per epoch (mean over the val set).

        Only the keys that are present are logged: with
        ``diagnostics=False`` (the train steps between two ``log_interval``
        points, see ``training_step``) ``acc``/``code_accs``/
        ``code_usage_entropy`` are absent by design.

        The per-codebook vectors are transferred to the CPU as *one* stacked
        tensor: the old code did ``.detach().float().cpu().tolist()`` per
        vector per codebook, i.e. ~30 GPU->CPU syncs every step, each of which
        stalls the pipeline for a handful of scalars.
        """
        on_epoch = not on_step
        kw = dict(on_step=on_step, on_epoch=on_epoch, logger=True)
        self.log(f"{prefix}_loss", out["loss"], prog_bar=True, **kw)
        for key in ("spec_loss", "pitch_loss", "acc", "spec_acc", "pitch_acc",
                    "fim_middle_loss", "fim_middle_acc",
                    "fim_middle_acc_early", "fim_middle_acc_mid",
                    "fim_middle_acc_late", "fim_seam_acc",
                    "fim_lookahead_loss", "fim_lookahead_acc", "fim_frac"):
            if key in out:
                self.log(f"{prefix}_{key}", out[key], **kw)

        # Per-codebook curves: loss, accuracy and predicted-distribution
        # entropy. Values are averaged over the batch, so accumulate them on
        # epoch as well as logging them per step.
        n_spec = self.net.spec_codebook_num
        present = [(name, tag, out[name])
                   for name, tag in (("code_losses", "loss"),
                                     ("code_accs", "acc"),
                                     ("code_usage_entropy", "usage_entropy"))
                   if name in out]
        if not present:
            return
        table = torch.stack([vec.detach().float() for _, _, vec in present]
                            ).cpu().tolist()  # one sync for all of them
        for (_, tag, _), row in zip(present, table):
            for i, v in enumerate(row):
                group = "spec" if i < n_spec else "pitch"
                self.log(f"{prefix}_{tag}/{group}{i}", float(v), **kw)

    @staticmethod
    def _grad_total_norm(params):
        """L2 norm of all grads as an on-device tensor (``None`` if no grads).

        Mirrors ``torch.nn.utils.clip_grad_norm_``'s reduction (upcast to fp32)
        without clipping, and without forcing a GPU->CPU sync.
        """
        sq_total = None
        for p in params:
            sq = p.grad.detach().float().pow(2).sum()
            sq_total = sq if sq_total is None else sq_total + sq
        return None if sq_total is None else sq_total.sqrt()

    def _clip_and_log_grads(self, optims):
        """Clip gradients (if configured), then log norms and learning rates.

        Called from ``training_step`` immediately before the optimizers step.
        In automatic optimization this lived in ``on_before_optimizer_step``,
        but Lightning never invokes that hook under
        ``automatic_optimization = False`` (it hangs off the precision plugin's
        automatic-mode step closure), so the clip has to be requested
        explicitly.  All values are kept as tensors until ``self.log`` so the
        call costs no GPU->CPU sync, and only tensors are compared, so the
        per-step statistic accumulation stays on-device.

        Every optimizer consumes the same gradient buffers (disjoint parameter
        sets, one backward pass), so clipping once covers all updates.
        ``optims`` is the normalised list from ``_optimizer_list`` (two entries
        normally, one in fine-tune mode).

        Logged: ``train_grad_norm`` (the **pre-clip** norm, i.e. the spike
        detector — unchanged name/meaning from before clipping existed),
        ``train_grad_clip_scale`` (the factor actually applied to the grads,
        ``1.0`` when clipping did not trigger), ``lr`` (AdamW) and
        ``lr_muon``.  Per epoch, ``train_grad_norm_max`` /
        ``train_grad_clip_frac`` catch spikes that fall between
        ``log_interval`` steps.

        Note: clipping cannot repair non-finite gradients. Lightning's
        default (``error_if_nonfinite=False``) is kept: an ``inf`` norm yields
        a clip factor of 0, i.e. a harmless no-op step, while ``nan`` grads
        stay ``nan`` — if ``train_grad_norm`` goes ``nan`` the run is already
        poisoned and needs a restart from a checkpoint.
        """
        clip_val = self.grad_clip_val
        every = max(int(self.config.train.get("log_interval", 50) or 50), 1)
        log_now = self.global_step % every == 0
        params = [p for p in self.net.parameters() if p.grad is not None]

        norm = None
        scale = None
        if clip_val > 0 and self.grad_clip_algorithm == "norm":
            # Returns the *pre-clip* total norm, so no second reduction is
            # needed to log the spike magnitude.
            norm = torch.nn.utils.clip_grad_norm_(params, clip_val)
            if norm is not None:
                scale = torch.clamp(clip_val / (norm + 1e-6), max=1.0)
        elif clip_val > 0:  # element-wise ("value") clipping
            norm = self._grad_total_norm(params)
            torch.nn.utils.clip_grad_value_(params, clip_val)
            if norm is not None:
                # No single scale is applied; report the ratio against the
                # equivalent total-norm bound so the number stays comparable.
                n_elements = sum(p.numel() for p in params)
                scale = torch.clamp(clip_val * (n_elements ** 0.5)
                                    / (norm + 1e-6), max=1.0)
        elif log_now:
            norm = self._grad_total_norm(params)

        if norm is not None and scale is not None:
            self._grad_norm_max = (norm if self._grad_norm_max is None
                                   else torch.maximum(self._grad_norm_max, norm))
            self._grad_clipped_steps += (scale < 1.0).float()
            self._grad_steps += 1

        if not log_now:
            return
        if norm is not None:
            self.log("train_grad_norm", norm, on_step=True,
                     on_epoch=False, logger=True)
        if scale is not None:
            self.log("train_grad_clip_scale", scale, on_step=True,
                     on_epoch=False, logger=True)
        self.log("lr", float(optims[0].param_groups[0]["lr"]), on_step=True,
                 on_epoch=False, logger=True)
        if len(optims) > 1:
            self.log("lr_muon", float(optims[1].param_groups[0]["lr"]),
                     on_step=True, on_epoch=False, logger=True)

    def training_step(self, batch, batch_idx):
        """Manual optimization: zero_grad -> backward -> clip -> step -> sched.

        Lightning requires manual optimization for multiple optimizers (see
        ``__init__``) and — as a direct consequence — forbids automatic
        gradient accumulation with it, so every batch is exactly one step of
        each optimizer (equivalent to ``accumulate_grad_batches=1``;
        ``__init__`` rejects a larger value).

        The per-codebook diagnostics (accuracy, usage entropy and the ~30
        TensorBoard scalars they feed) are only computed on the steps Lightning
        would log anyway (``train.log_interval``); on the other steps the batch
        only pays for the cross-entropies themselves.  The loss is the same
        either way, so the optimisation is unchanged — this is the same metric
        gating stage 1 uses.  ``train_loss`` (the progress bar) is still logged
        every step.
        """
        self.net.train()
        every = max(int(self.config.train.get("log_interval", 50) or 50), 1)
        diagnostics = self.global_step % every == 0
        out = self._forward_loss(batch, epoch=self.current_epoch,
                                 diagnostics=diagnostics)
        self._log_step_metrics("train", out, on_step=True)

        # One optimizer in fine-tune mode (--finetune), two otherwise; Lightning
        # returns a bare object for the single case, so normalise.
        optims = _optimizer_list(self.optimizers())
        scheds = _optimizer_list(self.lr_schedulers())
        # set_to_none keeps the no-grad buffers unallocated (as Lightning's
        # automatic loop does).
        for optim in optims:
            optim.zero_grad(set_to_none=True)

        loss = out["loss"]
        if getattr(self.net, "spk_kl_weight", 0.0) > 0:
            # VAE-style Gaussian prior on the low-rank speaker codes (see
            # SpeakerEmbedding).  Added to the optimisation loss only; val_loss
            # stays the pure code CE so it remains comparable across arms.
            spk_kl = self.net.spk_kl(batch["sid"])
            loss = loss + self.net.spk_kl_weight * spk_kl
            self.log("train_spk_kl", spk_kl.detach(), on_step=True,
                     on_epoch=True, prog_bar=False)
        # `manual_backward` (not `loss.backward()`) so Lightning keeps control
        # of AMP scaling / DDP grad sync should the precision ever change.
        self.manual_backward(loss)
        if self.cf_margin > 0 and torch.rand(()).item() < self.cf_rate:
            # Counterfactual arm: same targets, shifted clip-level hints.
            #
            # This runs *after* the main backward on purpose.  Building both
            # graphs before either backward doubles the peak activation memory
            # and OOMs at the shipped batch size (fp32, batch 24 x 256 frames,
            # ~13 GB of a 15.5 GB card -- measured).  A second backward
            # accumulates into the same .grad buffers, which is equivalent to
            # summing the two losses, at the cost of one extra backward rather
            # than one extra live graph.
            out_cf = self._forward_loss(self._counterfactual_batch(batch),
                                        epoch=self.current_epoch,
                                        diagnostics=False)
            gap = (out_cf["loss"] - loss.detach())
            # relu(margin - (CE_shift - CE_true)); gradient through the shifted
            # arm only (the true arm already got its own CE above).
            penalty = torch.relu(self.cf_margin - gap)
            self.manual_backward(self.cf_weight * penalty)
            # Logged on every counterfactual step, not only on the gated
            # diagnostics ones: at rate 0.25 and log_interval 50 that would be
            # ~1 sample in 200 steps, too sparse to see the margin engage.
            # `on_epoch=True` as well: Lightning writes only one step scalar per
            # log_interval, so the per-epoch mean is the readable trace of
            # whether the margin is being met.
            self.log("train_cf_gap", gap.detach(), on_step=True,
                     on_epoch=True, prog_bar=False)
            self.log("train_cf_penalty", penalty.detach(), on_step=True,
                     on_epoch=True, prog_bar=False)
        self._clip_and_log_grads(optims)
        # Step the *first* optimizer through its LightningOptimizer wrapper:
        # that is what advances Trainer.global_step (and hence max_steps,
        # cosine_anneal_end, every_n_train_steps and the log axis).  Every
        # wrapper .step() counts one global step, so the rest are stepped
        # through the wrapped torch optimizer instead — stepping both wrappers
        # would burn two global steps per batch and silently halve the
        # configured budget.  All updates consume the same (already clipped)
        # gradients; the wrapper only adds step accounting.
        for index, optim in enumerate(optims):
            if index == 0:
                optim.step()
            else:
                optim.optimizer.step()
        for sched in scheds:
            sched.step()
        # One-time warning if this step crossed train.cosine_anneal_end, after
        # which CosineAnnealingLR starts *raising* the LR again (see
        # `_check_anneal_end`); gated so the hot path costs a single int compare.
        if not self._anneal_warned and self.global_step >= self.anneal_end > 0:
            self._check_anneal_end()
        return loss

    def on_validation_epoch_start(self):
        self._reset_text_delta()
        self._reset_fim_probe()
        self._reset_hint_probe()
        self._reset_hint_channel_probe()

    @torch.no_grad()
    def _fim_probe(self, batch):
        """Teacher-forced FIM diagnostics on one *validation* batch.

        Validation items are plain AR (``dataset_codes`` only draws spans for
        training items), so the prefix/suffix split is synthesised here: with
        ``T = min(code_lengths)``, ``p = T // 3``, ``m = T // 3`` and
        ``s = T - p - m`` every item in the batch gets the *same* span, which is
        what makes the control arms comparable -- targets, masks and span stay
        fixed, only the probed input changes:

        ``val_fim_text_delta``
            middle CE with the batch's phonemes rotated (``> 0`` = text is used)
        ``val_fim_suffix_delta``
            middle CE with the suffix frames taken from the next sample; the
            suffix is the only thing that says what happens *after* the hole, so
            this should be clearly positive
        ``val_fim_region_delta`` / ``val_fim_length_delta``
            middle CE with the region tags / span-length buckets zeroed, i.e. the
            "is this side channel load-bearing?" controls

        ``val_fim_middle_acc`` (+ ``_early``/``_mid``/``_late``) is the base run's
        teacher-forced top-1 by third of the span.  Free-running code match
        (``sample/fim_middle_code_match_*``) is at chance even for a healthy
        model, so this teacher-forced profile is the number that actually says
        whether the model paces content across the span or only wraps up at its
        end.

        Cost: five teacher-forced forwards on ``fim_probe_batches`` validation
        batches per epoch, no sampling -- negligible next to validation itself.
        """
        if not self.net.fim or self.fim_probe_batches <= 0:
            return
        codes = torch.cat([batch["spec_codes"], batch["pitch_codes"]],
                          dim=1).long()
        lengths = batch["code_lengths"]
        T = int(lengths.min())
        p = max(1, T // 3)
        m = max(1, T // 3)
        if p + m >= T:               # keep at least one suffix frame
            m = max(1, T - p - 1)
        if T < 4 or m < 1:
            return
        B = codes.shape[0]
        dev = codes.device
        fim = torch.ones(B, dtype=torch.bool, device=dev)
        pfx = torch.full((B,), p, dtype=torch.long, device=dev)
        mid = torch.full((B,), m, dtype=torch.long, device=dev)
        rotate = torch.roll(torch.arange(B, device=dev), 1)

        # The encoder output is the same in four of the five arms, so encode
        # once per distinct text (twice in total) instead of five times.  The
        # hints must be the same state training used (the val batch carries
        # them), otherwise the probe measures a different function than the
        # loss it is compared against.
        hint_arg = batch if "hint_int" in batch else "null"
        mem, mmask = self.net.encode(batch["phone_ids"], batch["phone_lengths"],
                                     batch["sid"], hints=hint_arg)
        mem_rot, mmask_rot = self.net.encode(batch["phone_ids"][rotate],
                                             batch["phone_lengths"][rotate],
                                             batch["sid"], hints=hint_arg)


        def run(memory, mask, codes_use=None, **override):
            return self.net.code_loss(
                memory, mask, codes if codes_use is None else codes_use,
                code_lengths=lengths, sid=batch["sid"],
                complete=batch["complete"], diagnostics=True,
                hints=hint_arg,
                fim_mask=fim, fim_prefix_len=pfx, fim_mid_len=mid, **override)


        base = run(mem, mmask)
        if "fim_middle_loss" not in base:
            return
        ce_base = float(base["fim_middle_loss"])
        wrong_text = run(mem_rot, mmask_rot)
        codes_swap = codes.clone()
        codes_swap[:, :, p + m:] = codes[rotate][:, :, p + m:]
        wrong_suffix = run(mem, mmask, codes_use=codes_swap)
        no_region = run(mem, mmask, ablate_region=True)
        # The length arm only means something when the model actually has the
        # table; on a ``length_cond: false`` model it would be a no-op reported
        # as "0.000 nats", i.e. indistinguishable from "the channel is dead".
        no_length = (run(mem, mmask, ablate_length=True)
                     if self.net.decoder.len_emb is not None else None)

        self._fim_base += ce_base
        self._fim_text += float(wrong_text["fim_middle_loss"]) - ce_base
        self._fim_suffix += float(wrong_suffix["fim_middle_loss"]) - ce_base
        self._fim_region += float(no_region["fim_middle_loss"]) - ce_base
        if no_length is not None:
            self._fim_length += float(no_length["fim_middle_loss"]) - ce_base
            self._fim_length_count += 1
        if "fim_middle_acc" in base:
            self._fim_acc += float(base["fim_middle_acc"])
            for i, name in enumerate(("early", "mid", "late")):
                key = f"fim_middle_acc_{name}"
                if key in base:
                    self._fim_acc_bins[i] += float(base[key])
        if "fim_seam_acc" in base:
            self._fim_seam_acc += float(base["fim_seam_acc"])
            self._fim_seam_count += 1
        if "fim_lookahead_loss" in base:
            self._fim_la_loss += float(base["fim_lookahead_loss"])
            self._fim_la_acc += float(base["fim_lookahead_acc"])
            self._fim_la_count += 1
        self._fim_probe_count += 1

    def _log_fim_probe(self):
        """Log the accumulated ``_fim_probe`` values (one line per epoch)."""
        if not self._fim_probe_count:
            return
        n = float(self._fim_probe_count)
        length_delta = (self._fim_length / self._fim_length_count
                        if self._fim_length_count else None)
        vals = {
            "val_fim_middle_loss": self._fim_base / n,
            "val_fim_middle_acc": self._fim_acc / n,
            "val_fim_text_delta": self._fim_text / n,
            "val_fim_suffix_delta": self._fim_suffix / n,
            "val_fim_region_delta": self._fim_region / n,
        }
        if length_delta is not None:
            vals["val_fim_length_delta"] = length_delta
        if self._fim_seam_count:
            vals["val_fim_seam_acc"] = self._fim_seam_acc / self._fim_seam_count
        if self._fim_la_count:
            vals["val_fim_lookahead_loss"] = self._fim_la_loss / self._fim_la_count
            vals["val_fim_lookahead_acc"] = self._fim_la_acc / self._fim_la_count
        for i, name in enumerate(("early", "mid", "late")):
            vals[f"val_fim_middle_acc/{name}"] = self._fim_acc_bins[i] / n
        for tag, v in vals.items():
            self._add_scalar(tag, torch.tensor(float(v)))
        seam = (" | seam acc {:.3f} (last {} middle frames)".format(
                    vals["val_fim_seam_acc"], self.net.fim_seam_frames)
                if "val_fim_seam_acc" in vals else "")
        look = (" | lookahead CE {:.3f} (acc {:.3f}, {} frames)".format(
                    vals["val_fim_lookahead_loss"],
                    vals["val_fim_lookahead_acc"], self.net.fim_lookahead)
                if "val_fim_lookahead_loss" in vals else "")
        print("FIM probe: middle CE {:.3f} (acc {:.3f}, span thirds "
              "{:.2f}/{:.2f}/{:.2f}){}{} | deltas (nats): text {:+.3f}, suffix "
              "{:+.3f}, region {:+.3f}, length {}".format(
                  vals["val_fim_middle_loss"], vals["val_fim_middle_acc"],
                  vals["val_fim_middle_acc/early"],
                  vals["val_fim_middle_acc/mid"],
                  vals["val_fim_middle_acc/late"],
                  seam, look,
                  vals["val_fim_text_delta"], vals["val_fim_suffix_delta"],
                  vals["val_fim_region_delta"],
                  "n/a (length_cond off)" if length_delta is None
                  else f"{length_delta:+.3f}"))
        if vals["val_fim_suffix_delta"] < 0.05:
            print("  NOTE: the model barely uses the *suffix* when filling the "
                  "hole (delta < 0.05 nats); a span cannot be paced from what "
                  "comes before it alone, so check this before the length "
                  "conditioning (transformer.length_cond).")

    @torch.no_grad()
    def _accumulate_text_delta(self, batch, out):
        """Add ``CE(wrong phonemes) - CE(real phonemes)`` for one batch.

        The phonemes *and* their lengths are rotated across the batch, so each
        sample is scored against another sample's text while keeping its own
        speaker id and its own code targets.  A delta of ~0 means the decoder
        is not using the encoder memory at all (see `text_delta_batches` in
        ``configs/transformer.yaml``); anything above ~0.05 nats means the text
        is genuinely contributing.
        """
        phones = batch["phone_ids"]
        if phones.shape[0] < 2:
            return
        lengths = batch["phone_lengths"]
        rotate = torch.roll(torch.arange(phones.shape[0], device=phones.device), 1)
        wrong = self.net.compute_batch_loss(
            {**batch, "phone_ids": phones[rotate],
             "phone_lengths": lengths[rotate]},
            epoch=None,  # fixed (final) codebook weights, like val_loss
            # Only the three loss scalars are used below; the per-codebook
            # accuracy/entropy extras would double the cost of this probe.
            diagnostics=False,
        )
        # float() forces a device sync; only the first `text_delta_batches`
        # batches of the epoch pay it.
        self._text_delta_sum += float(wrong["loss"] - out["loss"])
        self._text_delta_spec_sum += float(wrong["spec_loss"] - out["spec_loss"])
        self._text_delta_pitch_sum += float(wrong["pitch_loss"] - out["pitch_loss"])
        self._text_delta_count += 1

    def on_validation_epoch_end(self):
        """Log ``val_text_delta`` (the "is the text conditioning alive?" metric)."""
        self._log_fim_probe()
        self._log_hint_probe()
        self._log_hint_channel_probe()
        if self.text_delta_batches <= 0 or not self._text_delta_count:
            return
        n = float(self._text_delta_count)
        delta = self._text_delta_sum / n
        self._add_scalar("val_text_delta", torch.tensor(delta))
        self._add_scalar("val_text_delta/spec",
                         torch.tensor(self._text_delta_spec_sum / n))
        self._add_scalar("val_text_delta/pitch",
                         torch.tensor(self._text_delta_pitch_sum / n))
        message = (f"val_text_delta = {delta:+.4f} nats "
                   f"(CE with rotated phonemes - CE with the real ones)")
        if delta < 0.05:
            message += ("  <-- text conditioning is not carrying information "
                        "yet; if this persists for several epochs, check "
                        "transformer.memory_norm / "
                        "test_stage2_conditioning.py")
        print(message)

    def validation_step(self, batch, batch_idx):
        self.net.eval()
        out = self._forward_loss(batch)  # fixed (final) codebook weights
        self._log_step_metrics("val", out, on_step=False)
        if batch_idx < self.text_delta_batches:
            self._accumulate_text_delta(batch, out)
        if batch_idx < self.hint_probe_batches:
            self._accumulate_hint_gap(batch, out)
        if batch_idx < self.hint_channel_probe:
            self._accumulate_hint_channel(batch)
        if batch_idx < self.fim_probe_batches:
            self._fim_probe(batch)
        return out["loss"]



def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/transformer.yaml")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--transfer_from", type=str, default=None)
    # LoRA fine-tuning (modeling/lora.py).  The flag switches the mode on; the
    # knobs stay YAML keys so a run is reproducible from its config, because
    # rank/alpha/targets are part of the checkpoint contract.
    parser.add_argument("--finetune", action="store_true",
                        help="train LoRA adapters on a frozen base model")
    parser.add_argument("--finetune_rank", type=int, default=None,
                        help="override finetune.rank (implies --finetune)")
    parser.add_argument("--finetune_lr", type=float, default=None,
                        help="override finetune.lr (implies --finetune)")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    return config, args


def apply_finetune_args(config, args):
    """Fold the ``--finetune*`` flags into ``config.finetune`` in place."""
    rank = getattr(args, "finetune_rank", None)
    lr = getattr(args, "finetune_lr", None)
    if not (getattr(args, "finetune", False) or rank is not None
            or lr is not None):
        return config
    if config.get("finetune", None) is None:
        config["finetune"] = OmegaConf.create({})
    config.finetune.enabled = True
    if rank is not None:
        if int(rank) <= 0:
            raise SystemExit(f"--finetune_rank must be > 0, got {rank}")
        config.finetune.rank = int(rank)
    if lr is not None:
        if float(lr) <= 0:
            raise SystemExit(f"--finetune_lr must be > 0, got {lr}")
        config.finetune.lr = float(lr)
    return config


def check_special_layout(state_dict, net, source, strict: bool = True):
    """Fail loudly when a checkpoint's special-class layout disagrees with ``net``.

    ``transformer.eos_token`` and ``transformer.fim`` each add classes to every
    codebook (EOS; SUF/MID), so switching either changes the embedding/head
    shapes.  A ``--resume_from`` cannot absorb that (Lightning's strict load
    would fail on ~30 tensors and the optimizer state is positional), and a
    ``--transfer_from`` can only *grow* a narrower checkpoint
    (``expand_state_dict_with_special`` has no way to shrink one), so both paths
    call this first.  ``transformer.length_cond`` changes the parameter *set*
    without touching any class count, so it is checked separately (a missing
    ``decoder.len_emb`` table would otherwise be silently re-initialised).

    Args:
        state_dict: checkpoint tensors, keyed with the ``net.`` prefix.
        net: the model being loaded into.
        source: checkpoint path, for the error message.
    """
    key = "net.decoder.embeds.0.weight"
    w = state_dict.get(key)
    want = net.decoder.num_classes[0]
    if w is not None and w.shape[0] != want:
        advice = (f"Either match transformer.eos_token / transformer.fim to this "
                  f"checkpoint, or start a new exp_name and warm-start with "
                  f"--transfer_from {source} (which zero-pads the smaller layout up "
                  f"to the new special classes).  There is no helper to *drop* a "
                  f"special class again.")
        raise SystemExit(
            f"ERROR: {source} has {key} with {w.shape[0]} rows, but this model "
            f"needs {want} (eos_token={net.eos_token}, fim={net.fim}).  {advice}")

    # ``transformer.length_cond`` adds a table (no class-count change), so it
    # has to be detected separately: a shape mismatch would not show up in the
    # check above, and silently dropping/re-initing the table is exactly the
    # kind of half-load that makes an infill run look broken for weeks.
    #
    # Only the "checkpoint has it, model does not" direction is fatal: those are
    # trained weights that would be thrown away.  The other direction (model
    # wants the table, checkpoint predates it) is what ``--transfer_from`` is
    # *for* -- ``load_state_dict_mismatch`` leaves the new table freshly
    # initialised and everything else transfers -- so it is a printed note there
    # and an error only for ``--resume_from`` (whose strict load cannot absorb a
    # missing key anyway).
    has_len = "net.decoder.len_emb.weight" in state_dict
    if has_len and not net.length_cond:
        raise SystemExit(
            f"ERROR: {source} has net.decoder.len_emb.weight (trained span-length "
            f"conditioning), but this model was built with length_cond=False.  "
            f"Set transformer.length_cond: true and n_len_buckets to the trained "
            f"value ({int(state_dict['net.decoder.len_emb.weight'].shape[0])}).")
    if net.length_cond and not has_len:
        message = (f"{source} has no net.decoder.len_emb.weight, but this model "
                   f"was built with length_cond=True: the span-length table "
                   f"starts from scratch and has to be learned.")
        if strict:
            raise SystemExit(
                f"ERROR: {message}  Use --transfer_from {source} instead of "
                f"--resume_from (a strict load cannot absorb the missing key), "
                f"or start a fresh exp_name.")
        print(f"NOTE: {message}")

    # ``transformer.fim_lookahead`` reorders the *training* FIM stream (it
    # repeats the first suffix frames after the middle) but adds no parameters
    # and changes no shape, so a mismatched checkpoint would load completely.
    # The persistent ``fim_layout_marker`` records the trained value; both
    # directions are checked because "checkpoint has it, model does not" would
    # silently evaluate the model on a prompt it was never trained for.
    marker = state_dict.get("net.fim_layout_marker")
    want_look = int(getattr(net, "fim_lookahead", 0))
    if marker is not None:
        got_look = int(marker.reshape(-1)[0].item())
        if got_look != want_look:
            raise SystemExit(
                f"ERROR: {source} was trained with transformer.fim_lookahead="
                f"{got_look} (net.fim_layout_marker), but this model is built "
                f"with fim_lookahead={want_look}.  A checkpoint trained that way "
                f"can only be resumed/evaluated by matching the value; there is "
                f"no way to convert it.")
    elif want_look > 0:
        message = (f"{source} has no net.fim_layout_marker, i.e. it was trained "
                   f"without the FIM suffix lookahead, but this model is built "
                   f"with transformer.fim_lookahead={want_look}.  Every weight "
                   f"transfers, but the seam objective has to be learned.")
        if strict:
            raise SystemExit(
                f"ERROR: {message}  Use --transfer_from {source} instead of "
                f"--resume_from, or start a fresh exp_name.")
        print(f"NOTE: {message}")

    # Low-rank / VAE / FiLM speaker layout (``transformer.spk_rank``,
    # ``spk_norm``, ``spk_init``, ``spk_film_memory``, ``spk_kl_weight``).
    # Rank changes parameter *shapes*; the others change the function while
    # keeping shapes, so the persistent marker is the only trace for those.
    spk_marker = state_dict.get("net.spk_layout_marker")
    want_spk = getattr(net, "spk_layout", (0, "none", "default", False, False))
    if spk_marker is not None:
        got = spk_marker.reshape(-1).tolist()
        got_spk = (int(got[0]), "l2" if got[1] else "none",
                   "zero" if got[2] else "default", bool(got[3]),
                   bool(got[4]))
        if got_spk != want_spk:
            raise SystemExit(
                f"ERROR: {source} was trained with a different speaker layout "
                f"(spk_rank/spk_norm/spk_init/spk_film/spk_kl = {got_spk}) "
                f"than this model ({want_spk}).  Rank changes the embedding "
                f"shapes and the other flags change the function; start a "
                f"fresh exp_name with the checkpoint's config.")
    elif want_spk != (0, "none", "default", False, False):
        message = (f"{source} has no net.spk_layout_marker, i.e. it was trained "
                   f"with the full per-id speaker tables, but this model is "
                   f"built with {want_spk}.  The base weights transfer, the "
                   f"new speaker parameters start fresh.")
        if strict:
            raise SystemExit(
                f"ERROR: {message}  Use --transfer_from (fresh exp_name) rather "
                f"than --resume_from.")
        print("NOTE: " + message)

    # ``transformer.rope`` changes no tensor shape (the rotary embeddings have
    # no parameters), so a mismatched checkpoint would load *completely* and
    # silently score every position differently.  The persistent
    # ``rope_marker`` buffer is the only trace it leaves.
    has_rope = "net.rope_marker" in state_dict
    if has_rope and not net.rope:
        raise SystemExit(
            f"ERROR: {source} was trained with transformer.rope: true (it has "
            f"net.rope_marker), but this model was built with rope: false.  "
            f"Set transformer.rope: true; the sinusoidal model is a different "
            f"function and would otherwise load with no shape error.")
    if net.rope and not has_rope:
        message = (f"{source} has no net.rope_marker, i.e. it was trained with "
                   f"the absolute sinusoidal encoding, but this model is built "
                   f"with transformer.rope: true.  The attention weights "
                   f"transfer, but positional behaviour (and hence the "
                   f"length-extrapolation fix) has to be learned from scratch.")
        if strict:
            raise SystemExit(
                f"ERROR: {message}  Use --transfer_from {source} instead of "
                f"--resume_from, or start a fresh exp_name.")
        print(f"NOTE: {message}")

    # ``finetune.enabled`` (LoRA adapters, modeling/lora.py) adds
    # ``lora_A``/``lora_B``/``lora_qkv_*`` tensors plus a ``lora_marker``
    # buffer.  Dropping them is silent in the worst possible way: every base key
    # and shape still matches, so the run would evaluate the *base* model and
    # report it as the fine-tune.  The check lives in modeling/lora.py so the
    # CPU test can exercise it without importing this script.
    check_lora_layout(state_dict, net, source, strict=strict)

    # ``transformer.hints`` adds its own tables (no class-count change), so it
    # needs the same treatment -- plus one extra check that length_cond does not
    # need: the *widths* come from hints_json, so a different hints.json would
    # silently reinterpret every stored level id.
    has_hints = any(k in state_dict for k in ("net.int_emb.weight",
                                              "net.pitch_emb.weight",
                                              "net.slot_emb.weight",
                                              "net.int_dir",
                                              "net.pitch_val_dir",
                                              "net.word_proj.weight"))
    has_mono = "net.int_dir" in state_dict
    if has_hints and not net.hints:
        raise SystemExit(
            f"ERROR: {source} has net.int_emb.weight (trained conditioning-hint "
            f"embeddings), but this model was built with transformer.hints=false. "
            f"Set transformer.hints: true and point transformer.hints_json at the "
            f"hints.json used for that run.")
    if net.hints and not has_hints:
        message = (f"{source} has no hint tables, but this model was built with "
                   f"hints=true: the hint embeddings start from scratch and have "
                   f"to be learned (everything else transfers).")
        if strict:
            raise SystemExit(
                f"ERROR: {message}  Use --transfer_from {source} instead of "
                f"--resume_from, or start a fresh exp_name.")
        print(f"NOTE: {message}")
    # ``transformer.hint_monotone`` replaces the intensity/register level
    # tables with an ordered direction, i.e. it changes the parameter *set* the
    # same way hints does.  Both directions are fatal for the trained weights:
    #   * monotone checkpoint -> table model: the trained directions are dropped
    #     and the level tables start at zero (silently no level signal at all);
    #   * table checkpoint -> monotone model: the trained tables are dropped and
    #     the directions start at zero.
    # The second is the intended warm-start only if the caller asks for it, so
    # it is a NOTE under --transfer_from and an error under --resume_from.
    if has_mono and not net.hint_monotone:
        raise SystemExit(
            f"ERROR: {source} has {sorted(k for k in state_dict if '_dir' in k)[:3]} "
            f"(a monotone hint encoding), but this model was built with "
            f"transformer.hint_monotone=False.  Set it to true, or use a "
            f"checkpoint with per-level tables.")
    if net.hint_monotone and has_hints and not has_mono:
        message = (f"{source} stores per-level hint tables, but this model uses "
                   f"the monotone (ordered) encoding: int_emb/reg_emb are "
                   f"dropped and int_dir/reg_dir start at zero, so the two "
                   f"clip-level channels have no effect until they are trained.")
        if strict:
            raise SystemExit(
                f"ERROR: {message}  Use --transfer_from {source} instead of "
                f"--resume_from, or start a fresh exp_name.")
        print(f"NOTE: {message}")

    if has_hints and net.hints:
        # ``transformer.hint_pitch_mode`` swaps the whole pitch representation:
        #   table   -> (n_pitch_levels+2, H) per-phone level embedding (v1);
        #   anchors -> per-slot directions over up to A word anchors (v2).
        # Both directions drop trained weights, so neither may happen silently.
        has_pitch_table = "net.pitch_emb.weight" in state_dict
        has_pitch_anchor = "net.pitch_val_dir" in state_dict
        if net.hint_pitch_mode == "anchors" and has_pitch_table:
            message = (f"{source} stores the v1 per-phone pitch level table "
                       f"(net.pitch_emb.weight), but this model uses "
                       f"transformer.hint_pitch_mode=anchors: the table is "
                       f"dropped and the anchor directions start at zero, so "
                       f"the pitch channel has no effect until it is trained.")
            if strict:
                raise SystemExit(
                    f"ERROR: {message}  Use --transfer_from {source} instead of "
                    f"--resume_from, or set hint_pitch_mode: table to evaluate "
                    f"that checkpoint.")
            print(f"NOTE: {message}")
        if net.hint_pitch_mode == "table" and has_pitch_anchor:
            raise SystemExit(
                f"ERROR: {source} has the Hints V2 pitch-anchor parameters "
                f"(net.pitch_val_dir / net.pitch_absent), but this model was "
                f"built with transformer.hint_pitch_mode=table.  Set "
                f"hint_pitch_mode: anchors and hint_pitch_anchors to the "
                f"trained value "
                f"({int(state_dict['net.pitch_val_dir'].shape[0])}).")
        if net.hint_pitch_mode == "anchors" and has_pitch_anchor:
            have = int(state_dict["net.pitch_val_dir"].shape[0])
            if have != net.hint_pitch_anchors:
                raise SystemExit(
                    f"ERROR: {source} stores {have} pitch anchors per word, but "
                    f"transformer.hint_pitch_anchors={net.hint_pitch_anchors}.  "
                    f"Anchor slot i means a different point of the word's "
                    f"contour at a different count, so the ids cannot be "
                    f"reinterpreted.  Set hint_pitch_anchors: {have} and use a "
                    f"corpus built with --pitch-anchors {have}.")
        # V2.1 word states: a fresh projection is fine under --transfer_from
        # (it starts at zero, so the warm start reproduces its source exactly),
        # but dropping a *trained* one is never acceptable, and the width is
        # part of the contract because the encoder's dimensions would otherwise
        # be reinterpreted.
        has_word = "net.word_proj.weight" in state_dict
        if net.hint_word_sem and not has_word and has_hints:
            message = (f"{source} has no per-word semantic state "
                       f"(net.word_proj.weight), but this model uses "
                       f"transformer.hint_word_sem: the projection starts at "
                       f"zero, so the channel has no effect until it is trained.")
            if strict:
                raise SystemExit(
                    f"ERROR: {message}  Use --transfer_from {source} instead of "
                    f"--resume_from, or set hint_word_sem: false.")
            print(f"NOTE: {message}")
        if has_word and not net.hint_word_sem:
            raise SystemExit(
                f"ERROR: {source} has the Hints V2.1 per-word state parameters "
                f"(net.word_proj / net.word_null), but this model was built "
                f"with transformer.hint_word_sem=false.  Set hint_word_sem: "
                f"true and hint_word_dim to the trained width "
                f"({int(state_dict['net.word_proj.weight'].shape[1])}).")
        if has_word and net.hint_word_sem:
            have = int(state_dict["net.word_proj.weight"].shape[1])
            if have != net.hint_word_dim:
                raise SystemExit(
                    f"ERROR: {source} stores {have}-dim per-word states, but "
                    f"transformer.hint_word_dim={net.hint_word_dim}.  Set "
                    f"hint_word_dim: {have}, or use a corpus built with "
                    f"--word-bert-model matching that width.")
        # The sentence-CLS channel was removed in V2.2 (it cost 0.004 nats);
        # a pre-removal checkpoint still carries the tensors.  They are simply
        # unused now, so this is a note -- the warm start is otherwise exact.
        if "net.sem_proj.weight" in state_dict:
            print(f"NOTE: {source} carries the retired sentence-CLS channel "
                  f"(net.sem_proj / net.sem_null); it was removed in V2.2 and "
                  f"the weights are ignored.")
        tables = [("net.int_emb.weight", net.int_emb),
                  ("net.reg_emb.weight", net.reg_emb),
                  ("net.pitch_emb.weight", net.pitch_emb),
                  ("net.slot_emb.weight", net.slot_emb)]
        for key, module in tables:
            if module is None or key not in state_dict:
                continue
            rows = module.num_embeddings
            have = int(state_dict[key].shape[0])
            if have != rows:
                raise SystemExit(
                    f"ERROR: {source} has {key} with {have} rows, but this model "
                    f"needs {rows}. transformer.hints_json describes a different "
                    f"level layout than the one the checkpoint was trained with "
                    f"(the dataset reads the same file, so the level ids would be "
                    f"silently reinterpreted). Point hints_json at the "
                    f"training-time hints.json.")
        # The register table is the one hint channel whose *meaning* can change
        # without changing its width (absolute -> relative, v0.9): row counts
        # cannot catch it, and no checkpoint stores which parameterisation it
        # was trained with.  v0.9 is the *relative* run, so only a pre-v0.9
        # hints checkpoint (v0.8) is affected; there is no way to tell them
        # apart from the file, so this stays a conditional note rather than a
        # statement about the checkpoint.
        if net.relative_register:
            print(f"NOTE: if {source} predates v0.9 it carries hint tables "
                  f"trained with *absolute* register levels, while this model "
                  f"buckets register on the deviation from the speaker median. "
                  f"reg_emb then transfers as a warm start and has to be "
                  f"relearned. (v0.9+ checkpoints are already relative; the "
                  f"file does not record which, so this cannot be checked.)")


def train(config, resume_from=None, transfer_from=None):
    # Stage 2 keeps its own output tree, separate from stage 1's
    # (`checkpoints_stage1/` + `logs_stage1/`), so deleting a failed
    # transformer run can never take the months-long stage-1 GAN checkpoints
    # with it -- and vice versa. Everything below must go through
    # `checkpoints_dir` / `log_dir`; never a literal 'checkpoints' / 'logs'.
    checkpoints_dir = "checkpoints_stage2"
    # Optional override: `train.log_dir` in configs/transformer.yaml defaults
    # to 'logs_stage2'.
    log_dir = config.train.get("log_dir", "logs_stage2")

    net = CodeTransformer(config)
    print(net.describe())

    if os.path.exists(f"{checkpoints_dir}/{config.exp_name}/last.ckpt"):
        print("Detected interrupted training - resuming from last.ckpt")
        resume_from = f"{checkpoints_dir}/{config.exp_name}/last.ckpt"
    if resume_from is not None:
        print(f"Resuming from lightning checkpoint: {resume_from}")
        d = torch.load(resume_from, map_location="cpu", weights_only=False)
        # ``transformer.eos_token`` / ``transformer.fim`` change the codebook
        # embedding/head shapes; exit with instructions instead of letting
        # Lightning fail (or, worse, letting ``--transfer_from`` silently
        # reinitialise 30 tensors).
        check_special_layout(d.get("state_dict", {}), net, resume_from)
        # Lightning restores optimizer states positionally
        # (``zip(self.optimizers, checkpoint["optimizer_states"])``), so a
        # checkpoint written by the old single-AdamW code would silently leave
        # Muon's momentum/step counters at zero.  Say so loudly.
        n_opt_states = len(d.get("optimizer_states") or [])
        n_expected = 1 if getattr(net, "lora_enabled", False) else 2
        if n_opt_states != n_expected:
            print(f"WARNING: {resume_from} contains {n_opt_states} optimizer "
                  f"state(s), but this code trains with {n_expected} "
                  f"({'a single AdamW in fine-tune mode' if n_expected == 1 else 'AdamW + Muon'}). "
                  f"Lightning restores them positionally, so the remaining "
                  f"state cannot be recovered -- prefer --transfer_from if the "
                  f"checkpoint predates the Muon optimizer (or the --finetune "
                  f"switch).")
        if d["global_step"] >= config.train.get("max_steps", 1000000):
            print("Maximum steps reached. Exiting.")
            return
    elif transfer_from is not None:
        print(f"Transferring from checkpoint: {transfer_from}")
        state = torch.load(transfer_from, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        # A narrower checkpoint's codebook embeddings/heads (fewer special
        # classes) are zero-padded instead of letting load_state_dict_mismatch
        # skip (and therefore re-init) all 30 of them.  ``check_special_layout``
        # afterwards catches what the helper cannot fix (a *wider* checkpoint
        # going into a smaller config).
        state = expand_state_dict_with_special(net, state)
        # strict=False: a checkpoint that predates ``transformer.length_cond``
        # is the *expected* warm-start case here -- the new table stays fresh and
        # is reported, the rest transfers.  (The fatal direction, a checkpoint
        # with trained length conditioning going into a model without it, still
        # raises.)
        check_special_layout(state, net, transfer_from, strict=False)
        load_submodule_prefix(net, "net.", state)
    else:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! No checkpoint file found - starting from scratch !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        if getattr(net, "lora_enabled", False):
            print("WARNING: --finetune without --transfer_from/--resume_from: "
                  "only the LoRA adapters are trainable, so the whole model "
                  "would be trained through a ~2 % subspace on a randomly "
                  "initialised base.  Pass --transfer_from <base>.ckpt.")

    training_module = TrainModule(net=net, config=config)

    logger = pl.loggers.TensorBoardLogger(
        log_dir, name=config.exp_name,
        version=config.get("tensorboard_version", 0),
    )
    print(f"Checkpoints: {checkpoints_dir}/{config.exp_name}")
    print(f"TensorBoard: {logger.log_dir}")

    print("Loading data...")
    hint_cfg = config.transformer
    hint_kwargs = dict(
        hints=bool(hint_cfg.get("hints", False)),
        hints_json=hint_cfg.get("hints_json", None),
        hint_dropout=dict(hint_cfg.get("hint_dropout", {}) or {}),
        hint_null_all=float(hint_cfg.get("hint_null_all", 0.0) or 0.0),
        hint_pitch_edges=hint_cfg.get("hint_pitch_edges", None),
        n_slot_buckets=int(hint_cfg.get("n_slot_buckets", 8) or 8),
    )
    train_dataset = CodeDataset(
        config.train.train_filelist,
        is_train=True,
        max_frames=config.train.get("max_frames", None),
        max_phone_length=config.train.get("max_phone_length", None),
        speaker_map=config.train.get("speaker_map", None),
        # Fill-in-the-middle mixing: per-item Bernoulli(train.fim_rate) span
        # reorder, independent of codebook weights.  Requires transformer.fim.
        fim=bool(config.transformer.get("fim", False)),
        fim_rate=float(config.train.get("fim_rate", 0.0) or 0.0),
        fim_min_frames=int(config.transformer.get("fim_min_frames", 8) or 8),
        fim_mid_frac=float(config.transformer.get("fim_mid_frac", 0.5) or 0.5),
        # Crop the phonemes in lockstep with a random training window (see
        # dataset_codes._crop_frames).  On by default; needs the word timings
        # written by `hints/build.py --words`, otherwise it falls back to the
        # historical "full transcript + random audio window" pairing.
        word_crop=bool(config.train.get("word_crop", True)),
        # Optional list of clips whose forced alignment gave a word an
        # impossible duration (see scratch/audit_word_alignment.py).  Those
        # teach the model to swallow the word; dropping them is the cheap fix.
        alignment_blocklist=config.train.get("alignment_blocklist", None),
        stage_direction_blocklist=config.train.get(
            "stage_direction_blocklist", None),
        # Transcript-audit drops (scratch/audit_transcripts.py).  Basenames, so
        # the preprocessed filelist can be filtered without an audio path.
        transcript_blocklist=config.train.get("transcript_blocklist", None),
        # Conditioning hints: the level edges come from the same hints.json the
        # model reads, so dataset and model cannot disagree about a level id.
        **hint_kwargs,
    )
    val_dataset = CodeDataset(
        config.train.val_filelist,
        is_train=False,
        max_frames=config.train.get("max_frames", None),
        max_phone_length=config.train.get("max_phone_length", None),
        speaker_map=config.train.get("speaker_map", None),
        word_crop=bool(config.train.get("word_crop", True)),
        # A blocklist built from the whole corpus may also hit val items; if it
        # was built from train only this is a no-op.
        alignment_blocklist=config.train.get("alignment_blocklist", None),
        stage_direction_blocklist=config.train.get(
            "stage_direction_blocklist", None),
        # Validation never applies hint dropout (`is_train=False`), so the val
        # loss is measured with the hints fully present -- and `val_loss_nohint`
        # gives the same batches with every channel dropped.
        **hint_kwargs,
    )
    n_block = int(getattr(train_dataset, "n_blocklisted", 0))
    n_block_val = int(getattr(val_dataset, "n_blocklisted", 0))
    if n_block or n_block_val:
        print(f"Blocklists: dropped {n_block} train / {n_block_val} val clip(s) "
              f"(alignment + stage-direction + transcript; see train.*_blocklist)")
    if train_dataset.hints:
        reg_mode = ("relative to the speaker median "
                    f"({len(train_dataset.register_speaker_means)} offsets)"
                    if train_dataset.register_speaker_means else "absolute")
        print(f"Hints: on -- {train_dataset.n_intensity_levels} intensity / "
              f"{train_dataset.n_register_levels} register level(s) [{reg_mode}], "
              f"{train_dataset.n_pitch_levels} voiced pitch bins + unvoiced, "
              f"{train_dataset.n_slot_buckets} slot buckets, "
              f"decoder-side={'yes' if config.transformer.get('decoder_hints', True) else 'no'}")
        print(f"       dropout={train_dataset.hint_dropout}, "
              f"null_all={train_dataset.hint_null_all}")

    # Guard against speaker ids that fall outside the transformer's speaker
    # embedding table (and, since ids are canonicalised into the stage-1
    # space, outside the Generator's table too).
    all_sids = train_dataset.speaker_ids() + val_dataset.speaker_ids()
    n_speakers = int(config.transformer.get("n_speakers", 0))
    if all_sids and (max(all_sids) >= n_speakers or min(all_sids) < 0):
        print(f"WARNING: speaker ids span [{min(all_sids)}, {max(all_sids)}] but "
              f"transformer.n_speakers={n_speakers}. Increase n_speakers or fix "
              f"the speaker map to avoid out-of-range / untrained embeddings.")
    collator = CodeCollator(config=config)
    num_workers = config.train.get("num_workers", 0)
    # Length-bucketed batching (see dataset_codes.LengthBucketBatchSampler):
    # groups similarly long clips so a batch is padded to its own width instead
    # of the corpus-wide one.  Measured on this repo's 146 k-item corpus: mean
    # batch width 330 -> 160 frames and 184 k -> 263 k items/h at bs=16, or
    # 387 k items/h at bs=24 (the Muon step is batch-independent, so the larger
    # batch amortises it).  Same items, same crops, same FIM draws -- only the
    # order changes.  Set `train.bucket_batches: false` for the historical
    # uniform shuffle (e.g. to reproduce an older run's batch composition).
    bucket = bool(config.train.get("bucket_batches", True))
    if bucket:
        lengths = train_dataset.compute_lengths()
        pool_batches = int(config.train.get("bucket_pool_batches", 8) or 8)
        train_sampler = LengthBucketBatchSampler(
            lengths, int(config.train.batch_size),
            pool_batches=pool_batches, drop_last=True)
        train_dataloader = DataLoader(
            train_dataset, batch_sampler=train_sampler,
            collate_fn=collator, num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )
        mean_len = sum(lengths) / max(len(lengths), 1)
        print(f"Length-bucketed batching: pool={pool_batches} batches "
              f"({pool_batches * int(config.train.batch_size)} items), "
              f"{len(train_sampler)} batches/epoch, "
              f"mean item width {mean_len:.0f} frames")
    else:
        train_dataloader = DataLoader(
            train_dataset, batch_size=config.train.batch_size, shuffle=True,
            collate_fn=collator, num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.train.batch_size, shuffle=False,
        collate_fn=collator, num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    print("Done")

    val_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"{checkpoints_dir}/{config.exp_name}",
        filename="best-checkpoint",
        save_top_k=config.train.keep_ckpts,
        mode="min",
    )
    default_last_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"{checkpoints_dir}/{config.exp_name}",
        filename="last",
        every_n_epochs=1,
        save_top_k=-1,
    )
    default_interval_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"{checkpoints_dir}/{config.exp_name}",
        filename="model-{step:06d}",
        every_n_train_steps=200000,
        save_top_k=-1,
    )
    callbacks = [val_checkpoint_callback, default_interval_callback, default_last_callback]
    if config.train.get("save_interval") is not None:
        callbacks.append(pl.callbacks.ModelCheckpoint(
            every_n_epochs=config.train.save_interval,
            dirpath=f"{checkpoints_dir}/{config.exp_name}",
            filename="interval-checkpoint-{epoch:04d}",
            save_top_k=-1,
        ))

    trainer = pl.Trainer(
        logger=logger,
        accelerator=config.train.get("accelerator", "gpu"),
        precision=config.train.get("precision", "bf16-mixed"),
        max_steps=config.train.get("max_steps", -1),
        max_epochs=config.train.get("max_epochs", -1),
        callbacks=callbacks,
        check_val_every_n_epoch=config.train.get("val_interval", 1),
        log_every_n_steps=config.train.get("log_interval", 50),
        # Optional cap on the *epoch* length (Lightning `limit_train_batches`).
        # Validation and the per-epoch checkpoints (`last.ckpt`,
        # `best-checkpoint.ckpt`) fire on epoch end, so a short run that stops
        # at `max_steps` inside its first epoch would never validate and would
        # write no checkpoint. Setting this equal to `max_steps` makes the run
        # exactly one (shorter) epoch, which is what the HINTS_V1_PLAN Phase 2
        # 2 k-step wiring arms need to be inspectable. `null`/absent keeps the
        # natural epoch length.
        limit_train_batches=config.train.get("limit_train_batches", None),
        # The bucketed train loader is driven by a custom ``batch_sampler``;
        # Lightning's distributed-sampler rewrite only understands
        # ``sampler``-based loaders, so it must stay out of the way.
        use_distributed_sampler=not bucket,
    )
    trainer.fit(training_module, train_dataloader, val_dataloader,
                ckpt_path=resume_from, weights_only=False)


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)  # needed on Linux
    # TF32 tensor cores for the fp32 matmuls (see train.precision in
    # configs/transformer.yaml). Autocast bf16 corrupts the decoder LayerNorm
    # gradients; TF32 only rounds matmul operands, so the elementwise
    # LayerNorm/softmax maths stays fp32 and the gradients stay correct.
    # NB: use the legacy setter (not torch.backends.cuda.matmul.fp32_precision)
    # -- Lightning 2.6 still calls the legacy getter and mixing the two APIs
    # raises "checking the matmul precision ... mix of legacy and new APIs".
    torch.set_float32_matmul_precision("high")

    config, args = parse_args()
    config = apply_finetune_args(config, args)
    train(config, resume_from=args.resume_from, transfer_from=args.transfer_from)
