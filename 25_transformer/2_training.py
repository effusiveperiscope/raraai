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
from dataset_codes import CodeCollator, CodeDataset
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
        """
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

    def _apply_muon_settings(self):
        """Re-apply ``train.muon_ns_steps`` after a checkpoint restore.

        ``Optimizer.load_state_dict`` restores ``param_groups`` wholesale, and
        ``ns_steps`` lives in the Muon group's dict — so a ``--resume_from``
        would otherwise silently keep the value the checkpoint was trained
        with, no matter what the config says.  Called from ``on_train_start``,
        which runs after Lightning has restored the optimizers.
        """
        optims = self.optimizers()
        if len(optims) < 2:
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
                             region=None, length=None):
            # Explicit signature (not a ``lambda``) because ``code_loss`` and
            # ``infill`` call the decoder as ``decoder(inputs, memory,
            # memory_mask, sid=sid, region=region, length=length)`` and
            # ``torch.compile`` rejects unknown keyword arguments.
            return impl(codes, memory, memory_mask, sid, region, length, False)

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
        self._sample_and_log()
        self._maybe_compile_decoder()

    def on_train_epoch_end(self):
        self._log_grad_epoch_stats()
        self._sample_and_log()
        return super().on_train_epoch_end()

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
        print("=== Sampling ===")
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

            codes, gen_lens = self.net.generate(
                batch["phone_ids"], batch["phone_lengths"], batch["sid"],
                max_frames=gen_frames, temperature=temperature, top_k=top_k,
                return_lengths=True,
            )  # (B, K, n) with n <= gen_frames when EOS stopped it early
            n_gen = int(codes.shape[-1])
            if self.net.eos_token:
                # How many frames the model decided to produce (max over the
                # batch): with EOS this is the "did it learn to stop?" signal.
                self._add_scalar(f"sample/gen_frames_{j}", gen_lens.max())

            # Ground truth for the same item (spec rows, then pitch rows).
            gt = torch.cat([batch["spec_codes"][0], batch["pitch_codes"][0]],
                           dim=0)[:, :n_gen]

            self._log_code_maps(j, gt, codes[0])

            if self.net_g is not None:
                self._log_audio(j, gt, codes[0], batch["sid"])

            if self.net.fim:
                # Regenerate a fixed-length middle span of the *ground-truth*
                # item and log target-vs-regenerated codes/audio.  This is the
                # direct "does infill work?" view (the free-running samples above
                # are plain AR generation).
                try:
                    self._log_infill(j, batch, temperature, top_k)
                except Exception as e:
                    print(f"FIM infill probe failed: {e}")

            if probe_n > 0 and len(candidates) > 1:
                own = int(batch["sid"][0].item())
                probe = [own] + [s for s in candidates if s != own][:probe_n]
                self._log_speaker_probe(j, batch, probe, gen_frames,
                                        temperature, top_k)

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

    def _log_code_maps(self, j, gt, pred):
        """Target-vs-generated codebook heatmaps, one figure per group."""
        n_spec = self.net.spec_codebook_num
        groups = (
            ("spec", gt[:n_spec], pred[:n_spec], self.net.vocab_sizes[0]),
            ("pitch", gt[n_spec:], pred[n_spec:], self.net.vocab_sizes[-1]),
        )
        for name, g, p, pad in groups:
            if g.shape[0] == 0:
                continue
            labels = [f"book {i}" for i in range(g.shape[0])]
            self._add_image(
                f"codes/{name}_{j}",
                plot_code_comparison_figure(
                    g, p, pad, f"{name} codes @ step {self.global_step}",
                    labels))
            # Free-running agreement on real (non-pad) target frames: unlike
            # teacher-forced accuracy this is the number that decides whether
            # the generated audio is intelligible.
            valid = g != int(pad)
            if bool(valid.any()):
                self._add_scalar(
                    f"sample/{name}_code_match_frac_{j}",
                    (g == p).float()[valid].mean())

    def _decode_codes(self, codes, sid):
        """(K, T) codes -> (waveform, f0 in Hz (T,)) through stage 1."""
        n_spec = self.net.spec_codebook_num
        audio, f0 = self.net_g.infer_from_codes(
            codes[:n_spec].unsqueeze(1), codes[n_spec:].unsqueeze(1), sid,
            return_pitch=True)
        return audio[0].squeeze(0).float().cpu().numpy(), f0[0]

    def _log_audio(self, j, gt, pred, sid):
        try:
            audio_gt, f0_gt = self._decode_codes(gt, sid)
            audio_pred, f0_pred = self._decode_codes(pred, sid)
        except Exception as e:  # sampling is optional
            print(f"Sample synthesis failed: {e}")
            return
        sr = int(self.config.data.sampling_rate)
        hop = int(self.config.data.hop_length)
        self._add_audio(f"audio/target_{j}", audio_gt, sr)
        self._add_audio(f"audio/generated_{j}", audio_pred, sr)
        self._add_image(
            f"pitch/{j}",
            plot_pitch_curves_figure(
                [("target codes", f0_gt), ("generated codes", f0_pred)],
                hop, sr, f"f0 decoded from codes @ step {self.global_step}"))
        self._add_image(
            f"mel/{j}",
            plot_mel_comparison_figure(
                audio_gt, audio_pred, sr, hop,
                f"log-mel @ step {self.global_step}"))

    def _log_infill(self, j, batch, temperature, top_k):
        """Regenerate a fixed-length middle span of a ground-truth item.

        Picks ``prefix = T//3`` and ``middle = T//3`` frames (leaving a suffix),
        blanks the middle, calls :meth:`CodeTransformer.infill` with the known
        prefix/suffix, and logs target-vs-regenerated codes (and audio when a
        stage-1 Generator is loaded).  ``sample/fim_middle_code_match_{j}`` is
        the free-running agreement of the regenerated middle with the original;
        it is a *sample* number (voice sampling variance means it is never 1.0
        for a good model), but a value near chance means the middle is ignored.
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
        )                                                       # (1, K, m)
        regen = torch.cat([prefix, middle, suffix], dim=2)[0]  # (K, T)
        self._log_code_maps(f"fim{j}", codes, regen)
        match = (middle[0, :, :m] == codes[:, p:p + m]).float().mean()
        self._add_scalar(f"sample/fim_middle_code_match_{j}", match)
        self._add_scalar(f"sample/fim_spec_code_match_{j}",
                         (middle[0, :n_spec, :m] == codes[:n_spec, p:p + m])
                         .float().mean())
        if self.net_g is not None:
            self._log_audio(f"fim{j}", codes, regen, batch["sid"][:1])

    def _log_speaker_probe(self, j, batch, sids, gen_frames, temperature,
                           top_k):
        """Generate the same phonemes for several speakers.

        Logs (a) how much of the generated code stream changes when only
        ``sid`` changes and (b) the decoded f0 contours / audio, which makes
        speaker conditioning (and speaker bleed) directly visible.
        """
        n_spec = self.net.spec_codebook_num
        b = len(sids)
        phone_ids = batch["phone_ids"].repeat(b, 1)
        phone_lengths = batch["phone_lengths"].repeat(b)
        sid_t = torch.tensor(sids, dtype=torch.long, device=self.device)

        codes, lens = self.net.generate(phone_ids, phone_lengths, sid_t,
                                        max_frames=gen_frames,
                                        temperature=temperature, top_k=top_k,
                                        return_lengths=True)
        if self.net.eos_token:
            # Trim to the shortest stream: past its own end each row would be
            # the repeat-filled last frame (see CodeTransformer.generate), which
            # would show up as a spurious *difference* between speakers.
            codes = codes[:, :, : int(lens.min())]
        diff = (codes[1:] != codes[:1]).float()  # (B-1, K, T)
        if self.net.eos_token:
            # Only meaningful with EOS: otherwise every stream is `gen_frames`.
            self._add_scalar("probe/gen_frames", lens.max())
        self._add_scalar("probe/spec_code_change_frac", diff[:, :n_spec].mean())
        self._add_scalar("probe/pitch_code_change_frac", diff[:, n_spec:].mean())

        if self.net_g is None:
            return
        try:
            audio, f0 = self.net_g.infer_from_codes(
                codes[:, :n_spec].permute(1, 0, 2),
                codes[:, n_spec:].permute(1, 0, 2),
                sid_t, return_pitch=True)
        except Exception as e:
            print(f"Speaker probe synthesis failed: {e}")
            return

        sr = int(self.config.data.sampling_rate)
        hop = int(self.config.data.hop_length)
        for i, s in enumerate(sids):
            self._add_audio(f"probe/audio_sid{s}_{j}",
                            audio[i].squeeze(0).float().cpu().numpy(), sr)
        self._add_image(
            f"probe/pitch_{j}",
            plot_pitch_curves_figure(
                [(f"sid {s}", f0[i]) for i, s in enumerate(sids)],
                hop, sr,
                f"same phonemes, different speaker @ step {self.global_step}"))

        # Mean |delta f0| vs the item's own speaker, over frames voiced in
        # both contours: near-zero => the model is ignoring `sid`.
        deltas = []
        for i in range(1, len(sids)):
            both = (f0[0] > 0) & (f0[i] > 0)
            if bool(both.any()):
                deltas.append((f0[i][both] - f0[0][both]).abs().mean())
        if deltas:
            self._add_scalar("probe/f0_delta_hz", torch.stack(deltas).mean())

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

        for i, (text, ids) in enumerate(items[:limit]):
            phone_ids = torch.tensor(ids, dtype=torch.long,
                                     device=self.device).unsqueeze(0)
            phone_lengths = torch.tensor([len(ids)], dtype=torch.long,
                                         device=self.device)
            codes, lens = self.net.generate(
                phone_ids, phone_lengths, sid, max_frames=frames,
                temperature=temperature, top_k=top_k, return_lengths=True,
            )  # (1, K, n)
            codes = codes[0]  # (K, n)
            n_gen = int(codes.shape[-1])
            if self.net.eos_token:
                # The whole point of the probe is to run past the trained
                # length; with EOS the model may instead decide to stop, so log
                # the length it chose (`frames` is only the upper bound).
                self._add_scalar(f"ood/gen_frames_{i}", lens.max())

            if self.logger is not None:
                self.logger.experiment.add_text(
                    f"ood/text_{i}", text, global_step=self.global_step)
            print(f"OOD sample {i}: {len(ids)} phonemes -> {n_gen} code "
                  f"frames: {text}")
            self._log_generated_code_maps(f"ood/{i}", codes)

            if self.net_g is None:
                continue
            try:
                audio, f0 = self._decode_codes(codes, sid)
            except Exception as e:  # sampling is optional
                print(f"OOD synthesis failed: {e}")
                continue
            sr = int(self.config.data.sampling_rate)
            hop = int(self.config.data.hop_length)
            self._add_audio(f"ood/audio_{i}", audio, sr)
            self._add_image(
                f"ood/pitch_{i}",
                plot_pitch_curves_figure(
                    [("generated", f0)], hop, sr,
                    f"OOD f0 @ step {self.global_step}"))
            self._add_image(
                f"ood/mel_{i}",
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
                    "fim_middle_acc_late", "fim_frac"):
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

    def _clip_and_log_grads(self, optim_adamw, optim_muon):
        """Clip gradients (if configured), then log norms and learning rates.

        Called from ``training_step`` immediately before the optimizers step.
        In automatic optimization this lived in ``on_before_optimizer_step``,
        but Lightning never invokes that hook under
        ``automatic_optimization = False`` (it hangs off the precision plugin's
        automatic-mode step closure), so the clip has to be requested
        explicitly.  All values are kept as tensors until ``self.log`` so the
        call costs no GPU->CPU sync, and only tensors are compared, so the
        per-step statistic accumulation stays on-device.

        Both optimizers consume the same gradient buffers (disjoint parameter
        sets, one backward pass), so clipping once covers both updates.

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
        self.log("lr", float(optim_adamw.param_groups[0]["lr"]), on_step=True,
                 on_epoch=False, logger=True)
        self.log("lr_muon", float(optim_muon.param_groups[0]["lr"]),
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

        optim_adamw, optim_muon = self.optimizers()
        sched_adamw, sched_muon = self.lr_schedulers()
        # set_to_none keeps the no-grad buffers unallocated (as Lightning's
        # automatic loop does).
        optim_adamw.zero_grad(set_to_none=True)
        optim_muon.zero_grad(set_to_none=True)

        loss = out["loss"]
        # `manual_backward` (not `loss.backward()`) so Lightning keeps control
        # of AMP scaling / DDP grad sync should the precision ever change.
        self.manual_backward(loss)
        self._clip_and_log_grads(optim_adamw, optim_muon)
        # Step AdamW through its LightningOptimizer wrapper: that is what
        # advances Trainer.global_step (and hence max_steps,
        # cosine_anneal_end, every_n_train_steps and the log axis).  Every
        # wrapper .step() counts one global step, so Muon is stepped through
        # the wrapped torch optimizer instead — stepping both wrappers would
        # burn two global steps per batch and silently halve the configured
        # budget.  Both updates consume the same (already clipped) gradients;
        # the wrapper only adds step accounting.
        optim_adamw.step()
        optim_muon.optimizer.step()
        sched_adamw.step()
        sched_muon.step()
        return loss

    def on_validation_epoch_start(self):
        self._reset_text_delta()
        self._reset_fim_probe()

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
        # once per distinct text (twice in total) instead of five times.
        mem, mmask = self.net.encode(batch["phone_ids"], batch["phone_lengths"],
                                     batch["sid"])
        mem_rot, mmask_rot = self.net.encode(batch["phone_ids"][rotate],
                                             batch["phone_lengths"][rotate],
                                             batch["sid"])

        def run(memory, mask, codes_use=None, **override):
            return self.net.code_loss(
                memory, mask, codes if codes_use is None else codes_use,
                code_lengths=lengths, sid=batch["sid"],
                complete=batch["complete"], diagnostics=True,
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
        for i, name in enumerate(("early", "mid", "late")):
            vals[f"val_fim_middle_acc/{name}"] = self._fim_acc_bins[i] / n
        for tag, v in vals.items():
            self._add_scalar(tag, torch.tensor(float(v)))
        print("FIM probe: middle CE {:.3f} (acc {:.3f}, span thirds "
              "{:.2f}/{:.2f}/{:.2f}) | deltas (nats): text {:+.3f}, suffix "
              "{:+.3f}, region {:+.3f}, length {}".format(
                  vals["val_fim_middle_loss"], vals["val_fim_middle_acc"],
                  vals["val_fim_middle_acc/early"],
                  vals["val_fim_middle_acc/mid"],
                  vals["val_fim_middle_acc/late"],
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
        if batch_idx < self.fim_probe_batches:
            self._fim_probe(batch)
        return out["loss"]



def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/transformer.yaml")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--transfer_from", type=str, default=None)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    return config, args


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
        if n_opt_states != 2:
            print(f"WARNING: {resume_from} contains {n_opt_states} optimizer "
                  f"state(s), but this code trains with 2 (AdamW + Muon). "
                  f"Lightning restores them positionally, so the Muon state "
                  f"cannot be recovered -- prefer --transfer_from if the "
                  f"checkpoint predates the Muon optimizer.")
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

    training_module = TrainModule(net=net, config=config)

    logger = pl.loggers.TensorBoardLogger(
        log_dir, name=config.exp_name,
        version=config.get("tensorboard_version", 0),
    )
    print(f"Checkpoints: {checkpoints_dir}/{config.exp_name}")
    print(f"TensorBoard: {logger.log_dir}")

    print("Loading data...")
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
    )
    val_dataset = CodeDataset(
        config.train.val_filelist,
        is_train=False,
        max_frames=config.train.get("max_frames", None),
        max_phone_length=config.train.get("max_phone_length", None),
        speaker_map=config.train.get("speaker_map", None),
    )

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
    train(config, resume_from=args.resume_from, transfer_from=args.transfer_from)
