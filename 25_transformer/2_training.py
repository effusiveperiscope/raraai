"""Stage-2 training for the discrete-code transformer.

Mirrors the layout of ``1_training.py`` (Lightning module + ``train``
function + checkpoint/resume handling), but trains ``modeling.transformer
.CodeTransformer`` instead of the VITS-style Generator/Discriminator pair.

All model-side computations (delay pattern, shifted targets, masked
cross-entropy, sampling) live in ``modeling/transformer.py``; this file only
owns the optimisation loop, data loading, logging and checkpointing.
"""

import itertools
import os

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
from modeling.transformer import CodeTransformer

import ultimate_xc
import logging
# Something above this is setting DEBUG but I don't know what
logging.getLogger().setLevel(logging.WARNING)


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


class TrainModule(pl.LightningModule):
    def __init__(self, net: CodeTransformer, config: OmegaConf):
        super().__init__()
        self.net = net
        self.config = config
        self.net_g = None

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
                self.net_g = load_generator(ckpt, hp, self.device)
                for p in self.net_g.parameters():
                    p.requires_grad = False
                print(f"Loaded stage-1 Generator for sampling: {ckpt}")
            except Exception as e:  # sampling is optional
                print(f"Could not load Generator ({ckpt}): {e}")
                self.net_g = None
        self.hp = hp

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.config.train.lr,
            betas=tuple(self.config.train.betas),
            eps=self.config.train.eps,
            weight_decay=self.config.train.get("weight_decay", 0.0),
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optim,
            T_max=self.config.train.get("cosine_anneal_end", 1000000),
            eta_min=1e-6,
        )
        return {"optimizer": optim, "lr_scheduler": sched}

    def on_train_start(self):
        self._sample_and_log()

    def on_train_epoch_end(self):
        self._sample_and_log()
        return super().on_train_epoch_end()

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

        for j, batch in enumerate(itertools.islice(self.test_dataloader, n)):
            batch = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            code_len = int(batch["code_lengths"][0].item())
            gen_frames = max(1, min(frames, code_len))

            codes = self.net.generate(
                batch["phone_ids"], batch["phone_lengths"], batch["sid"],
                max_frames=gen_frames, temperature=temperature, top_k=top_k,
            )  # (B, K, gen_frames)

            # Ground truth for the same item (spec rows, then pitch rows).
            gt = torch.cat([batch["spec_codes"][0], batch["pitch_codes"][0]],
                           dim=0)[:, :gen_frames]

            self._log_code_maps(j, gt, codes[0])

            if self.net_g is not None:
                self._log_audio(j, gt, codes[0], batch["sid"])

            if probe_n > 0 and len(candidates) > 1:
                own = int(batch["sid"][0].item())
                probe = [own] + [s for s in candidates if s != own][:probe_n]
                self._log_speaker_probe(j, batch, probe, gen_frames,
                                        temperature, top_k)

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

        codes = self.net.generate(phone_ids, phone_lengths, sid_t,
                                  max_frames=gen_frames,
                                  temperature=temperature, top_k=top_k)
        diff = (codes[1:] != codes[:1]).float()  # (B-1, K, T)
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
    # Train / val
    # ------------------------------------------------------------------ #
    def _forward_loss(self, batch):
        batch = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        return self.net.compute_batch_loss(batch)

    def _log_step_metrics(self, prefix, out, on_step):
        """Log the loss dict produced by ``CodeTransformer.code_loss``.

        Train logs per step (so curves track the actual optimisation), val
        logs per epoch (mean over the val set).
        """
        on_epoch = not on_step
        kw = dict(on_step=on_step, on_epoch=on_epoch, logger=True)
        self.log(f"{prefix}_loss", out["loss"], prog_bar=True, **kw)
        for key in ("spec_loss", "pitch_loss", "acc", "spec_acc", "pitch_acc"):
            if key in out:
                self.log(f"{prefix}_{key}", out[key], **kw)

        # Per-codebook curves: loss, accuracy and predicted-distribution
        # entropy. Values are averaged over the batch, so accumulate them on
        # epoch as well as logging them per step.
        n_spec = self.net.spec_codebook_num
        for name, tag in (("code_losses", "loss"), ("code_accs", "acc"),
                          ("code_usage_entropy", "usage_entropy")):
            vec = out.get(name)
            if vec is None:
                continue
            for i, v in enumerate(vec.detach().float().cpu().tolist()):
                group = "spec" if i < n_spec else "pitch"
                self.log(f"{prefix}_{tag}/{group}{i}", float(v), **kw)

    def on_before_optimizer_step(self, optimizer):
        """Log global grad norm and LR at the configured log interval."""
        every = max(int(self.config.train.get("log_interval", 50) or 50), 1)
        if self.global_step % every != 0:
            return
        total = None
        for p in self.net.parameters():
            if p.grad is None:
                continue
            sq = p.grad.detach().float().pow(2).sum()
            total = sq if total is None else total + sq
        if total is not None:
            self.log("train_grad_norm", total.sqrt(), on_step=True,
                     on_epoch=False, logger=True)
        self.log("lr", float(optimizer.param_groups[0]["lr"]), on_step=True,
                 on_epoch=False, logger=True)

    def training_step(self, batch, batch_idx):
        self.net.train()
        out = self._forward_loss(batch)
        self._log_step_metrics("train", out, on_step=True)
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        self.net.eval()
        out = self._forward_loss(batch)
        self._log_step_metrics("val", out, on_step=False)
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


def train(config, resume_from=None, transfer_from=None):
    net = CodeTransformer(config)
    print(net.describe())

    if os.path.exists(f"checkpoints/{config.exp_name}/last.ckpt"):
        print("Detected interrupted training - resuming from last.ckpt")
        resume_from = f"checkpoints/{config.exp_name}/last.ckpt"
    if resume_from is not None:
        print(f"Resuming from lightning checkpoint: {resume_from}")
        d = torch.load(resume_from, map_location="cpu", weights_only=False)
        if d["global_step"] >= config.train.get("max_steps", 1000000):
            print("Maximum steps reached. Exiting.")
            return
    elif transfer_from is not None:
        print(f"Transferring from checkpoint: {transfer_from}")
        state = torch.load(transfer_from, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        load_submodule_prefix(net, "net.", state)
    else:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! No checkpoint file found - starting from scratch !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    training_module = TrainModule(net=net, config=config)

    logger = pl.loggers.TensorBoardLogger(
        config.train.get("log_dir", "logs"), name=config.exp_name,
        version=config.get("tensorboard_version", 0),
    )

    print("Loading data...")
    train_dataset = CodeDataset(
        config.train.train_filelist,
        is_train=True,
        max_frames=config.train.get("max_frames", None),
        max_phone_length=config.train.get("max_phone_length", None),
        speaker_map=config.train.get("speaker_map", None),
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
        dirpath=f"checkpoints/{config.exp_name}",
        filename="best-checkpoint",
        save_top_k=config.train.keep_ckpts,
        mode="min",
    )
    default_last_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"checkpoints/{config.exp_name}",
        filename="last",
        every_n_epochs=1,
        save_top_k=-1,
    )
    default_interval_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"checkpoints/{config.exp_name}",
        filename="model-{step:06d}",
        every_n_train_steps=200000,
        save_top_k=-1,
    )
    callbacks = [val_checkpoint_callback, default_interval_callback, default_last_callback]
    if config.train.get("save_interval") is not None:
        callbacks.append(pl.callbacks.ModelCheckpoint(
            every_n_epochs=config.train.save_interval,
            dirpath=f"checkpoints/{config.exp_name}",
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

    config, args = parse_args()
    train(config, resume_from=args.resume_from, transfer_from=args.transfer_from)
