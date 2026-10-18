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

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from commons import load_state_dict_mismatch, load_submodule_prefix
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
        self.test()

    def on_train_epoch_end(self):
        self.test()
        return super().on_train_epoch_end()

    # ------------------------------------------------------------------ #
    # Optional sampling / audio logging
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def test(self):
        if self.current_epoch % self.config.train.get("sample_interval", 1):
            return
        print("=== Sampling ===")
        self.net.eval()
        n = self.config.train.get("sample_count", 2)
        frames = self.config.train.get("sample_frames", 128)

        for j, batch in enumerate(itertools.islice(self.test_dataloader, n)):
            batch = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            code_len = int(batch["code_lengths"][0].item())
            gen_frames = max(1, min(frames, code_len))

            codes = self.net.generate(
                batch["phone_ids"], batch["phone_lengths"], batch["sid"],
                max_frames=gen_frames,
            )  # (B, K, gen_frames)

            if self.net_g is None or self.logger is None:
                continue
            try:
                spec_num = self.net.spec_codebook_num
                spec_codes = codes[:, :spec_num].permute(1, 0, 2)  # (Q, B, T)
                pitch_codes = codes[:, spec_num:].permute(1, 0, 2)
                audio = self.net_g.infer_from_codes(
                    spec_codes, pitch_codes, batch["sid"].to(self.device))
                audio = audio[0].squeeze(0).cpu().numpy()
                self.logger.experiment.add_audio(
                    tag=f"sample_{j}",
                    snd_tensor=audio,
                    global_step=self.global_step,
                    sample_rate=self.config.data.sampling_rate,
                )
                self.logger.experiment.flush()
            except Exception as e:
                print(f"Sample synthesis failed: {e}")

    # ------------------------------------------------------------------ #
    # Train / val
    # ------------------------------------------------------------------ #
    def _forward_loss(self, batch):
        batch = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        return self.net.compute_batch_loss(batch)

    def training_step(self, batch, batch_idx):
        self.net.train()
        out = self._forward_loss(batch)
        self.log("train_loss", out["loss"], prog_bar=True, logger=True)
        self.log("train_spec_loss", out["spec_loss"], logger=True)
        self.log("train_pitch_loss", out["pitch_loss"], logger=True)
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        self.net.eval()
        out = self._forward_loss(batch)
        self.log("val_loss", out["loss"], prog_bar=True, logger=True)
        self.log("val_spec_loss", out["spec_loss"], logger=True)
        self.log("val_pitch_loss", out["pitch_loss"], logger=True)
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
