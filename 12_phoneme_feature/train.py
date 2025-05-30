from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
from dataset import FeatureDataset, FeatureCollator
from torch import nn
from torch.utils.data import DataLoader
from utils import load_state_dict_mismatch
from model import PASIFVAE
from einops import rearrange
import torch.nn.functional as F

class TrainingModule(pl.LightningModule):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.model = PASIFVAE(config)
        self.config = config

        self.ctc_loss = nn.CTCLoss(
            blank=config.model.space_id, zero_infinity=True,
            reduction="none"
            )

    def step(self, batch, batch_idx, is_train=True):
        whisper = batch["whisper"]
        phones = batch["phones"]
        spk_id = batch["spk_ids"]
        whisper_mask = batch["whisper_mask"]
        phones_mask = batch["phones_mask"]

        if torch.rand(1).item() < self.config.train.phoneme_hint_prob:
            phoneme_hint = batch["phones"]
            phoneme_hint_mask = batch["phones_mask"]
        else:
            phoneme_hint = None
            phoneme_hint_mask = torch.zeros_like(phones_mask).bool()

        phone_logits, m_p, log_var_p, y, speaker_logits = self.model(
            whisper, whisper_mask, spk_id, phoneme_hint, phoneme_hint_mask)

        phone_logits = rearrange(phone_logits, "B T C -> T B C")
        phone_log_probs = F.log_softmax(phone_logits.float(), dim=-1)

        # assert whisper_mask.sum(-1) >= phones_mask.sum(-1)

        ctc_loss = self.ctc_loss(phone_log_probs.float(), phones, whisper_mask.sum(-1), phones_mask.sum(-1))
        recon_loss = F.l1_loss(y, whisper)
        kl_loss = (-0.5 * torch.sum(1 + log_var_p - m_p.pow(2) - log_var_p.exp())) / whisper.shape[0]

        if sum(ctc_loss > 0).item() > 0:
            ctc_loss = ctc_loss[ctc_loss > 0].mean()
        else:
            print("Warning: all ctc losses are invalid. Setting to zero.")
            ctc_loss = torch.tensor(0.0)

        speaker_loss = nn.CrossEntropyLoss()(speaker_logits, spk_id)

        loss = (ctc_loss * self.config.train.lam_ctc 
            + recon_loss * self.config.train.lam_recon
            + kl_loss * self.config.train.lam_kl
            + speaker_loss * self.config.train.lam_spk)
        if is_train:
            if ctc_loss > 0:
                self.log("ctc_loss", ctc_loss, on_step=True, logger=True, prog_bar=True)
            self.log("recon_loss", recon_loss, on_step=True, logger=True)
            self.log("kl_loss", kl_loss, on_step=True, logger=True)
            self.log("speaker_loss", speaker_loss, on_step=True, logger=True)
        else:
            if ctc_loss > 0:
                self.log("val_ctc_loss", ctc_loss, logger=True)
            self.log("val_recon_loss", recon_loss, logger=True)
            self.log("val_kl_loss", kl_loss, logger=True)
            self.log("val_speaker_loss", speaker_loss, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self.step(batch, batch_idx, is_train=False)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.train.learning_rate)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.config.train.warmup_iters
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warmup_scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--version', type=int)
    parser.add_argument('--resume_from', type=str)
    parser.add_argument('--transfer_from', type=str)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    logger = pl.loggers.TensorBoardLogger(
        config.train.get('log_dir', 'logs'), name=config.train.exp_name,
        version=args.version
    )

    model = TrainingModule(config)

    if args.transfer_from is not None:
        # Assuming transfer from a lightning checkpoint
        load_state_dict_mismatch(model, torch.load(args.transfer_from)['state_dict'])

    train_dataset = FeatureDataset(config, is_train=True)
    val_dataset = FeatureDataset(config, is_train=False)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        collate_fn=FeatureCollator(config),
        shuffle=True,
        persistent_workers=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        num_workers=0,
        collate_fn=FeatureCollator(config)
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'checkpoints/{config.train.exp_name}',
        filename='best-checkpoint',
        save_top_k=2,
        mode='min',
        save_last=True
    )

    trainer = pl.Trainer(
        max_epochs=config.train.epochs,
        accelerator="gpu",
        logger=logger,
        precision='16-mixed',
        #detect_anomaly=True,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=args.resume_from)
