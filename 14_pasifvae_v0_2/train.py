from einops import rearrange
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import FeatureDataset, FeatureCollator
from modeling.model import PASIFVAE
from utils import load_state_dict_mismatch, reset_corrupt_batchnorm_stats

class TrainingModule(pl.LightningModule):
    def __init__(self, config):
        super(TrainingModule, self).__init__()
        self.model = PASIFVAE(config)
        self.config = config
        self.kl_weight = config.train.kl_loss_weight_init
        self.grl_lambda = config.train.grl_lambda_start

    def on_train_epoch_start(self):
        reset_corrupt_batchnorm_stats(self.model)
        return super().on_train_epoch_start()

    def on_train_batch_start(self, batch, batch_idx):
        if self.global_step > self.config.train.kl_annealing_start and self.global_step < self.config.train.kl_annealing_end:
            self.kl_weight = (
                (self.global_step - self.config.train.kl_annealing_start) / 
                (self.config.train.kl_annealing_end - self.config.train.kl_annealing_start)
            ) * (self.config.train.kl_loss_weight_max - self.config.train.kl_loss_weight_init) + self.config.train.kl_loss_weight_init
        elif self.global_step >= self.config.train.kl_annealing_end:
            self.kl_weight = self.config.train.kl_loss_weight_max
        else:
            self.kl_weight = self.config.train.kl_loss_weight_init

        if self.global_step > self.config.train.grl_schedule_start and self.global_step < self.config.train.grl_schedule_end:
            self.grl_lambda = (
                (self.global_step - self.config.train.grl_schedule_start) / 
                (self.config.train.grl_schedule_end - self.config.train.grl_schedule_start)
            ) * (self.config.train.grl_lambda_max - self.config.train.grl_lambda_start) + self.config.train.grl_lambda_start
        elif self.global_step >= self.config.train.grl_schedule_end:
            self.grl_lambda = self.config.train.grl_lambda_max
        else:
            self.grl_lambda = self.config.train.grl_lambda_start

    def step(self, batch, batch_idx, is_train=True):
        whisper = batch['whisper']
        phones = batch['phones']
        pitches = batch['pitches']
        spk_id = batch['spk_ids']
        whisper_mask = batch['whisper_mask']
        phones_mask = batch['phones_mask']

        # Step 1: Add BOS token
        phones = torch.cat(
            [
                torch.full((phones.shape[0], 1), self.config.model.bos_token_id, dtype=torch.long, device=phones.device),
                phones
            ],
            dim=1
        )
        # Step 2: Update phones_mask for BOS token
        phones_mask = torch.cat(
            [
                torch.ones((phones_mask.shape[0], 1), dtype=torch.bool, device=phones_mask.device),  # True for BOS
                phones_mask
            ],
            dim=1
        )

        # Step 3: Add EOS token
        # Create a new tensor for phones with +1 length for EOS
        new_phones = torch.full(
            (phones.shape[0], phones.shape[1] + 1),
            self.config.model.pad_token_id,
            dtype=torch.long,
            device=phones.device
        )
        new_phones_mask = torch.zeros(
            (phones_mask.shape[0], phones_mask.shape[1] + 1),
            dtype=torch.bool,
            device=phones_mask.device
        )

        # Copy original phones and mask
        new_phones[:, :-1] = phones
        new_phones_mask[:, :-1] = phones_mask

        new_phones = new_phones.to(phones.device)
        new_phones_mask = new_phones_mask.to(phones_mask.device)

        # Find the position to insert EOS (first False in mask or end of sequence)
        for i in range(phones.shape[0]):
            # Find the index where valid tokens end (first False or end of sequence)
            valid_length = phones_mask[i].sum().item()  # Number of True values
            eos_pos = valid_length  # Position after the last valid token
            new_phones[i, eos_pos] = self.config.model.eos_token_id
            new_phones_mask[i, eos_pos] = True  # Mark EOS as valid

        y, phone_logits, spk_logits, m_p, log_var_p = self.model(
            whisper,
            whisper_mask,
            new_phones,
            new_phones_mask,
            spk_id,
            pitch=pitches if self.config.model.pitch_cond else None,
            grl_lambda=self.grl_lambda)

        recon_loss = nn.L1Loss()(y, whisper)
        phone_ce_loss = nn.CrossEntropyLoss(
                ignore_index=self.config.model.pad_token_id,
                label_smoothing=self.config.train.label_smoothing)(
            rearrange(phone_logits[:, :-1, :], 'b s c -> b c s'), new_phones[:, 1:])
        if self.global_step < self.config.train.speaker_adversarial_start:
            spk_ce_loss = 0
        else:
            spk_ce_loss = nn.CrossEntropyLoss(reduction='none')(spk_logits, spk_id)
            if spk_ce_loss.isnan().all():
                spk_ce_loss = torch.zeros_like(spk_ce_loss)
                print("Warning - skipping speaker loss due to nan")
            spk_ce_loss = spk_ce_loss[spk_ce_loss.isnan() == False]
            spk_ce_loss = torch.mean(spk_ce_loss)

        kl_loss = 0.5 * torch.mean(-1 - log_var_p + torch.exp(log_var_p) + m_p**2)

        loss = (
            recon_loss * self.config.train.reconstruction_weight +
            phone_ce_loss * self.config.train.phoneme_loss_weight +
            self.kl_weight * kl_loss + 
            self.config.train.speaker_adversarial_weight * spk_ce_loss)

        if is_train:
            self.log('recon_loss', recon_loss, on_step=True, logger=True, prog_bar=True)
            self.log('phone_ce_loss', phone_ce_loss, on_step=True, logger=True, prog_bar=True)
            if self.global_step >= self.config.train.speaker_adversarial_start:
                self.log('spk_ce_loss', spk_ce_loss, on_step=True, logger=True)
            self.log('kl_loss', kl_loss, on_step=True, logger=True)
            self.log('loss', loss, on_step=True, logger=True, prog_bar=True)
        else:
            self.log('val_recon_loss', recon_loss, logger=True)
            self.log('val_phone_ce_loss', phone_ce_loss, logger=True)
            if self.global_step >= self.config.train.speaker_adversarial_start:
                self.log('val_spk_ce_loss', spk_ce_loss, logger=True)
            self.log('val_kl_loss', kl_loss, logger=True)
            self.log('val_loss', loss, logger=True)

        self.log('kl_weight', self.kl_weight, on_step=True, logger=True)
        self.log('grl_lambda', self.grl_lambda, on_step=True, logger=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, logger=True)
        
        return loss

    
    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self.step(batch, batch_idx, is_train=False)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.train.learning_rate)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.config.train.warmup_steps
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
