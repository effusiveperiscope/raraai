from omegaconf import OmegaConf
import pytorch_lightning as pl
from model import IntensityModel
from torch import nn
from dataset import dataset2, WhisperContext, PedalboardContext
from commons import sequence_mask
import torch
import ultimate_xc

import signal
import traceback
import sys

def debug_signal_handler(sig, frame):
    print("\n=== Stack trace ===")
    traceback.print_stack(frame)
    import pdb; pdb.set_trace()
    sys.exit(0)
signal.signal(signal.SIGINT, debug_signal_handler)

class TrainModule(pl.LightningModule):
    def __init__(self, model: IntensityModel, config: OmegaConf):
        super().__init__()
        self.model = model
        self.mse = nn.MSELoss()
        self.config = config
        self.pedalboard_context = PedalboardContext()
        self.whisper_context = WhisperContext()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.config.train.lr)

    def step(self, batch, batch_idx):
        wave = batch['wave']
        wave_lengths = batch['wave_length']

        wave_np = wave.detach().cpu().numpy()
        waves_processed = []
        for i, wave in enumerate(wave_np):
            waves_processed.append(self.pedalboard_context.process_wave(wave[:wave_lengths[i]]))
        whisper_features, feature_len = self.whisper_context.extract_features_batched(
            waves_processed)
        whisper_features = whisper_features.to(self.dtype).to(self.device)
        interp_whisper_features = self.whisper_context.interp2(whisper_features)
        feature_mask = sequence_mask(feature_len).to(torch.long).to(self.device)
        interp_whisper_features = interp_whisper_features[:, :feature_len.max(), :]

        intensity = batch['intensity'].to(self.dtype).unsqueeze(1)
        intensity = (intensity - 1) / 8 # (1-9) -> (0-1)

        intensity = (
            intensity + 
            torch.randn_like(intensity) * self.config.train.get('label_noise', 0)) # label noise

        intensity_pred, attn = self.model(interp_whisper_features, feature_mask)
        intensity_pred = (intensity_pred * attn).sum(dim=1)
        loss = self.mse(intensity_pred, intensity) 
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/base.yaml")
    parser.add_argument("--resume_from", type=str)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    model = IntensityModel(**config.model)
    train_module = TrainModule(model, config)

    train_dataset = dataset2(config.data.train_filelist, is_train=True)
    val_dataset = dataset2(config.data.val_filelist, is_train=False)
    train_dataloader = train_dataset.loader(
        batch_size=config.train.batch_size, num_workers=config.train.num_workers,
        persistent_workers=config.train.num_workers > 0)
    val_dataloader = val_dataset.loader(
        batch_size=config.train.batch_size, num_workers=config.train.num_workers,
        persistent_workers=config.train.num_workers > 0)

    val_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'checkpoints/{config.exp_name}',
        filename='best-checkpoint',
        save_top_k=config.train.keep_ckpts,
        mode='min',
        save_last=True
    )
    callbacks = [val_checkpoint_callback]
    logger = pl.loggers.TensorBoardLogger(
        config.get('log_dir', 'logs'), name=config.exp_name,
        version=0
    )

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=-1,
        accelerator='gpu',
        precision='bf16-mixed',
        callbacks=callbacks)

    trainer.fit(train_module, train_dataloader, val_dataloader, ckpt_path=args.resume_from)

if __name__ == '__main__':
    main()