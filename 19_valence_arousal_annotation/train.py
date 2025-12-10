from omegaconf import OmegaConf
import pytorch_lightning as pl
from model import ValenceArousalPredictor, ConcordanceCorrelationLoss
from torch import nn
from dataset import dataset
import torch

class TrainModule(pl.LightningModule):
    def __init__(self, model: ValenceArousalPredictor, config: OmegaConf):
        super().__init__()
        self.model = model
        self.criterion = ConcordanceCorrelationLoss()
        self.mse = nn.MSELoss()
        self.config = config

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.config.train.lr)

    def step(self, batch, batch_idx):
        whisper_features = batch['whisper']
        valence = batch['valence'].to(self.dtype).unsqueeze(1)
        arousal = batch['arousal'].to(self.dtype).unsqueeze(1)

        valence_pred, arousal_pred = self.model(whisper_features)
        loss = (self.criterion(valence_pred, valence) + self.criterion(arousal_pred, arousal)) + \
            (self.mse(valence_pred, valence) + self.mse(arousal_pred, arousal))
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
    model = ValenceArousalPredictor(**config.model)
    train_module = TrainModule(model, config)

    train_dataset = dataset(config.data.train_filelist, is_train=True)
    val_dataset = dataset(config.data.val_filelist, is_train=False)
    train_dataloader = train_dataset.loader(
        batch_size=config.train.batch_size, num_workers=config.train.num_workers,
        persistent_workers=True)
    val_dataloader = val_dataset.loader(
        batch_size=config.train.batch_size, num_workers=config.train.num_workers,
        persistent_workers=True)

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