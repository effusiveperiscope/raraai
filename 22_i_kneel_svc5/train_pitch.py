import pytorch_lightning as pl
from modeling.pitch_predictor import PitchPredictorV0
from omegaconf import OmegaConf
import torch

class TrainModule(pl.LightningModule):
    def __init__(self,
        net: PitchPredictorV0,
        config: OmegaConf):
        super().__init__()

        self.net = net
        self.config = config

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.net_g.parameters(),
            lr=self.config.train.lr,
            betas=self.config.train.betas,
            eps=self.config.train.eps
        )
        return [optim], []

    def step(self, batch):
        f0 = batch['f0']

        x_1 = f0
        x_0 = torch.randn_like(x_1)
        x_mask = (f0 != 0)
        t = torch.rand(x_1.shape[0], device=self.device).unsqueeze(-1)

        loss = self.net.compute_loss(x_0=x_0, x_1=x_1, t=t, x_mask=x_mask)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch)

    def validation_step(self, batch, batch_idx):
        return self.step(batch)

