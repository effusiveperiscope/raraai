# Stage 2. E2E training with RVC objectives
from omegaconf import OmegaConf
import pytorch_lightning as pl

class RVCTrainingModule(pl.LightningModule):
    def __init__(self, rvc_model, config : OmegaConf):
        super().__init__()
        self.rvc_model = rvc_model

    def step(self, batch, batch_idx):
        x = batch
        whisp_feat = x['whisp_feat']
        pitch = x['pitch']
        pitch_fine = x['pitch_fine']
        lens = x['lengths']

        y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = (
            self.rvc_model(whisp_feat, lens, pitch, pitch_fine)
        )