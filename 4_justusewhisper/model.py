import torch
from torch import nn, optim
import lightning as L
from mambapy.mamba import Mamba, MambaConfig
from lightning.pytorch import loggers as pl_loggers

class Block(nn.Module):
    def __init__(self,
        n_embed,
        dropout):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout)
        )
        mamba_config = MambaConfig(
            d_model = n_embed,
            d_state=16,
            d_conv=4,
            expand_factor=1,
            n_layers=1
        )
        self.mamba = Mamba(mamba_config)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.mamba(self.ln1(x))
        x = x + self.ffn(self.ln2(x)) # (B,N,C)
        return x

class FeatureConvertor(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_emb_size = config['model']['in_emb_size']
        out_emb_size = config['model']['out_emb_size']
        hidden_dim = config['model']['hidden_dim']
        dropout = config['model']['dropout']
        n_layers = config['model']['n_layers']

        self.in_proj = nn.Linear(in_emb_size, hidden_dim)
        self.blocks = nn.Sequential(*[
            Block(hidden_dim, dropout) for _ in range(n_layers)
        ])
        self.out_proj = nn.Linear(hidden_dim, out_emb_size)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.blocks(x)
        x = self.out_proj(x)
        return x

class LitFeatureConvertor(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.feature_convertor = FeatureConvertor(config)
        self.config = config

    def training_step(self, batch, batch_idx):
        hubert, whisper = batch
        hubert_hat = self.feature_convertor(whisper)
        loss = nn.functional.mse_loss(hubert_hat, hubert)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        hubert, whisper = batch
        hubert_hat = self.feature_convertor(whisper)
        loss = nn.functional.mse_loss(hubert_hat, hubert)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), self.config['train']['lr'])
        return optimizer

    def infer(self, feats):
        with torch.no_grad():
            return self.feature_convertor(feats.to(self.device))