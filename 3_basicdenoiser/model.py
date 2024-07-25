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

class FeatureDenoiser(nn.Module):
    def __init__(self,
        config):
        super().__init__()

        speech_emb_size = config['model']['speech_emb_size']
        hidden_dim = config['model']['hidden_dim']
        dropout = config['model']['dropout']
        n_layers = config['model']['n_layers']
        
        self.in_proj = nn.Linear(
            speech_emb_size, hidden_dim)
        self.blocks = nn.Sequential(*[
            Block(hidden_dim, dropout) for _ in range(n_layers)
        ])
        self.out_proj = nn.Linear(hidden_dim, speech_emb_size)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.blocks(x)
        x = self.out_proj(x)
        return x

class LitFeatureDenoiser(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.feature_denoiser = FeatureDenoiser(config)
    
    def training_step(self, batch, batch_idx):
        clean, noisy = batch
        clean_hat = self.feature_denoiser(noisy)
        loss = nn.functional.mse_loss(clean_hat, clean)
        #with torch.no_grad():
        #    print('clean ',clean.mean(), clean.std())
        #    print('clean hat ',clean_hat.mean(), clean_hat.std())
        #    print('mse loss', loss)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        clean, noisy = batch
        clean_hat = self.feature_denoiser(noisy)
        loss = nn.functional.mse_loss(clean_hat, clean)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        #clean_hat_0 = clean_hat[0]
        #tensorboard = self.logger.experiment
        #tensorboard.add_audio(f'test_audio')
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=6e-4)
        return optimizer