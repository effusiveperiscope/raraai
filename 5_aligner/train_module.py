import lightning as L
import torch
from model import SimpleAlignmentModel
from torch import nn, optim
import math
import commons

class LitModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.model = SimpleAlignmentModel(
            bottleneck_channels=config['model']['bottleneck_size']
        )
        self.config = config

    def training_step(self, batch, batch_idx):
        (hubert_feature, hubert_lens,
         whisper_decoder_feature, whisper_decoder_lens) = batch
        (align_m, align_logs, latent_feats,
            true_decoded, pred_decoded) = self.model(
            x = whisper_decoder_feature,
            x_lens = whisper_decoder_lens,
            z = hubert_feature,
            z_lens = hubert_lens)

        z_mask = torch.unsqueeze(
            commons.sequence_mask(hubert_lens), 1)
        align_loss = (torch.sum(align_logs) + 
            0.5 * torch.sum(torch.exp(-2 * align_logs) * (
                (latent_feats.transpose(1,2) - align_m) ** 2))) / (
                    torch.sum(z_mask.long())
                ) + (0.5*math.log(2*math.pi))
        decoder_loss = nn.functional.mse_loss(true_decoded, hubert_feature)
        mse_loss = nn.functional.mse_loss(pred_decoded, hubert_feature)
        loss = align_loss + mse_loss + decoder_loss

        self.log('align_loss', align_loss, on_epoch=True)
        self.log('decoder_loss', decoder_loss, on_epoch=True)
        self.log('mse_loss', mse_loss, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (hubert_feature, hubert_lens,
         whisper_decoder_feature, whisper_decoder_lens) = batch
        (align_m, align_logs, latent_feats,
            true_decoded, pred_decoded) = self.model(
            x = whisper_decoder_feature,
            x_lens = whisper_decoder_lens,
            z = hubert_feature,
            z_lens = hubert_lens)

        z_mask = torch.unsqueeze(
            commons.sequence_mask(hubert_lens), 1)
        align_loss = (torch.sum(align_logs) + 
            0.5 * torch.sum(torch.exp(-2 * align_logs) * (
                (latent_feats.transpose(1,2) - align_m) ** 2))) / (
                    torch.sum(z_mask.long())
                ) + (0.5*math.log(2*math.pi))
        decoder_loss = nn.functional.mse_loss(true_decoded, hubert_feature)
        mse_loss = nn.functional.mse_loss(pred_decoded, hubert_feature)
        loss = align_loss + mse_loss + decoder_loss

        self.log('val_align_loss', align_loss, on_epoch=True)
        self.log('val_decoder_loss', decoder_loss, on_epoch=True)
        self.log('val_mse_loss', mse_loss, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), self.config['train']['lr'])
        return optimizer