import lightning as L
import torch
from model import SimpleAlignmentModel
from torch import nn, optim
import math
import commons
from einops import rearrange

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
        (align_m, align_logs, decoded, logdet) = self.model(
            x = whisper_decoder_feature,
            x_lens = whisper_decoder_lens,
            z = hubert_feature,
            z_lens = hubert_lens)

        z_mask = torch.unsqueeze(
            commons.sequence_mask(hubert_lens), 1)
        align_m, align_logs, decoded = commons.truncate_to_common_length(
            align_m, align_logs, decoded,
            dim_match=2
        )
        loss = torch.sum(align_logs) + 0.5 * torch.sum(torch.exp(
            -2*align_logs) * ((decoded - align_m) ** 2))
        loss = loss - torch.sum(logdet)
        loss = loss / torch.sum(z_mask.long())
        loss = loss + 0.5 * math.log(2*math.pi)

        self.log('loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (hubert_feature, hubert_lens,
         whisper_decoder_feature, whisper_decoder_lens) = batch
        (align_m, align_logs, decoded, logdet) = self.model(
            x = whisper_decoder_feature,
            x_lens = whisper_decoder_lens,
            z = hubert_feature,
            z_lens = hubert_lens)

        z_mask = torch.unsqueeze(
            commons.sequence_mask(hubert_lens), 1)
        
        align_m, align_logs, decoded = commons.truncate_to_common_length(
            align_m, align_logs, decoded,
            dim_match=2
        )

        loss = torch.sum(align_logs) + 0.5 * torch.sum(torch.exp(
            -2*align_logs) * ((decoded - align_m) ** 2))
        loss = loss - torch.sum(logdet)
        loss = loss / torch.sum(z_mask.long())
        loss = loss + 0.5 * math.log(2*math.pi)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        (align_m, align_logs, decoded, logdet) = self.model(
            x = whisper_decoder_feature,
            x_lens = whisper_decoder_lens,
            z = hubert_feature,
            z_lens = hubert_lens, infer=True)

        hubert_feature = rearrange(hubert_feature, 'b n c -> b c n')
        hubert_feature, decoded = commons.truncate_to_common_length(
            hubert_feature, decoded,
            dim_match=2
        )
        mse_loss = nn.functional.mse_loss(decoded, hubert_feature)
        self.log('val_mse', mse_loss, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), self.config['train']['lr'])
        return optimizer