import torch
from torch.nn import functional as F
from omegaconf import OmegaConf
from svc_helper.svc.rvc.lib.infer_pack.models import (MultiPeriodDiscriminatorV2) 
from modeling.v08.rvc import V08Synthesizer
from teacher.dataset_teacher import FeatureDataset, FeatureCollator
from commons import load_state_dict_mismatch
import pytorch_lightning as pl
from einops import rearrange

class TeacherTrainingModule(pl.LightningModule):
    def __init__(self,
        net_g : V08Synthesizer,
        config : OmegaConf):
        super().__init__()
        self.net_g = net_g
        self.config = config

    def step(self, batch, batch_idx):
        whisp_feat = batch['whisp_feat']
        svc5_feat = batch['svc5_feat']
        pitch_fine = batch['pitch_fine']
        length = batch['length']
        spk = batch['spk'] # Speaker embeddings, [B, 256]

        if self.training:
            # Noise augmentation
            whisp_feat_noised = whisp_feat + torch.randn_like(whisp_feat) * self.config.train.noise_scale
        else:
            whisp_feat_noised = whisp_feat
        whisp_feat_noised = whisp_feat_noised.to(self.device).to(whisp_feat.dtype)

        x_mask, spk_emb_pred, pre_proj_x = \
            self.net_g.prior_only(whisp_feat_noised, length, pitch_fine.to(whisp_feat.dtype))

        x_mask = rearrange(x_mask, 'b 1 t -> b t 1')
        recon_loss = F.l1_loss(
            rearrange(pre_proj_x, 'b c t -> b t c') * x_mask, svc5_feat * x_mask)
        spk_loss = F.cosine_embedding_loss(spk_emb_pred, spk, torch.ones(spk.shape[0]).to(self.device))
        loss = recon_loss * self.config.train.lam_recon + spk_loss * self.config.train.lam_spk

        if self.training:
            self.log('recon_loss', recon_loss.detach().cpu(), prog_bar=True, on_step=True, logger=True)
            self.log('spk_loss', spk_loss.detach().cpu(), prog_bar=True, on_step=True, logger=True)
        else:
            self.log('val_recon_loss', recon_loss.detach().cpu(), prog_bar=True, logger=True)
            self.log('val_spk_loss', spk_loss.detach().cpu(), prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('train_loss', loss.detach().cpu(), prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('val_loss', loss.detach().cpu(), prog_bar=True, logger=True)
        return loss.detach()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.net_g.parameters(), lr=self.config.train.lr, 
            betas=(0.9, 0.999), weight_decay=self.config.train.weight_decay)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/teacher_test.yaml')
    parser.add_argument('--gen_ckpt', type=str, default=None) # RVC G_ checkpoint
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--version', type=int, default=None, help='tensorboard log version')

    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    if args.gen_ckpt is not None:
        print('Using RVC G_ checkpoint: {}'.format(args.gen_ckpt))

        gen_state = torch.load(args.gen_ckpt, map_location='cpu')
        net_g = V08Synthesizer(**config.model, is_half=True)
        load_state_dict_mismatch(net_g, gen_state['model'])

        training_module = TeacherTrainingModule(net_g, config)
        
    logger = pl.loggers.TensorBoardLogger(
        config.train.get('log_dir', 'logs'), name=config.exp_name,
        version=args.version
    )

    train_dataset = FeatureDataset(config, is_train=True)
    val_dataset = FeatureDataset(config, is_train=False)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = config.train.batch_size,
        shuffle = True,
        collate_fn = FeatureCollator(),
        num_workers=0,
        #persistent_workers=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = config.train.batch_size,
        shuffle = False,
        collate_fn = FeatureCollator(),
        num_workers=0,
        #persistent_workers=True
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'checkpoints/teacher/{config.exp_name}',
        filename='best-checkpoint',
        save_top_k=2,
        mode='min',
        save_last=True
    )

    trainer = pl.Trainer(
        logger=logger,
        accelerator='gpu',
        precision='bf16',
        max_epochs=config.train.max_epochs,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(training_module, train_dataloader, val_dataloader, ckpt_path=args.resume_from)