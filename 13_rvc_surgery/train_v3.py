# Stage 2. E2E training with RVC objectives
from svc_helper.svc.rvc.lib.infer_pack.models import MultiPeriodDiscriminatorV2
from modeling.my_rvc import AltSynthesizer
from modeling.spk_classifier import SpeakerClassifier
from modeling.grl import grad_reverse
from dataset import FeatureCollator, FeatureDataset
from rvc_losses import feature_loss, discriminator_loss, generator_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
import torch.nn.functional as F
from commons import slice_segments_general, load_state_dict_mismatch, smooth_random_amplitude_modulation
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
from einops import rearrange
from itertools import chain

import pdb
import sys
sys.excepthook = lambda exc_type, exc_value, exc_traceback: print(exc_type, exc_value, exc_traceback) or pdb.post_mortem(exc_traceback)

class RVCTrainingModule(pl.LightningModule):
    def __init__(self, net_g : AltSynthesizer, net_d: MultiPeriodDiscriminatorV2, config : OmegaConf):
        super().__init__()
        self.net_g = net_g
        self.net_d = net_d
        self.spk_clf = SpeakerClassifier(
            inter_channels=config.model.inter_channels, 
            num_speakers=config.model.spk_embed_dim,
            n_layers=config.model.spk_class_n_layers)
        self.config = config
        self.automatic_optimization = False
        self.last_d_norm = 0

    def on_train_start(self):
        if self.config.train.get('finetune', False):
            print('=== Finetune mode ===')
            # Finetune freezes prior before speaker conditioning 
            for param in self.net_g.parameters():
                param.requires_grad = True
            for param in self.net_d.parameters():
                param.requires_grad = True
            for param in self.spk_clf.parameters():
                param.requires_grad = True

            for param in self.net_g.enc_p.parameters():
                param.requires_grad = False
            for param in self.net_g.enc_p.cond_g.parameters():
                param.requires_grad = True
            for param in self.net_g.enc_p.proj.parameters():
                param.requires_grad = True

        if self.config.train.get('freeze_prior', False):
            print('=== Freezing prior encoder ===')
            for param in self.net_g.enc_p.parameters():
                param.requires_grad = False
        return super().on_train_start()

    def on_train_epoch_start(self):
        # - this was only for retraining titan base model for whisper -

        # if self.current_epoch < self.config.train.stage1_train:
        #     # Retrain prior encoder for speaker invariance
        #     # + train speaker classifier
        #     # Also need to train posterior to match new speaker targets
        #     for param in self.net_g.parameters():
        #         param.requires_grad = False
        #     for param in self.net_d.parameters():
        #         param.requires_grad = False

        #     for param in self.net_g.enc_p.parameters():
        #         param.requires_grad = True
        #     for param in self.net_g.enc_q.parameters():
        #         param.requires_grad = True
        #     for param in self.net_g.emb_g.parameters():
        #         param.requires_grad = True
        #     for param in self.spk_clf.parameters():
        #         param.requires_grad = True
        # else:
        #     # Enable all
        #     for param in self.net_g.parameters():
        #         param.requires_grad = True
        #     for param in self.net_d.parameters():
        #         param.requires_grad = True
        #     for param in self.spk_clf.parameters():
        #         param.requires_grad = True

        pass

    def step(self, batch, batch_idx, is_train=True):
        x = batch
        whisp_feat = x['whisp_feat']
        pitch = x['pitch']
        pitch_fine = x['pitch_fine']
        lens = x['lengths']
        spec = x['spec']
        wave = x['wave']
        sids = x['sids']
        lengths = x['lengths']

        assert spec is not None and wave is not None and len(spec) > 0

        disc_optim, gen_optim = self.optimizers()

        # --- Data augmentation ---
        # Speech feature noise
        whisp_aug = whisp_feat + torch.randn_like(whisp_feat) * self.config.train.whisper_aug_scale
        # Spec power modulation
        spec_aug = smooth_random_amplitude_modulation(spec, 
            min_gain=self.config.train.spec_am_min,
            max_gain=self.config.train.spec_am_max,
            points=self.config.train.spec_am_points
        )
        # Spec noise
        spec_aug = spec_aug + torch.randn_like(spec_aug) * self.config.train.spec_aug_scale

        y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), x = (
            self.net_g(
                whisp_aug, lens, 
                pitch, pitch_fine,
                rearrange(spec_aug, 'b t d -> b d t'), lens,
                sids
            )
        )

        mel_spec = spec_to_mel_torch(
            rearrange(spec, 'b t d -> b d t'), self.config.data.n_fft, self.config.data.num_mels, self.config.data.sampling_rate, self.config.data.fmin, self.config.data.fmax)
        y_mel = slice_segments_general(
            mel_spec, ids_slice, self.config.data.segment_size // self.config.data.hop_length
        )
        y_hat_mel = mel_spectrogram_torch(
            y_hat.float().squeeze(1),
            n_fft=self.config.data.n_fft,
            num_mels=self.config.data.num_mels,
            sampling_rate=self.config.data.sampling_rate,
            hop_size=self.config.data.hop_length,
            win_size=self.config.data.win_size,
            fmin=self.config.data.fmin,
            fmax=self.config.data.fmax,
            center=False
        )
        wave = slice_segments_general(
            wave, ids_slice * self.config.data.hop_length, self.config.data.segment_size
        )

        # not sure if this is correct,
        wave = wave[:, :y_hat.shape[2]] 
        y_mel = y_mel[:, :, :y_hat_mel.shape[2]]

        # but the generated wave seems to only account for the 
        # incomplete length corresponding to mel specs
        # whereas the original wave gets the full subsegment length

        spk_logits = self.spk_clf(grad_reverse(x, self.config.train.lam_grl))
        loss_spk = F.cross_entropy(spk_logits, sids)

        # Noise discriminator inputs
        wave_noise = wave + torch.randn_like(wave) * self.config.train.disc_noise_scale
        y_hat_noise = y_hat + torch.randn_like(y_hat) * self.config.train.disc_noise_scale

        # Discriminator
        y_d_hat_r, y_d_hat_g, _, _ = self.net_d(wave_noise.unsqueeze(1), y_hat_noise.detach())
        loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g, label_alpha=
            self.config.train.label_alpha)

        if is_train and (self.global_step % self.config.train.disc_every == 0):
            disc_optim.zero_grad()
            self.manual_backward(loss_disc)
            d_norm = torch.nn.utils.clip_grad_norm_(self.net_d.parameters(), 10_000.)
            disc_optim.step()
            self.last_d_norm = d_norm
        else:
            d_norm = self.last_d_norm

        # Generator
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(wave_noise.unsqueeze(1), y_hat_noise)
        loss_mel = F.l1_loss(y_mel, y_hat_mel)
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) 
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, _ = generator_loss(y_d_hat_g)
        loss_gen_all = loss_gen + (
            loss_fm + 
            loss_mel*self.config.train.lam_mel + 
            loss_kl*self.config.train.lam_kl + 
            loss_spk*self.config.train.lam_spk)

        if is_train:
            gen_optim.zero_grad()
            self.manual_backward(loss_gen_all)
            g_norm = torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 10_000.)
            gen_optim.step()
        else:
            g_norm = 0

        ret = {
            'loss_gen_all': loss_gen_all,
            'loss_mel': loss_mel,
            'loss_kl': loss_kl,
            'loss_disc': loss_disc,
            'loss_fm': loss_fm, 
            'loss_spk': loss_spk,
            'd_norm': d_norm,
            'g_norm': g_norm
        }

        if not is_train and batch_idx == 0:
            audio = y_hat[0].cpu()
            self.logger.experiment.add_audio(
                tag='gen_audio',
                snd_tensor=audio,
                global_step=self.global_step,
                sample_rate=self.config.data.sampling_rate
            )

        return ret

    def training_step(self, batch, batch_idx):
        out = self.step(batch, batch_idx)
        loss_gen_all = out['loss_gen_all']
        loss_mel = out['loss_mel']
        loss_kl = out['loss_kl']
        loss_disc = out['loss_disc']
        loss_fm = out['loss_fm']
        loss_spk = out['loss_spk']
        d_norm = out['d_norm']
        g_norm = out['g_norm']

        self.log('gen_loss', loss_gen_all, prog_bar=True, logger=True)
        self.log('mel_loss', loss_mel, logger=True)
        self.log('kl_loss', loss_kl, logger=True)
        self.log('disc_loss', loss_disc, prog_bar=True, logger=True)
        self.log('fm_loss', loss_fm, logger=True)
        self.log('spk_loss', loss_spk, logger=True)
        self.log('d_norm', d_norm, logger=True)
        self.log('g_norm', g_norm, logger=True)
        return loss_gen_all + loss_mel + loss_kl + loss_disc + loss_fm

    def validation_step(self, batch, batch_idx):
        out = self.step(batch, batch_idx, is_train=False)
        loss_gen_all = out['loss_gen_all']
        loss_mel = out['loss_mel']
        loss_kl = out['loss_kl']
        loss_disc = out['loss_disc']
        loss_fm = out['loss_fm']
        loss_spk = out['loss_spk']
        d_norm = out['d_norm']
        g_norm = out['g_norm']

        self.log('val_gen_loss', loss_gen_all, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mel_loss', loss_mel, on_epoch=True, logger=True)
        self.log('val_kl_loss', loss_kl, on_epoch=True, logger=True)
        self.log('val_disc_loss', loss_disc, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_fm_loss', loss_fm, on_epoch=True, logger=True)
        self.log('val_spk_loss', loss_spk, on_epoch=True, logger=True)
        self.log('val_loss', loss_gen_all + loss_mel + loss_kl + loss_disc + loss_fm, on_epoch=True, prog_bar=True, logger=True)
        return loss_gen_all + loss_mel + loss_kl + loss_disc + loss_fm

    def configure_optimizers(self):
        disc_optim = torch.optim.AdamW(self.net_d.parameters(), lr=self.config.train.lr, betas=(0.9, 0.999))
        gen_optim = torch.optim.AdamW(
            chain(self.net_g.parameters(), self.spk_clf.parameters()), lr=self.config.train.lr, betas=(0.9, 0.999))
        return [disc_optim, gen_optim], []

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/titan_spk_v3.yaml')
    parser.add_argument('--gen_ckpt', type=str, default=None) # RVC G_ checkpoint
    parser.add_argument('--disc_ckpt', type=str, default=None) # RVC D_ checkpoint
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--transfer_from', type=str, default=None) # transfer learning
    parser.add_argument('--version', type=int, default=None, help='tensorboard log version')

    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    if args.gen_ckpt is not None:
        if args.disc_ckpt is None:
            print('Error: disc_ckpt is required when gen_ckpt is specified')
            exit(1)

        print('Using RVC G_ checkpoint: {}'.format(args.gen_ckpt))
        print('Using RVC D_ checkpoint: {}'.format(args.disc_ckpt))

        gen_state = torch.load(args.gen_ckpt, map_location='cpu')
        net_g = AltSynthesizer(**config.model, is_half=True)
        load_state_dict_mismatch(net_g, gen_state['model'])

        disc_state = torch.load(args.disc_ckpt, map_location='cpu')
        net_d = MultiPeriodDiscriminatorV2(use_spectral_norm=False)
        net_d.load_state_dict(disc_state['model'])
        training_module = RVCTrainingModule(net_g, net_d, config)

    elif args.resume_from is not None:
        print('Resuming from lightning checkpoint: {}'.format(args.resume_from))
        net_g = AltSynthesizer(**config.model, is_half=True)
        net_d = MultiPeriodDiscriminatorV2(use_spectral_norm=False)
        training_module = RVCTrainingModule(net_g, net_d, config)
    elif args.transfer_from is not None:
        print('Transferring from lightning checkpoint: {}'.format(args.transfer_from))
        net_g = AltSynthesizer(**config.model, is_half=True)
        net_d = MultiPeriodDiscriminatorV2(use_spectral_norm=False)
        state = torch.load(args.transfer_from, map_location='cpu')['state_dict']
        submodule_prefix = 'net_g.'
        state_dict = {
            k[len(submodule_prefix):]: v 
            for k, v in state.items() 
            if k.startswith(submodule_prefix)
        }
        # We will experiment with different param shapes here
        load_state_dict_mismatch(net_g, state_dict)
        submodule_prefix = 'net_d.'
        state_dict = {
            k[len(submodule_prefix):]: v 
            for k, v in state.items() 
            if k.startswith(submodule_prefix)
        }
        net_d.load_state_dict(state_dict, strict=False)
        training_module = RVCTrainingModule(net_g, net_d, config)
        submodule_prefix = 'spk_clf.'
        state_dict = {
            k[len(submodule_prefix):]: v 
            for k, v in state.items() 
            if k.startswith(submodule_prefix)
        }
        training_module.spk_clf.load_state_dict(state_dict, strict=False)
    else:
        print('!!! Warning: No checkpoint provided. Training from scratch. !!!')
        net_g = AltSynthesizer(**config.model, is_half=True)
        net_d = MultiPeriodDiscriminatorV2(use_spectral_norm=False)
        training_module = RVCTrainingModule(net_g, net_d, config)

    logger = pl.loggers.TensorBoardLogger(
        config.train.get('log_dir', 'logs'), name=config.exp_name,
        version=args.version
    )

    train_dataset = FeatureDataset(config, is_train=True)
    val_dataset = FeatureDataset(config, is_train=False)
    collator = FeatureCollator()
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        collate_fn=collator,
        shuffle=True,
        num_workers=config.data.num_workers
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        collate_fn=collator,
        num_workers=config.data.num_workers
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'checkpoints/{config.exp_name}',
        filename='best-checkpoint',
        save_top_k=2,
        mode='min',
        save_last=True
    )

    trainer = pl.Trainer(
        max_epochs=config.train.epochs,
        accelerator='auto',
        logger=logger,
        precision='16-mixed',
        callbacks=[checkpoint_callback],
        # detect_anomaly=True
    )
    trainer.fit(training_module, train_dataloader, val_dataloader, ckpt_path=args.resume_from)