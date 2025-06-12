from svc_helper.svc.rvc.lib.infer_pack.models import MultiPeriodDiscriminatorV2
from features import MyFeatures
from modeling.v05.rvc import SynthesizerV05
from dataset_paired import FeatureDatasetPaired, paired_feature_collator
from dataset import FeatureDataset, FeatureCollator
from rvc_losses import feature_loss, discriminator_loss, generator_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
import torch.nn.functional as F
from commons import (
    slice_segments_general, load_state_dict_mismatch,
    smooth_random_amplitude_modulation, random_subsample_segments)
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
from einops import rearrange

class V05TrainingModule(pl.LightningModule):
    def __init__(self, 
        net_g : SynthesizerV05, 
        net_d: MultiPeriodDiscriminatorV2, config : OmegaConf):
        super().__init__()
        self.net_g = net_g
        self.net_d = net_d
        self.config = config
        self.automatic_optimization = False
        self.stage1 = False

    def on_train_start(self):
        self.test()

    def on_train_step_start(self):
        if self.global_step < self.config.train.get('stage1_train_step', 0):
            self.stage1 = True
            # Only train decoder
            # for param in self.net_g.parameters():
                # param.requires_grad = False
            # for param in self.net_d.parameters():
                # param.requires_grad = False
            # for param in self.net_g.dec.parameters():
                # param.requires_grad = True

            # Train speaker content classifier, latent discriminator, and speaker embeddings
            for param in self.net_g.parameters():
                if 'emb_g' not in param.name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            for param in self.net_d.parameters():
                param.requires_grad = False
            for param in self.net_g.enc_p.speaker_classifier.parameters():
                param.requires_grad = True
            for param in self.net_g.enc_p.speaker_discriminator.parameters():
                param.requires_grad = True
        else:
            self.stage1 = False
            for param in self.net_g.parameters():
                param.requires_grad = True
            for param in self.net_d.parameters():
                param.requires_grad = True

    def step_lerp(self, min=0, max=1, start=0, end=10000):
        if self.global_step < start:
            return min
        elif self.global_step < end:
            return (self.global_step - start) / (end - start) * (max - min) + min
        else:
            return max

    def test(self):
        print('=== Testing ===')
        self.net_g.eval()
        self.net_d.eval()

        if not hasattr(self, 'test_dataset'):
            self.test_dataset = FeatureDataset(self.config, is_train=False, 
                override_filelist=self.config.train.test_filelist)
            self.test_dataloader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=self.config.train.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=FeatureCollator(),
            )

        for batch in self.test_dataloader:
            if self.config.train.get('octave_transpose_test', True):
                batch['pitch_fine'] = batch['pitch_fine'] * 2 # Octave transpose
                batch['pitch'] = MyFeatures.f0_to_coarse(batch['pitch_fine']).squeeze(0) # Recalculate coarse
            with torch.no_grad():
                o, x_mask, z_stats = self.net_g.infer(
                    batch['whisp_feat'].to(self.device).to(self.dtype), 
                    batch['lengths'].to(self.device), 
                    batch['pitch'].to(self.device), 
                    batch['pitch_fine'].to(self.device).to(self.dtype),
                    batch['sids'].to(self.device),
                    noise_scale=self.config.train.noise_scale_test
                )
                for i, audio in enumerate(o):
                    audio = audio.cpu()[:, :batch['lengths'][i] * self.config.data.hop_length]
                    self.logger.experiment.add_audio(
                        tag=f'test_{i}',
                        snd_tensor=audio,
                        global_step=self.global_step,
                        sample_rate=self.config.data.sampling_rate
                    )
                self.logger.experiment.flush()

    def step(self, batch, batch_idx, is_train=True, is_val=False):
        phone_A = batch['A']['rvc_feat']
        phone_lengths_A = batch['A']['lengths']
        phone_B = batch['B']['rvc_feat']
        phone_lengths_B = batch['B']['lengths']
        pitchf_A = batch['A']['pitch_fine']
        spks_A = batch['A']['sids']
        spks_B = batch['B']['sids']
        y_A = batch['A']['spec']
        wave_A = batch['A']['wave']

        disc_optim, gen_optim = self.optimizers()

        # --- Data augmentation ---
        # Speech feature noising
        phone_A_aug = phone_A + torch.randn_like(phone_A) * self.config.train.phone_aug_scale
        phone_B_aug = phone_B + torch.randn_like(phone_B) * self.config.train.phone_aug_scale
        # Spec power modulation
        y_A_aug = smooth_random_amplitude_modulation(y_A,
            min_gain=self.config.train.spec_am_min,
            max_gain=self.config.train.spec_am_max,
            points=self.config.train.spec_am_points)
        # Spec noise
        y_A_aug = y_A_aug + torch.randn_like(y_A_aug) * self.config.train.spec_aug_scale

        if self.global_step % self.config.train.grl_every_step == 0:
            lam_grl = self.step_lerp(
                max=self.config.train.lam_grl_max,
                start=self.config.train.stage1_train_step, # GRL should not have any coefficient before this
                end=self.config.train.lam_grl_end
            )
        else:
            lam_grl = 0

        # During stage 1, the discriminator isn't touched by any gradients
        y_hat, z_mask, ids_slice, \
            (z, z_p, m_p_A, logs_p_A, m_q_A, logs_q_A), \
            (spk_loss, fake_loss, real_loss) = self.net_g(
                phone_A=phone_A_aug, phone_lengths_A=phone_lengths_A,
                phone_B=phone_B_aug, phone_lengths_B=phone_lengths_B,
                pitchf_A=pitchf_A, 
                spks_A=spks_A, spks_B=spks_B,
                y_A=y_A_aug, y_lengths_A=batch['A']['lengths'],
                # 0 to maximum
                lambda_grl = lam_grl,
                label_alpha = self.config.train.label_alpha
            )

        mel_A = spec_to_mel_torch(rearrange(y_A, 'b t d -> b d t'),
            self.config.data.n_fft, self.config.data.num_mels, 
            self.config.data.sampling_rate, self.config.data.fmin, 
            self.config.data.fmax)
        y_A_mel = slice_segments_general(
            mel_A, ids_slice, 
            self.config.data.segment_size // self.config.data.hop_length)
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
            wave_A, ids_slice * self.config.data.hop_length, 
            self.config.data.segment_size
        )
        wave = wave[:, :y_hat.shape[2]] 
        y_mel = y_A_mel[:, :, :y_hat_mel.shape[2]]

        if not self.stage1:
            # Train discriminator
            wave_noise = wave + torch.randn_like(wave) * self.config.train.disc_noise_scale
            y_hat_noise = y_hat + torch.randn_like(y_hat) * self.config.train.disc_noise_scale
            y_d_hat_r, y_d_hat_g, _, _ = self.net_d(wave_noise.unsqueeze(1), y_hat_noise.detach())
            loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g, label_alpha=
                self.config.train.label_alpha)

            if loss_disc.isnan().any():
                loss_disc = torch.zeros_like(loss_disc)
                print("Warning - NaN detected in loss_disc")
                return None

            if loss_disc.requires_grad:
                disc_optim.zero_grad()
                self.manual_backward(loss_disc)
                d_norm = torch.nn.utils.clip_grad_norm_(self.net_d.parameters(), 1000.)
                disc_optim.step()
            else:
                d_norm = None
        else:
            loss_disc = None
            d_norm = None

        # Train generator
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(wave.unsqueeze(1), y_hat)
        loss_mel = F.l1_loss(y_mel, y_hat_mel)

        # Regularization of prior to prevent numerical instability
        active_elements = torch.sum(z_mask)
        if active_elements == 0:
            loss_kl_reg = torch.tensor(0.0, device=self.device)
        else:
            loss_kl_reg = torch.sum(logs_p_A ** 2 * z_mask) * \
                self.config.train.lam_kl_reg_logs / active_elements \
                + torch.sum(m_p_A ** 2 * z_mask) * \
                self.config.train.lam_kl_reg_mus / active_elements

        loss_kl = kl_loss(z_p, logs_q_A, m_p_A, logs_p_A, z_mask) 
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, _ = generator_loss(y_d_hat_g)
        loss_gen_all = loss_gen + (
            loss_fm + 
            loss_mel*self.config.train.lam_mel + 
            loss_kl*self.config.train.lam_kl + 
            spk_loss*self.config.train.lam_spk + # Content discrminator
            (fake_loss + real_loss)*self.config.train.lam_spk # Speaker conditioned discriminator
            + loss_kl_reg
        )

        if loss_gen_all.isnan().any():
            loss_gen_all = torch.zeros_like(loss_gen_all)
            print("Warning - NaN detected in loss_gen_all")
            #import pdb; pdb.set_trace()
            return None
        
        if loss_gen_all.requires_grad:
            gen_optim.zero_grad()
            self.manual_backward(loss_gen_all)
            g_norm = torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 10000.)
            gen_optim.step()
        else:
            g_norm = None

        ret = {
            'loss_disc': loss_disc,
            'loss_gen': loss_gen,
            'loss_gen_all': loss_gen_all,
            'loss_mel': loss_mel,
            'loss_kl': loss_kl,
            'loss_kl_reg': loss_kl_reg,
            'loss_fm': loss_fm,
            'loss_spk': spk_loss,
            #'loss_c': c_loss,
            #'loss_align': align_loss,
            'loss_fake': fake_loss,
            'loss_real': real_loss,
            'd_norm': d_norm,
            'g_norm': g_norm,
        }

        if is_val and batch_idx == 0:
            audio = y_hat[0].cpu()
            self.logger.experiment.add_audio(
                tag='gen_audio',
                snd_tensor=audio,
                global_step=self.global_step,
                sample_rate=self.config.data.sampling_rate
            )

        return ret

    def training_step(self, batch, batch_idx):
        ret = self.step(batch, batch_idx, is_train=True, is_val=False)
        if ret is None:
            return None

        self.log('gen_loss', ret['loss_gen'], prog_bar=True, logger=True)
        self.log('gen_loss_all', ret['loss_gen_all'], prog_bar=True, logger=True)
        self.log('mel_loss', ret['loss_mel'], logger=True)
        self.log('kl_loss', ret['loss_kl'], logger=True)
        self.log('loss_kl_reg', ret['loss_kl_reg'], logger=True, on_step=True)
        self.log('fm_loss', ret['loss_fm'], logger=True)
        self.log('spk_loss', ret['loss_spk'], logger=True)
        #self.log('c_loss', ret['loss_c'], logger=True)
        #self.log('align_loss', ret['loss_align'], logger=True)
        self.log('spk_fake_loss', ret['loss_fake'], logger=True)
        self.log('spk_real_loss', ret['loss_real'], logger=True)
        if ret['loss_disc'] is not None:
            self.log('disc_loss', ret['loss_disc'], prog_bar=True, logger=True)
        if ret['d_norm'] is not None:
            self.log('d_norm', ret['d_norm'], logger=True)
        if ret['g_norm'] is not None:
            self.log('g_norm', ret['g_norm'], logger=True)
        return ret['loss_gen_all']

    def validation_step(self, batch, batch_idx):
        ret = self.step(batch, batch_idx, is_train=False, is_val=True)
        if ret is None:
            return None

        self.log('val_gen_loss', ret['loss_gen'], on_epoch=True, prog_bar=True, logger=True)
        self.log('val_gen_loss_all', ret['loss_gen_all'], on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mel_loss', ret['loss_mel'], on_epoch=True, logger=True)
        self.log('val_kl_loss', ret['loss_kl'], on_epoch=True, logger=True)
        self.log('val_fm_loss', ret['loss_fm'], on_epoch=True, logger=True)
        self.log('val_spk_loss', ret['loss_spk'], on_epoch=True, logger=True)
        #self.log('val_c_loss', ret['loss_c'], on_epoch=True, logger=True)
        #self.log('val_align_loss', ret['loss_align'], on_epoch=True, logger=True)
        self.log('val_spk_fake_loss', ret['loss_fake'], on_epoch=True, logger=True)
        self.log('val_spk_real_loss', ret['loss_real'], on_epoch=True, logger=True)
        
        val_loss = ret['loss_gen_all']
        if ret['loss_disc'] is not None:
            val_loss += ret['loss_disc']
            self.log('val_disc_loss', ret['loss_disc'], on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)
        return ret['loss_gen_all']

    def on_train_epoch_end(self):
        self.test()
        return super().on_train_epoch_end()

    def configure_optimizers(self):
        disc_optim = torch.optim.AdamW(
            self.net_d.parameters(), lr=self.config.train.lr, betas=(0.9, 0.999),
            weight_decay=self.config.train.weight_decay)
        gen_optim = torch.optim.AdamW(
            self.net_g.parameters(), lr=self.config.train.lr, betas=(0.9, 0.999),
            weight_decay=self.config.train.weight_decay)
        return [disc_optim, gen_optim], []

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/v07.yaml')
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
        net_g = SynthesizerV05(config, **config.model, is_half=True)
        load_state_dict_mismatch(net_g, gen_state['model'])

        disc_state = torch.load(args.disc_ckpt, map_location='cpu')
        net_d = MultiPeriodDiscriminatorV2(use_spectral_norm=False)
        net_d.load_state_dict(disc_state['model'])
        training_module = V05TrainingModule(net_g, net_d, config)

    elif args.resume_from is not None:
        print('Resuming from lightning checkpoint: {}'.format(args.resume_from))
        net_g = SynthesizerV05(config, **config.model, is_half=True)
        net_d = MultiPeriodDiscriminatorV2(use_spectral_norm=False)
        training_module = V05TrainingModule(net_g, net_d, config)
    elif args.transfer_from is not None:
        print('Transferring from lightning checkpoint: {}'.format(args.transfer_from))
        net_g = SynthesizerV05(config, **config.model, is_half=True)
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
        training_module = V05TrainingModule(net_g, net_d, config)
    else:
        print('!!! Warning: No checkpoint provided. Training from scratch. !!!')
        net_g = SynthesizerV05(config, **config.model, is_half=True)
        net_d = MultiPeriodDiscriminatorV2(use_spectral_norm=False)
        training_module = V05TrainingModule(net_g, net_d, config)

    logger = pl.loggers.TensorBoardLogger(
        config.train.get('log_dir', 'logs'), name=config.exp_name,
        version=args.version
    )

    train_dataset = FeatureDatasetPaired(config, is_train=True)
    val_dataset = FeatureDatasetPaired(config, is_train=False)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        collate_fn=paired_feature_collator,
        shuffle=True,
        num_workers=config.data.num_workers,
        persistent_workers=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        collate_fn=paired_feature_collator,
        num_workers=config.data.num_workers,
        persistent_workers=True
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
        precision='bf16',
        callbacks=[checkpoint_callback],
        # detect_anomaly=True
    )
    trainer.fit(training_module, train_dataloader, val_dataloader, ckpt_path=args.resume_from)
