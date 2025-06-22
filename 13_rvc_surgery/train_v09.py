from einops import rearrange
from svc_helper.svc.rvc.lib.infer_pack.models import MultiPeriodDiscriminatorV2
from svc_helper.pitch.utils import discretize_f0_log
from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from commons import load_state_dict_mismatch, load_submodule_prefix, slice_segments_general, smooth_random_amplitude_modulation
from dataset_paired import FeatureDatasetPaired, paired_feature_collator
from dataset import FeatureDataset, FeatureCollator
from features import MyFeatures
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from modeling.v09.rvc import V09Synthesizer
from svc_helper.svc.rvc.lib.infer_pack import commons
from rvc_losses import discriminator_loss, feature_loss, generator_loss, kl_loss

import pdb
import sys
import traceback
def custom_excepthook(exc_type, exc_value, exc_traceback):
    """
    Custom exception hook that prints the exception information
    and then drops into a pdb debugger session.
    """
    # First, print the exception information as Python normally would.
    # We use traceback.print_exception to ensure consistent formatting.
    print("An unhandled exception occurred:")
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("\nDropping into debugger...")

    # Then, drop into the pdb debugger.
    # The post_mortem function starts the debugger at the point of the exception.
    pdb.post_mortem(exc_traceback)

# Set the custom exception hook
sys.excepthook = custom_excepthook



class V09TrainingModule(pl.LightningModule):
    def __init__(self, 
        net_g : V09Synthesizer, 
        net_d: MultiPeriodDiscriminatorV2, 
        config : OmegaConf):
        super().__init__()
        self.net_g = net_g
        self.net_d = net_d
        self.config = config
        self.automatic_optimization = False
        self.stage1 = False

    def on_train_start(self):
        self.test()

    def step_lerp(self, min=0, max=1, start=0, end=10000):
        if self.global_step < start:
            return min
        elif self.global_step < end:
            return (self.global_step - start) / (end - start) * (max - min) + min
        else:
            return max

    def on_train_batch_start(self):
        if self.global_step < self.config.train.get('stage1_train_step', 0):
            self.stage1 = True
            print("=== Stage 1 ===")
            # Freeze everything except the prior
            for param in self.net_d.parameters():
                param.requires_grad = False
            for param in self.net_g.parameters():
                param.requires_grad = False
            for param in self.net_g.enc_p.parameters():
                param.requires_grad = True
        else:
            print("=== Stage 2 ===")
            self.stage1 = False
            for param in self.net_d.parameters():
                param.requires_grad = True
            for param in self.net_g.parameters():
                param.requires_grad = True

    def test(self):
        if self.current_epoch % self.config.train.get('test_every_n_epochs', 1):
            return
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
                self.net_g : V09Synthesizer
                o, x_mask, z_stats = self.net_g.infer(
                    phone=batch['whisp_feat'].to(self.device).to(self.dtype), 
                    phone_lengths=batch['lengths'].to(self.device), 
                    nsff0=batch['pitch_fine'].to(self.device).to(self.dtype),
                    sid=batch['sids'].to(self.device),
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
        phone_A = batch['A']['whisp_feat']
        phone_lengths_A = batch['A']['lengths']
        phone_B = batch['B']['whisp_feat']
        phone_lengths_B = batch['B']['lengths']
        pitchf_A = batch['A']['pitch_fine'].to(phone_A.dtype)
        pitchf_B = batch['B']['pitch_fine'].to(phone_A.dtype)
        sids_A = batch['A']['sids']
        spk_feat_A = batch['A']['spk_feat']
        y_A = batch['A']['spec']
        y_lengths_A = batch['A']['lengths']
        wave_A = batch['A']['wave']

        disc_optim, gen_optim = self.optimizers()
        disc_scheduler, gen_scheduler = self.lr_schedulers()

        # --- Data augmentation ---
        # Speech feature noising
        phone_aug_A = phone_A + torch.randn_like(phone_A) * self.config.train.phone_aug_scale
        phone_aug_B = phone_B + torch.randn_like(phone_B) * self.config.train.phone_aug_scale
        # Spec power modulation
        y_aug_A = smooth_random_amplitude_modulation(y_A,
            min_gain=self.config.train.spec_am_min,
            max_gain=self.config.train.spec_am_max,
            points=self.config.train.spec_am_points)
        # Spec noise
        y_aug_A = y_aug_A + torch.randn_like(y_aug_A) * self.config.train.spec_aug_scale

        # if self.config.model.get('use_pitch_predictor', False):
            # pitchq = []
            # for f0 in pitchf:
                # pitchq.append(
                    # discretize_f0_log(
                        # f0, 
                        # self.config.model.get('pitch_quant_dim', 8), 
                        # hold_length=10))
        # else:
            # pitchq = None
        pitchq_A = None # We will not train this for now

        y_hat, z_mask, ids_slice, \
            (z, z_p, m_p, logs_p, m_q, logs_q), \
                (loss_content_inv, spk_fake_loss, spk_real_loss) = self.net_g(
                    phone_A = phone_aug_A, phone_A_mask = commons.sequence_mask(phone_lengths_A, phone_A.size(1)),
                    phone_B = phone_aug_B, phone_B_mask = commons.sequence_mask(phone_lengths_B, phone_B.size(1)),
                    pitchf_A = pitchf_A, pitchf_B = pitchf_B,
                    y_A = y_aug_A, y_lengths_A = y_lengths_A,
                    spk_A = sids_A, spk_feat_A = spk_feat_A,
                    lam_grl = self.config.train.lam_grl,
                    pitchq_A = pitchq_A
                )

        mel = spec_to_mel_torch(rearrange(y_A, 'b t d -> b d t'),
            self.config.data.n_fft, self.config.data.num_mels, 
            self.config.data.sampling_rate, self.config.data.fmin, 
            self.config.data.fmax)
        y_mel = slice_segments_general(
            mel, ids_slice, 
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
        y_mel = y_mel[:, :, :y_hat_mel.shape[2]]

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
            disc_scheduler.step()
        else:
            d_norm = None

        # Train generator
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(wave.unsqueeze(1), y_hat)
        loss_mel = F.l1_loss(y_mel, y_hat_mel)

        # Regularization of prior to prevent numerical instability
        active_elements = torch.sum(z_mask)
        if active_elements == 0:
            loss_kl_reg = torch.tensor(0.0, device=self.device)
        else:
            loss_kl_reg = torch.sum(logs_p ** 2 * z_mask) * \
                self.config.train.lam_kl_reg_logs / active_elements \
                + torch.sum(m_p ** 2 * z_mask) * \
                self.config.train.lam_kl_reg_mus / active_elements

        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) 
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, _ = generator_loss(y_d_hat_g)
        loss_spk_disc = spk_fake_loss + spk_real_loss
        loss_gen_all = loss_gen + (
            loss_fm + 
            loss_mel*self.config.train.lam_mel + 
            loss_kl*self.config.train.lam_kl + 
            loss_content_inv*self.config.train.lam_content_inv +
            loss_spk_disc * self.config.train.lam_spk_disc
            + loss_kl_reg)

        # if f0_pred is not None:
        #     loss_f0 = F.l1_loss(f0_pred, pitchf)
        #     loss_gen_all += loss_f0 * self.config.train.get('lam_f0', 0.0)

        if loss_gen_all.isnan().any():
            loss_gen_all = torch.zeros_like(loss_gen_all)
            print("Warning - NaN detected in loss_gen_all")
            return None
        
        if loss_gen_all.requires_grad:
            gen_optim.zero_grad()
            self.manual_backward(loss_gen_all)
            g_norm = torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 10000.)
            gen_optim.step()
            gen_scheduler.step()
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
            'loss_content_inv': loss_content_inv,
            'loss_spk_disc': loss_spk_disc,
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
        self.log('content_inv_loss', ret['loss_content_inv'], logger=True)
        self.log('spk_disc_loss', ret['loss_spk_disc'], logger=True)
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
        self.log('val_content_inv_loss', ret['loss_content_inv'], logger=True)
        self.log('val_spk_disc_loss', ret['loss_spk_disc'], logger=True)
        
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
        disc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=disc_optim, 
            T_max=self.config.train.get('cosine_anneal_end', 50000),
            eta_min=1e-6
        )
        gen_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=gen_optim, 
            T_max=self.config.train.get('cosine_anneal_end', 50000),
            eta_min=1e-6
        )
        return [disc_optim, gen_optim], [disc_scheduler, gen_scheduler]



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/sing_base.yaml')
    parser.add_argument('--stage1_ckpt', type=str, default=None)
    parser.add_argument('--disc_ckpt', type=str, default=None) # RVC D_ checkpoint
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--transfer_from', type=str, default=None)
    parser.add_argument('--version', type=int, default=None, help='tensorboard log version')

    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    net_g = V09Synthesizer(config)
    net_d = MultiPeriodDiscriminatorV2(use_spectral_norm=False)
    if args.stage1_ckpt is not None:
        print('Using stage 1 checkpoint:', args.stage1_ckpt)
        state_dict = torch.load(args.stage1_ckpt, map_location='cpu')['state_dict']
        load_submodule_prefix(net_g, 'net_g.', state_dict)
        state_dict = torch.load(args.disc_ckpt, map_location='cpu')['model'] 
        load_state_dict_mismatch(net_d, state_dict)
    elif args.resume_from is not None:
        print('Resuming from lightning checkpoint: {}'.format(args.resume_from))
    elif args.transfer_from is not None:
        print('Transferring from lightning checkpoint: {}'.format(args.transfer_from))
        state = torch.load(args.transfer_from, map_location='cpu')['state_dict']
        load_submodule_prefix(net_g, 'net_g.', state)
        load_submodule_prefix(net_d, 'net_d.', state)
    else:
        print('!!! No checkpoint file found - starting from scratch !!!')
    training_module = V09TrainingModule(net_g, net_d, config)
        
    logger = pl.loggers.TensorBoardLogger(
        config.train.get('log_dir', 'logs'), name=config.exp_name,
        version=args.version
    )
    train_dataset = FeatureDatasetPaired(config, is_train=True)
    val_dataset = FeatureDatasetPaired(config, is_train=False)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = config.train.batch_size,
        shuffle = True,
        collate_fn = paired_feature_collator,
        num_workers=4,
        persistent_workers=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = config.train.batch_size,
        shuffle = False,
        collate_fn = paired_feature_collator,
        num_workers=4,
        persistent_workers=True
    )

    val_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'checkpoints/teacher/{config.exp_name}',
        filename='best-checkpoint',
        save_top_k=2,
        mode='min',
        save_last=True
    )
    callbacks = [val_checkpoint_callback]
    if config.train.get('save_every_n_epochs'):
        interval_checkpoint_callback = pl.callbacks.ModelCheckpoint(
            every_n_epochs=config.train.save_every_n_epochs,
            dirpath=f'checkpoints/teacher/{config.exp_name}',
            filename='interval-checkpoint-{epoch:04d}',
            save_top_k=-1
        )
        callbacks.append(interval_checkpoint_callback)

    trainer = pl.Trainer(
        logger=logger,
        accelerator='gpu',
        precision='bf16-mixed',
        max_epochs=config.train.epochs,
        callbacks=callbacks,
    )
    trainer.fit(training_module, train_dataloader, val_dataloader, ckpt_path=args.resume_from)
