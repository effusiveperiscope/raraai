import itertools

from modeling.vits import commons
from modeling.generator import Generator
from modeling.vits_decoder.discriminator import Discriminator
from modeling.pitch_discriminator import PitchDiscriminator
from dataset_ac import dataset_ac
from einops import rearrange

import warnings
import logging
warnings.filterwarnings('error', category=RuntimeWarning)
warnings.simplefilter('ignore', category=UserWarning) # pyworld spams the log with messages
logging.getLogger('numba').setLevel(logging.WARNING)

from vits_extend.stft_loss import STFTLoss
from vits_extend.stft import TacotronSTFT
from vits_extend.stft_loss import MultiResolutionSTFTLoss
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from commons import (load_state_dict_mismatch, load_submodule_prefix,
                     matplotlib_to_tensorboard, slice_segments_general)

import pytorch_lightning as pl
from omegaconf import OmegaConf
import os

def weighted_mel_loss(mel_pred, mel_target, voiced_mask, unvoiced_weight=0.1):
    """
    mel_pred, mel_target: (B, n_mels, T)
    voiced_mask: (B, T) — 1 for voiced, 0 for unvoiced
    """

    # Expand mask to mel bins
    w = voiced_mask.unsqueeze(1)  # (B, 1, T)
    w = voiced_mask_to_weight(w, unvoiced_weight)
    
    loss = F.l1_loss(mel_pred, mel_target, reduction='none')  # (B, n_mels, T)
    weighted = (loss * w).sum() / (w.sum() * mel_pred.shape[1] + 1e-8)
    return weighted

def voiced_mask_to_weight(voiced_mask, unvoiced_weight=0.1):
    # voiced=1.0, unvoiced=unvoiced_weight
    return voiced_mask + (1.0 - voiced_mask) * unvoiced_weight

class TrainModule(pl.LightningModule):
    def __init__(self,
        net_g : Generator,
        net_d : Discriminator,
        net_dp : PitchDiscriminator,
        config : OmegaConf):
        super().__init__()
        self.automatic_optimization = False

        self.net_g = net_g
        self.net_d = net_d
        self.net_dp = net_dp
        self.config = config
        self.pitch_only = bool(config.train.get('pitch_only', False))

    def setup(self, stage=None):
        hp = self.config
        self.stft = TacotronSTFT(filter_length=hp.data.filter_length,
                            hop_length=hp.data.hop_length,
                            win_length=hp.data.win_length,
                            n_mel_channels=hp.data.mel_channels,
                            sampling_rate=hp.data.sampling_rate,
                            mel_fmin=hp.data.mel_fmin,
                            mel_fmax=hp.data.mel_fmax,
                            center=False,
                            device=self.device)
        self.stft_criterion = MultiResolutionSTFTLoss(self.device, eval(hp.mrd.resolutions))
        self.f0_stft = STFTLoss(self.device, 
            fft_size=512, shift_size=50, win_length=200, window="hann_window")
        self.test_dataset = dataset_ac(self.config.train.val_filelist, is_train=False)
        self.test_dataloader = self.test_dataset.loader()
        self.hp = hp

    def configure_optimizers(self):
        gen_optim = torch.optim.AdamW(
            [p for p in self.net_g.parameters() if p.requires_grad],
            lr=self.config.train.lr,
            betas=self.config.train.betas,
            eps=self.config.train.eps)
        disc_optim = torch.optim.AdamW(
            self.net_d.parameters(),
            lr=self.config.train.lr,
            betas=self.config.train.betas,
            eps=self.config.train.eps)
        pitch_disc_optim = torch.optim.AdamW(
            self.net_dp.parameters(),
            lr=self.config.train.lr,
            betas=self.config.train.betas,
            eps=self.config.train.eps)
        gen_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=gen_optim, 
            T_max=self.config.train.get('cosine_anneal_end', 500000),
            eta_min=1e-6
        )
        disc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=disc_optim, 
            T_max=self.config.train.get('cosine_anneal_end', 500000),
            eta_min=1e-6
        )
        pitch_disc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=pitch_disc_optim, 
            T_max=self.config.train.get('cosine_anneal_end', 500000),
            eta_min=1e-6
        )
        return [gen_optim, disc_optim, pitch_disc_optim], \
            [gen_scheduler, disc_scheduler, pitch_disc_scheduler]

    def on_train_start(self):
        self.test()

    def on_train_epoch_end(self):
        self.test()
        return super().on_train_epoch_end()

    def test(self):
        if self.current_epoch % self.config.train.get('test_interval', 1):
            return
        print('=== Testing ===')
        self.net_g.eval()
        self.net_d.eval()
        self.net_dp.eval()
        # Only use first 5 validation samples for testing
        for j,batch in itertools.islice(enumerate(self.test_dataloader), 5):
            with torch.no_grad():
                for k, x in batch.items():
                    if k in {'spec', 'f0', 'spec_length'}:
                        batch[k] = x.to(self.dtype).to(self.device)
                batch['sid'] = batch['sid'].to(self.device)
                self.net_g : Generator
                spec = batch['spec']
                spec = rearrange(spec, "b t c -> b c t") # channel first is expected for some reason
                out_audio, f0_hat = self.net_g.infer(
                    spec=spec,
                    spec_l=batch['spec_length'].squeeze(-1),
                    f0=batch['f0'],
                    sid=batch['sid'],
                    use_pitch_recon=True,
                    return_pitch=True
                )

                for i, audio in enumerate(out_audio):
                    audio = audio.squeeze(0).cpu().numpy()
                    audio = audio[:int(batch['spec_length'][i] * self.config.data.hop_length)]
                    self.logger.experiment.add_audio(
                        tag=f'test_prior_{i}_{j}',
                        snd_tensor=audio,
                        global_step=self.global_step,
                        sample_rate=self.config.data.sampling_rate
                    )
                    self.log_pitch_curve(
                        f0_ref=batch['f0'][i],
                        f0_pred=f0_hat[i],
                        tag=f'test_pitch_{i}_{j}'
                    )
                self.logger.experiment.flush()

    def log_pitch_curve(self, f0_ref, f0_pred, tag):
        """Log the target and codec-reconstructed f0 contours as an image.

        Args:
            f0_ref: (T,) ground-truth f0 in Hz, 0 == unvoiced.
            f0_pred: (T,) f0 produced by the pitch codec (the contour the
                decoder uses), in Hz, 0 == unvoiced.
            tag: TensorBoard image tag.
        """
        ref = f0_ref.detach().float().cpu().reshape(-1).numpy()
        pred = f0_pred.detach().float().cpu().reshape(-1).numpy()
        n = min(ref.size, pred.size)
        if n == 0:
            return
        # 0 Hz marks an unvoiced frame; NaN leaves a gap in the curve.
        ref = np.where(ref[:n] > 0, ref[:n], np.nan)
        pred = np.where(pred[:n] > 0, pred[:n], np.nan)
        if not (np.isfinite(ref).any() or np.isfinite(pred).any()):
            return

        hop = self.config.data.hop_length
        sr = self.config.data.sampling_rate
        t = np.arange(n) * hop / sr

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(t, ref, color='tab:blue', linewidth=1.2, label='target')
        ax.plot(t, pred, color='tab:orange', linewidth=1.2,
                label='codec reconstruction')
        # Pin the y-range to the target's range (+margin) so a diverging
        # codec early in training cannot squash the reference curve.
        finite_ref = ref[np.isfinite(ref)]
        if finite_ref.size:
            ax.set_ylim(0.0, max(float(finite_ref.max()) * 1.25, 1.0))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('f0 (Hz)')
        ax.set_title(f'Pitch contour (step {self.global_step})')
        ax.set_xlim(0, max(t[-1], 1e-3))
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        fig.tight_layout()

        self.logger.experiment.add_image(
            tag, matplotlib_to_tensorboard(fig), global_step=self.global_step)
        plt.close(fig)

    def step(self, batch, batch_idx, is_train=True):
        if is_train:
            assert self.net_g.training
            assert self.net_d.training
            assert self.net_dp.training
        else:
            assert not self.net_g.training
            assert not self.net_d.training
            assert not self.net_dp.training

        optim_g, optim_d, optim_dp = self.optimizers()
        sched_g, sched_d, sched_dp = self.lr_schedulers()

        pit = batch['f0'].to(self.dtype)
        audio = batch['wave']
        spec = batch['spec']
        spec = rearrange(spec, "b t c -> b c t") # channel first is expected for some reason
        spec_l = batch['spec_length'].squeeze(-1)
        sid = batch['sid']
        hp = self.hp
        pitch_only = self.pitch_only

        fake_audio, ids_slice, yq_ds, x_ds, vqloss, perplexity, \
            f0_hat, f0_target, pq_ds, px_ds, p_vqloss, p_perplexity, \
            p_uv_logits = self.net_g(
            spec, spec_l, pit, sid
        )
        audio = commons.slice_segments(
            audio.unsqueeze(1), ids_slice * hp.data.hop_length, hp.data.segment_size)  # slice

        # Sometimes these have slightly different lengths. Should still be aligned
        min_dim = min(fake_audio.shape[2], audio.shape[2])
        fake_audio = fake_audio[:, :, :min_dim]
        audio = audio[:, :, :min_dim]

        voiced_mask_ds = F.interpolate(
            (pit > 0).unsqueeze(1).float(), 
            size=(pit.size(-1) + 3) // 4).transpose(1,2) # 1/4 factor

        # f0 shape: (B, T) — 0 or NaN on unvoiced frames
        f0_sliced = slice_segments_general(pit, ids_slice, hp.data.segment_size // hp.data.hop_length)
        voiced_mask_sliced = (f0_sliced > 0).float()  # (B, T), 1=voiced, 0=unvoiced

        q_recon_loss = (F.mse_loss(yq_ds, x_ds,
            reduction='none') * voiced_mask_ds).sum() / (
                voiced_mask_ds.sum())

        # Pitch decoding loss: train the low-parameter pitch codec to
        # reconstruct the (log-normalized) f0 contour. Only voiced frames
        # carry pitch, so mask the MSE to f0_target > 0.
        # f0_hat / f0_target: (B, T_full) log-normalized pitch.
        pitch_voiced = (f0_target > 0).float()
        pitch_denom = pitch_voiced.sum().clamp_min(1.0)
        pitch_dec_loss = ((f0_hat - f0_target) ** 2 * pitch_voiced).sum() / pitch_denom
        # Pitch code-space reconstruction (mirrors q_recon_loss for spec).
        pitch_voiced_ds = F.interpolate(
            pitch_voiced.unsqueeze(1),
            size=pq_ds.shape[1]).transpose(1, 2)  # (B, T_ds, 1)
        pitch_q_recon_loss = (F.mse_loss(pq_ds, px_ds,
            reduction='none') * pitch_voiced_ds).sum() / (
                pitch_voiced_ds.sum().clamp_min(1.0))
        c_pitch_dec = self.config.train.get('c_pitch_dec', 1.0)
        c_pitch_q = self.config.train.get('c_pitch_q', 1.0)
        pitch_dec_loss = pitch_dec_loss * c_pitch_dec
        pitch_q_recon_loss = pitch_q_recon_loss * c_pitch_q

        # Voicing (V/UV) objective: the pitch codec predicts, from its codes,
        # whether a frame is voiced. This lets stage-2 synthesis render
        # unvoiced frames without an external voicing mask. BCE is applied on
        # every frame (1 == voiced, 0 == unvoiced).
        c_pitch_uv = self.config.train.get('c_pitch_uv', 1.0)
        c_pitch_uv_frame = self.config.train.get('c_pitch_uv_frame', 0.1)
        uv_target = pitch_voiced
        pitch_uv_loss = F.binary_cross_entropy_with_logits(
            p_uv_logits, uv_target) * c_pitch_uv
        # Also regress the decoded log-pitch towards 0 on unvoiced frames so a
        # mis-gated frame renders silence rather than a stray tone.
        pitch_unvoiced = 1.0 - pitch_voiced
        pitch_uv_frame_loss = ((f0_hat ** 2) * pitch_unvoiced).sum() / (
            pitch_unvoiced.sum().clamp_min(1.0)) * c_pitch_uv_frame

        # Smooth the mask slightly to avoid hard discontinuities at transitions
        # (optional but helps — unvoiced/voiced boundaries are ambiguous)
        voiced_mask_smooth = F.max_pool1d(
            voiced_mask_sliced.unsqueeze(1), kernel_size=5, stride=1, padding=2
        ).squeeze(1)

        # Spec-path reconstruction / adversarial losses. In pitch-only mode the
        # spectrogram codec, decoder and main discriminator are frozen, so these
        # are skipped entirely (no gradient flows to them).
        mel_loss = stft_loss = score_loss = feat_loss = None
        if not pitch_only:
            # Mel Loss
            mel_fake = self.stft.mel_spectrogram(fake_audio.squeeze(1))
            try:
                mel_real = self.stft.mel_spectrogram(audio.squeeze(1))
            except AssertionError:
                print("mel_real error, skip") # some audio triggers this for some reason, but not most
                return None
            mel_loss = weighted_mel_loss(mel_fake, mel_real, voiced_mask_smooth,
                unvoiced_weight = self.config.train.get('c_unvoiced', 0.8)) * hp.train.c_mel

            # multi-resolution stft loss
            sc_loss, mag_loss = self.stft_criterion(
                fake_audio.squeeze(1), audio.squeeze(1), voiced_mask_mel=voiced_mask_smooth)
            stft_loss = (sc_loss + mag_loss) * hp.train.c_stft

            # Generator Loss
            disc_fake = self.net_d(fake_audio)
            score_loss = 0.0
            for (_, score_fake) in disc_fake:
                score_loss += torch.mean(torch.pow(score_fake - 1.0, 2))
            score_loss = score_loss / len(disc_fake)

            # Feature Loss
            disc_real = self.net_d(audio)
            feat_loss = 0.0
            for (feat_fake, _), (feat_real, _) in zip(disc_fake, disc_real):
                for fake, real in zip(feat_fake, feat_real):
                    feat_loss += torch.mean(torch.abs(fake - real))
            feat_loss = feat_loss / len(disc_fake)
            feat_loss = feat_loss * 2

        # --- Pitch adversarial: patch critic over the reconstructed f0 ---
        # real/fake are log-normalized pitch contours (B, T) with unvoiced
        # frames zeroed. The critic is fed the contour *slope* internally and
        # only scores voiced windows, so it judges the shape of the movement
        # rather than the level already constrained by pitch_dec_loss.
        pitch_real = f0_target.unsqueeze(1)                    # (B, 1, T)
        pitch_fake = (f0_hat * pitch_voiced).unsqueeze(1)      # (B, 1, T)
        c_pitch_adv = self.config.train.get('c_pitch_adv', 1.0)
        c_pitch_feat = self.config.train.get('c_pitch_feat', 1.0)

        pitch_disc_fake = self.net_dp(pitch_fake, voiced=pitch_voiced)
        pitch_score_loss = 0.0
        for (_, score_fake) in pitch_disc_fake:
            pitch_score_loss += torch.mean(torch.pow(score_fake - 1.0, 2))
        pitch_score_loss = pitch_score_loss / len(pitch_disc_fake)

        pitch_disc_real = self.net_dp(pitch_real, voiced=pitch_voiced)
        pitch_feat_loss = 0.0
        for (feat_fake, _), (feat_real, _) in zip(pitch_disc_fake, pitch_disc_real):
            for fake, real in zip(feat_fake, feat_real):
                pitch_feat_loss += torch.mean(torch.abs(fake - real))
        pitch_feat_loss = pitch_feat_loss / len(pitch_disc_fake)
        pitch_feat_loss = pitch_feat_loss * 2

        # Loss
        if pitch_only:
            loss_g = p_vqloss.sum() + pitch_dec_loss + pitch_q_recon_loss \
                + pitch_uv_loss + pitch_uv_frame_loss \
                + c_pitch_adv * pitch_score_loss + c_pitch_feat * pitch_feat_loss
        else:
            loss_g = score_loss + feat_loss + mel_loss + stft_loss + vqloss.sum() + q_recon_loss \
                + p_vqloss.sum() + pitch_dec_loss + pitch_q_recon_loss \
                + pitch_uv_loss + pitch_uv_frame_loss \
                + c_pitch_adv * pitch_score_loss + c_pitch_feat * pitch_feat_loss

        if loss_g.requires_grad:
            optim_g.zero_grad()
            self.manual_backward(loss_g)
            optim_g.step()
            sched_g.step()

        # Discriminator Loss (main audio discriminator — skipped in pitch-only
        # mode, where net_d is frozen and does not participate).
        loss_d = None
        if not pitch_only:
            disc_fake = self.net_d(fake_audio.detach())
            disc_real = self.net_d(audio)
            loss_d = 0.0
            for (_, score_fake), (_, score_real) in zip(disc_fake, disc_real):
                loss_d += torch.mean(torch.pow(score_real - 1.0, 2))
                loss_d += torch.mean(torch.pow(score_fake, 2))
            loss_d = loss_d / len(disc_fake)

            if loss_d.requires_grad:
                optim_d.zero_grad()
                self.manual_backward(loss_d)
                optim_d.step()
                sched_d.step()

        # Pitch Discriminator Loss
        pitch_disc_fake = self.net_dp(pitch_fake.detach(), voiced=pitch_voiced)
        pitch_disc_real = self.net_dp(pitch_real, voiced=pitch_voiced)
        loss_dp = 0.0
        for (_, score_fake), (_, score_real) in zip(pitch_disc_fake, pitch_disc_real):
            loss_dp += torch.mean(torch.pow(score_real - 1.0, 2))
            loss_dp += torch.mean(torch.pow(score_fake, 2))
        loss_dp = loss_dp / len(pitch_disc_fake)

        if loss_dp.requires_grad:
            optim_dp.zero_grad()
            self.manual_backward(loss_dp)
            optim_dp.step()
            sched_dp.step()

        return {'loss_g': loss_g, 'loss_d': loss_d, 'loss_dp': loss_dp,
                'mel_loss': mel_loss,
                'stft_loss': stft_loss, 'score_loss': score_loss, 'feat_loss': feat_loss,
                'vqloss': vqloss.sum(), 
                'q_recon_loss': q_recon_loss.sum(), 
                'p_vqloss': p_vqloss.sum(),
                'pitch_dec_loss': pitch_dec_loss.sum(),
                'pitch_q_recon_loss': pitch_q_recon_loss.sum(),
                'pitch_uv_loss': pitch_uv_loss,
                'pitch_uv_frame_loss': pitch_uv_frame_loss,
                'pitch_score_loss': c_pitch_adv * pitch_score_loss,
                'pitch_feat_loss': c_pitch_feat * pitch_feat_loss,
                'perplexity': perplexity.sum(),
                'p_perplexity': p_perplexity.sum()}

    def training_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        self.net_g.train()
        self.net_d.train()
        self.net_dp.train()
        ret = self.step(batch, batch_idx, is_train=True)
        if ret is None:
            return None
        
        prog_bar_set = {'loss_g', 'loss_d'}
        for k,v in ret.items():
            if v is None:
                continue
            self.log(k, v, prog_bar=k in prog_bar_set, logger=True)
        return ret

    def validation_step(self, batch, batch_idx):
        self.net_g.eval()
        self.net_d.eval()
        self.net_dp.eval()
        ret = self.step(batch, batch_idx, is_train=False)
        if ret is None:
            return None
        
        for k,v in ret.items():
            if v is None:
                continue
            self.log('val_' + k, v, logger=True)

        val_loss = ret['loss_g']
        self.log('val_loss', val_loss, prog_bar=True, logger=True)

        return val_loss

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    parser.add_argument('--r22_ckpt', type=str, default=None)
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--transfer_from', type=str, default=None)
    parser.add_argument('--pitch_only', action='store_true',
        help='Freeze all components except the pitch codec (and keep only the '
             'pitch discriminator), training pitch losses alone.')

    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    if args.pitch_only:
        config.train.pitch_only = True
    return config, args

def train(config,
    resume_from=None,
    transfer_from=None,
    r22_ckpt=None):
    hp = config
    net_g = Generator(hp=hp)
    net_d = Discriminator(hp=hp)
    pitch_disc_cfg = config.get('pitch_discriminator', {})
    net_dp = PitchDiscriminator(
        channels=int(pitch_disc_cfg.get('channels', 32)),
        n_scales=int(pitch_disc_cfg.get('n_scales', 3)),
        kernel_size=int(pitch_disc_cfg.get('kernel_size', 5)),
        delta_scale=float(pitch_disc_cfg.get('delta_scale', 50.0)))

    if os.path.exists(f'checkpoints/{config.exp_name}/last.ckpt'):
        print('Detected interrupted training - resuming from last.ckpt')
        resume_from = f'checkpoints/{config.exp_name}/last.ckpt'
    if resume_from is not None:
        print('Resuming from lightning checkpoint: {}'.format(resume_from))
        d = torch.load(resume_from, map_location='cpu', weights_only=False)
        if d['global_step'] >= config.train.get('max_steps', 3000000):
            print('Maximum steps reached. Exiting.')
            return
    elif transfer_from is not None:
        print('Transferring from lightning checkpoint: {}'.format(transfer_from))
        state = torch.load(transfer_from, map_location='cpu', weights_only=False)['state_dict']
        load_submodule_prefix(net_g, 'net_g.', state)
        load_submodule_prefix(net_d, 'net_d.', state)
        load_submodule_prefix(net_dp, 'net_dp.', state)
    elif r22_ckpt is not None:
        print("Loading 22 checkpoint: {}".format(r22_ckpt))
        state_dict = torch.load(r22_ckpt, map_location='cpu', weights_only=False)['state_dict']
        load_submodule_prefix(net_g, 'net_g.', state_dict)
        load_submodule_prefix(net_d, 'net_d.', state_dict)
        load_submodule_prefix(net_dp, 'net_dp.', state_dict)
    else:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!! No checkpoint file found - starting from scratch !!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')


    pitch_only = bool(config.train.get('pitch_only', False))
    if pitch_only:
        print('=== Pitch-only mode: freezing all non-pitch components ===')
        net_g.freeze_non_pitch()
        for p in net_d.parameters():
            p.requires_grad_(False)

    training_module = TrainModule(
        net_g=net_g, net_d=net_d, net_dp=net_dp, config=config)

    logger = pl.loggers.TensorBoardLogger(
        config.train.get('log_dir', 'logs'), name=config.exp_name,
        version=config.get('tensorboard_version', 0)
    )
    print("Loading data...")
    train_dataset = dataset_ac(config.train.train_filelist, is_train=True)
    val_dataset = dataset_ac(config.train.val_filelist, is_train=False)
    print("Creating dataloaders...")
    num_workers = config.train.get('num_workers', 4)
    #num_workers = 0
    train_dataloader = train_dataset.loader(
        batch_size=config.train.batch_size, shuffle=True, num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=config.train.get('pin_memory', False))
    val_dataloader = val_dataset.loader(
        batch_size=config.train.batch_size, shuffle=False, num_workers=num_workers, 
            persistent_workers=num_workers > 0,
            pin_memory=config.train.get('pin_memory', False))
    print("Done")

    val_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'checkpoints/{config.exp_name}',
        filename='best-checkpoint',
        save_top_k=config.train.keep_ckpts,
        mode='min',
    )

    default_last_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f'checkpoints/{config.exp_name}',
        filename='last',
        every_n_epochs=1,
        save_top_k=-1,
    )
    default_interval_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f'checkpoints/{config.exp_name}',
        filename='model-{step:06d}',
        every_n_train_steps=200000,
        save_top_k=-1,
    )
    callbacks = [val_checkpoint_callback, default_interval_callback, default_last_callback]
    if config.train.get('save_interval') is not None:
        interval_checkpoint_callback = pl.callbacks.ModelCheckpoint(
            every_n_epochs=config.train.save_interval,
            dirpath=f'checkpoints/{config.exp_name}',
            filename='interval-checkpoint-{epoch:04d}',
            save_top_k=-1,
        )
        callbacks.append(interval_checkpoint_callback)

    trainer = pl.Trainer(
        logger=logger,
        accelerator='gpu',
        precision='bf16-mixed',
        max_steps=config.train.get('max_steps', -1),
        max_epochs=config.train.get('max_epochs', -1),
        callbacks=callbacks,
        check_val_every_n_epoch=config.train.get('val_interval', 1),
        #val_check_interval=2,
        log_every_n_steps=config.train.get('log_interval', 50),
    )
    trainer.fit(training_module, train_dataloader, val_dataloader, ckpt_path=resume_from,
        weights_only=False)

if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True) # This is needed on Linux

    config, args = parse_args()

    # Something above this is setting DEBUG but I don't know what
    logging.getLogger().setLevel(logging.WARNING)

    train(config,
        resume_from=args.resume_from,
        transfer_from=args.transfer_from,
        r22_ckpt=args.r22_ckpt)