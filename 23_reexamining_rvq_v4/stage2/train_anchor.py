import warnings
import librosa
import logging
warnings.filterwarnings('error', category=RuntimeWarning)
warnings.simplefilter('ignore', category=UserWarning) # pyworld spams the log with messages
logging.getLogger('numba').setLevel(logging.WARNING)

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.combined_loader import CombinedLoader
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
import random
import ultimate_xc
import math

from modeling.vits.models import SynthesizerTrn
from svc_helper.svc.rvc.lib.infer_pack.models import MultiPeriodDiscriminatorV2
from rvc_losses import generator_loss, feature_loss, discriminator_loss
from vits_extend.stft_loss import STFTLoss
from modeling.vits.losses import kl_loss
from modeling.vits import commons
from modeling.intensity import IntensityModel
from vits_extend.stft import TacotronSTFT
from vits_extend.stft_loss import MultiResolutionSTFTLoss
from svc_helper.pitch.utils import nonzero_mean, discretize_f0_log
from dataset import dataset
from commons import load_state_dict_mismatch, load_submodule_prefix, slice_segments_general
from utils import dump_batched_audio, dump_batched_spectrogram
import itertools
import sys
sys.path.append('..')
from rvq.vevo_repcodec import VevoRepCodec

import warnings
import numpy as np
import librosa
import logging
warnings.filterwarnings('error', category=RuntimeWarning)
logging.getLogger('numba').setLevel(logging.WARNING)

class TrainingModule(pl.LightningModule):
    def __init__(self,
        net_g : SynthesizerTrn, 
        net_d: MultiPeriodDiscriminatorV2, 
        net_i: IntensityModel,
        codec : VevoRepCodec,
        config : OmegaConf):
        super().__init__()
        self.net_g = net_g
        self.net_d = net_d
        self.net_i = net_i
        self.codec = codec
        self.config = config
        self.automatic_optimization = False

        self.use_adv = True # use adversarial losses
        self.spk_index = torch.load(self.config.train.spk_index)

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
        self.spkc_criterion = nn.CosineEmbeddingLoss()
        self.test_dataset = dataset(self.config.train.test_filelist, is_test=False)
        self.test_dataloader = self.test_dataset.loader()

    def test(self):
        if self.current_epoch % self.config.train.get('test_interval', 1):
            return
        print('=== Testing ===')
        self.net_g.eval()
        self.net_d.eval()
        for j,batch in enumerate(self.test_dataloader):
            with torch.no_grad():
                # because we created this dataloader ourselves
                # we have to manually move the data to the device
                ppg = batch['whisper'].to(self.dtype).to(self.device)
                ppg_interp = F.interpolate(rearrange(ppg, 'b t d -> b d t'), scale_factor=2)
                ppg_interp = rearrange(ppg_interp, 'b d t -> b t d')
                ppg_len = batch['whisper_length'] * 2

                _, ppg_q, _, _, _ = self.codec(ppg)
                ppg_q = F.interpolate(ppg_q, scale_factor=2)
                ppg_q = rearrange(ppg_q, "b c t -> b t c")

                f0 = batch['f0'].to(self.dtype).to(self.device)
                ppg_interp = ppg_interp[:,:f0.shape[1],:]
                ppg_q = ppg_q[:,:f0.shape[1],:]
                f0 = f0 * (2 ** (self.config.train.get('test_transpose', 0) / 12))

                f0_confidence = batch['f0_confidence'].to(ppg_interp.dtype).to(self.device)
                f0_subharmonic = batch['f0_subharmonic'].to(ppg_interp.dtype).to(self.device)
                f0_inharmonic = batch['f0_inharmonic'].to(ppg_interp.dtype).to(self.device)
                pitch_extras = {
                    'confidence': f0_confidence,
                    'subharmonic': f0_subharmonic,
                    'inharmonic': f0_inharmonic
                }

                key = self.config.train.get('test_sid', 0)
                if not key in self.spk_index:
                    key = str(key)
                spk = self.spk_index[key].to(self.dtype).to(self.device).unsqueeze(0)

                spec = batch['spec'].to(self.dtype).to(self.device).transpose(1,2)
                sid = batch['sid'].to(self.device)

                ppg_mask = commons.sequence_mask(ppg_len, max_length=f0.shape[1]).to(self.device)
                intensity_feat, attn = self.net_i(ppg_interp, ppg_mask)
                intensity = (intensity_feat * attn).sum(dim=1).unsqueeze(1)

                out_audio = self.net_g.infer(ppg_q, f0, spk, ppg_len, sid=sid, noise_scale = 
                    self.config.train.get('test_noise_scale', 0.34), pitch_extras=pitch_extras,
                    intensity=intensity)
                out_audio_post = self.net_g.posterior_test(spec, 
                    batch['spec_length'].to(self.device), f0, spk, 
                    pitch_extras=pitch_extras)

            for i, audio in enumerate(out_audio):
                audio = audio.squeeze(0).cpu().numpy()
                audio = audio[:int(ppg_len[i] * self.config.data.hop_length)]
                self.logger.experiment.add_audio(
                    tag=f'test_prior_{i}_{j}',
                    snd_tensor=audio,
                    global_step=self.global_step,
                    sample_rate=self.config.data.sampling_rate
                )
            for i, audio in enumerate(out_audio_post):
                audio = audio.squeeze(0).cpu().numpy()
                self.logger.experiment.add_audio(
                    tag=f'test_posterior_{i}_{j}',
                    snd_tensor=audio,
                    global_step=self.global_step,
                    sample_rate=self.config.data.sampling_rate
                )
            self.logger.experiment.flush()

    def on_train_start(self):
        self.test()

    def on_train_epoch_start(self):
        self.update_stage()
        if 'freeze_layers' in self.config.train:
            self.net_g.freeze_layers(**self.config.train.freeze_layers)

    def on_train_epoch_end(self):
        self.test()
        return super().on_train_epoch_end()

    def update_stage(self):
        pass

    def configure_optimizers(self):
        gen_optim = torch.optim.AdamW(
            self.net_g.parameters(), lr=self.config.train.lr, betas=self.config.train.betas,
            weight_decay=self.config.train.weight_decay)
        disc_optim = torch.optim.AdamW(
            itertools.chain(
                self.net_d.parameters()), lr=
                    self.config.train.lr * self.config.train.get('disc_lr_mul', 1.0),
                    betas=self.config.train.betas,
            weight_decay=self.config.train.weight_decay)
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
        return [gen_optim, disc_optim], [gen_scheduler, disc_scheduler]

    def feats_from_batch(self, batch):
        ppg = batch['whisper']
        ppg_interp = F.interpolate(rearrange(ppg, 'b t d -> b d t'), scale_factor=2)
        ppg_interp = rearrange(ppg_interp, 'b d t -> b t d')
        ppg_len = batch['whisper_length'] * 2

        with torch.no_grad():
            _, ppg_q, _, _, _ = self.codec(ppg)
            ppg_q = F.interpolate(ppg_q, scale_factor=2) # upsample quantized latent
            ppg_q = rearrange(ppg_q, "b c t -> b t c")

        f0 = batch['f0'].to(ppg_interp.dtype)
        ppg_q = ppg_q[:,:f0.shape[1],:]
        ppg_interp = ppg_interp[:,:f0.shape[1],:]
        f0_confidence = batch['f0_confidence'].to(ppg_interp.dtype)
        f0_subharmonic = batch['f0_subharmonic'].to(ppg_interp.dtype)
        f0_inharmonic = batch['f0_inharmonic'].to(ppg_interp.dtype)

        pitch_extras = {
            'confidence': f0_confidence,
            'subharmonic': f0_subharmonic,
            'inharmonic': f0_inharmonic
        }
        spec = batch['spec']
        spec = rearrange(spec, "b t c -> b c t") # channel first is expected for some reason
        spec_len = batch['spec_length']
        spk = batch['spk']
        wave = batch['wave']
        sid = batch['sid']
        return ppg, ppg_interp, ppg_q, ppg_len, f0, pitch_extras, spec, spec_len, spk, wave, sid

    def step(self, batch, batch_idx, is_train=True):
        hp = self.config
        optim_g, optim_d = self.optimizers()
        sched_g, sched_d = self.lr_schedulers()

        if 'anchor' in batch:
            # Anchor training
            anchor = batch['anchor']
            _, ppg_interp, ppg_q, ppg_len, f0, pitch_extras, spec, spec_len, spk, wave, sid = \
                self.feats_from_batch(anchor)
            with torch.no_grad():
                ppg_mask = commons.sequence_mask(ppg_len, max_length=f0.shape[1])
                intensity_feat, attn = self.net_i(ppg_interp, ppg_mask)
                intensity = (intensity_feat * attn).sum(dim=1).unsqueeze(1)
            if random.random() < self.config.train.get('p_pit_extra', 0.5):
                pitch_extras = None # unconditional
            fake_audio, ids_slice, z_mask, \
                (z_f, z_r, z_p, m_p, logs_p, z_q, m_q, 
                logs_q, logdet_f, logdet_r) = self.net_g(
                    ppg_q, f0, spec, spk, ppg_len, spec_len, sid=sid+1, # offset because these are anchors
                    intensity=intensity,
                    pitch_extras=pitch_extras)

            # Kl Loss
            loss_kl_f = kl_loss(z_f, logs_q, m_p, logs_p, logdet_f, z_mask) * hp.train.c_kl
            loss_kl_r = kl_loss(z_r, logs_p, m_q, logs_q, logdet_r, z_mask) * hp.train.c_kl

            audio = slice_segments_general(
                wave.unsqueeze(1), ids_slice * hp.data.hop_length, hp.data.segment_size)  # slice
            min_dim = min(fake_audio.shape[2], audio.shape[2])
            fake_audio = fake_audio[:, :, :min_dim]
            audio = audio[:, :, :min_dim]

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = self.net_d(audio, fake_audio.detach())
            anchor_loss, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
            anchor_loss = anchor_loss * hp.train.get('c_anchor', 0.2)

            if anchor_loss.requires_grad:
                optim_d.zero_grad()
                self.manual_backward(anchor_loss)
                d_norm = torch.nn.utils.clip_grad_norm_(self.net_d.parameters(), hp.train.grad_clip_thresh)
                optim_d.step()
                sched_d.step()

            # Regular training
            batch = batch['train']
        else:
            anchor_loss = None

        _, ppg_interp, ppg_q, ppg_len, f0, pitch_extras, spec, spec_len, spk, wave, sid = \
            self.feats_from_batch(batch)

        # intensity feature
        with torch.no_grad():
            ppg_mask = commons.sequence_mask(ppg_len, max_length=f0.shape[1])
            intensity_feat, attn = self.net_i(ppg_interp, ppg_mask)
            intensity = (intensity_feat * attn).sum(dim=1).unsqueeze(1)

        if random.random() < self.config.train.get('p_pit_extra', 0.5):
            pitch_extras = None # unconditional
        fake_audio, ids_slice, z_mask, \
            (z_f, z_r, z_p, m_p, logs_p, z_q, m_q, 
            logs_q, logdet_f, logdet_r) = self.net_g(
                ppg_q, f0, spec, spk, ppg_len, spec_len, sid=sid,
                intensity=intensity,
                pitch_extras=pitch_extras)

        audio = slice_segments_general(
            wave.unsqueeze(1), ids_slice * hp.data.hop_length, hp.data.segment_size)  # slice

        if self.global_step % 100 == 0 and is_train:
            self.logger.experiment.add_audio(
                tag='train_fake',
                snd_tensor=fake_audio[0].squeeze(1).detach().cpu().float().numpy(),
                global_step=self.global_step,
                sample_rate=self.config.data.sampling_rate
            )
            self.logger.experiment.add_audio(
                tag='train_real',
                snd_tensor=audio[0].squeeze(1).detach().cpu().float().numpy(),
                global_step=self.global_step,
                sample_rate=self.config.data.sampling_rate
            )
        # Sometimes these have slightly different lengths. Should still be aligned
        min_dim = min(fake_audio.shape[2], audio.shape[2])
        fake_audio = fake_audio[:, :, :min_dim]
        audio = audio[:, :, :min_dim]

        # Mel Loss
        mel_fake = self.stft.mel_spectrogram(fake_audio.squeeze(1))
        mel_real = self.stft.mel_spectrogram(audio.squeeze(1))
        mel_loss = F.l1_loss(mel_fake, mel_real) * hp.train.c_mel

        # # audio is of shape [b, 1, t]
        # dump_batched_audio(audio, prefix="gt_", sr=hp.data.sampling_rate)
        # # mels are of shape [b, 100, t]
        # dump_batched_spectrogram(mel_real, prefix="mel_real_")

        # Multi-Resolution STFT Loss
        sc_loss, mag_loss = self.stft_criterion(fake_audio.squeeze(1), audio.squeeze(1))
        stft_loss = (sc_loss + mag_loss) * hp.train.c_stft

        disc_label = self.config.train.get('disc_label', 1.0)
        # Generator Loss
        if self.use_adv:
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(audio, fake_audio)

            # score_loss
            score_loss, _ = generator_loss(y_d_hat_r)
            # feat_loss
            feat_loss = feature_loss(fmap_r, fmap_g)
        else:
            score_loss = torch.tensor(0.0).to(self.device)
            feat_loss = torch.tensor(0.0).to(self.device)

        # Kl Loss
        loss_kl_f = kl_loss(z_f, logs_q, m_p, logs_p, logdet_f, z_mask) * hp.train.c_kl
        loss_kl_r = kl_loss(z_r, logs_p, m_q, logs_q, logdet_r, z_mask) * hp.train.c_kl

        # Loss
        loss_g = score_loss + feat_loss + mel_loss + stft_loss + loss_kl_f + \
            loss_kl_r * 0.5

        if loss_g.requires_grad:
            optim_g.zero_grad()
            self.manual_backward(loss_g)

            g_norm = torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), hp.train.grad_clip_thresh)
            optim_g.step()
            sched_g.step()
        else:
            g_norm = None

        # Discriminator
        loss_d = 0.0
        if self.use_adv:
            y_d_hat_r, y_d_hat_g, _, _ = self.net_d(audio, fake_audio.detach())
            loss_d, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)

            loss_d = loss_d

        else:
            loss_d = torch.tensor(0.0).to(self.device)

        if loss_d.requires_grad:
            optim_d.zero_grad()
            self.manual_backward(loss_d)
            d_norm = torch.nn.utils.clip_grad_norm_(self.net_d.parameters(), hp.train.grad_clip_thresh)
            optim_d.step()
            sched_d.step()
        else:
            d_norm = None
        return {'loss_g': loss_g, 'loss_d': loss_d, 'anchor_loss': anchor_loss, 
                'g_norm': g_norm, 'd_norm': d_norm,
                'score_loss': score_loss, 'feat_loss': feat_loss, 'mel_loss': mel_loss,
                'stft_loss': stft_loss, 'loss_kl_f': loss_kl_f, 'loss_kl_r': loss_kl_r,
                }

    def training_step(self, batch, batch_idx):
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

if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True) # This is needed on Linux

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/char_anchor.yaml')

    parser.add_argument('--svc5_ckpt', type=str, default=None)
    parser.add_argument('--rvc_gen_ckpt', type=str, default=None)
    parser.add_argument('--rvc_disc_ckpt', type=str, default=None)
    parser.add_argument('--codec_ckpt', type=str, default=None)
    parser.add_argument('--int_ckpt', type=str, default=None)

    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--transfer_from', type=str, default=None)

    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    hp = config

    net_g = SynthesizerTrn(
        spec_channels=hp.data.filter_length // 2 + 1,
        segment_size=hp.data.segment_size // hp.data.hop_length,
        hp=hp
    )
    net_d = MultiPeriodDiscriminatorV2(use_spectral_norm=False)
    net_i = IntensityModel(in_channels=hp.codec.whisper_dim)
    codec = VevoRepCodec(
        input_channels=hp.codec.whisper_dim,
        output_channels=hp.codec.whisper_dim,
        encode_channels=hp.codec.whisper_dim,
        decode_channels=hp.codec.whisper_dim,
        code_dim=hp.codec.whisper_dim,
        codebook_num=1,
        codebook_size=hp.codec.codebook_size
    )

    if args.svc5_ckpt is not None:
        print("Loading SVC5 checkpoint: {}".format(args.svc5_ckpt))
        state_dict = torch.load(args.svc5_ckpt, map_location='cpu')['model_g']
        load_state_dict_mismatch(net_g, state_dict)

        assert args.rvc_gen_ckpt is not None
        print("Loading RVC generator checkpoint: {}".format(args.rvc_gen_ckpt))
        state_dict = torch.load(args.rvc_gen_ckpt, map_location='cpu')['model']
        load_submodule_prefix(net_g.dec, 'dec.', state_dict)

        assert args.codec_ckpt is not None
    elif args.resume_from is not None:
        print('Resuming from lightning checkpoint: {}'.format(args.resume_from))
    elif args.transfer_from is not None:
        print('Transferring from lightning checkpoint: {}'.format(args.transfer_from))
        state = torch.load(args.transfer_from, map_location='cpu', weights_only=False)['state_dict']
        load_submodule_prefix(net_g, 'net_g.', state)
        load_submodule_prefix(codec, 'codec.', state) # oops lol
        load_submodule_prefix(net_i, 'net_i.', state) # oops LOL
        load_submodule_prefix(net_d, 'net_d.', state)
    else:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!! No checkpoint file found - starting from scratch !!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    if args.codec_ckpt is not None:
        print("Loading codec checkpoint: {}".format(args.codec_ckpt))
        state_dict = torch.load(args.codec_ckpt, map_location='cpu')['state_dict']
        load_submodule_prefix(codec, 'model.', state_dict)

    if args.int_ckpt is not None:
        print("Loading intensity checkpoint: {}".format(args.int_ckpt))
        state_dict = torch.load(args.int_ckpt, map_location='cpu')['state_dict']
        load_submodule_prefix(net_i, 'model.', state_dict)

    if args.rvc_disc_ckpt is not None:
        print("Loading RVC discriminator checkpoint: {}".format(args.rvc_disc_ckpt))
        state_dict = torch.load(args.rvc_disc_ckpt, map_location='cpu')['model']
        load_state_dict_mismatch(net_d, state_dict)

    training_module = TrainingModule(
        net_g=net_g, net_d=net_d, net_i=net_i, codec=codec, config=config)
    logger = pl.loggers.TensorBoardLogger(
        config.train.get('log_dir', 'logs'), name=config.exp_name,
        version=config.get('tensorboard_version', 0)
    )
    print("Loading data...")
    train_dataset = dataset(config.train.train_filelist, is_test=True)
    anchor_dataset = dataset(config.train.anchor_filelist, is_test=True)

    val_dataset = dataset(config.train.val_filelist, is_test=False)
    print("Creating dataloaders...")
    num_workers = config.train.get('num_workers', 4)
    train_dataloader = train_dataset.loader(
        batch_size=config.train.batch_size, shuffle=True, num_workers=num_workers,
            persistent_workers=num_workers > 0)
    val_dataloader = val_dataset.loader(
        batch_size=config.train.batch_size, shuffle=False, num_workers=num_workers, 
            persistent_workers=num_workers > 0)
    anchor_dataloader = anchor_dataset.loader(
        batch_size=config.train.batch_size, shuffle=False, num_workers=num_workers, 
            persistent_workers=num_workers > 0)
    print("Done")

    val_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'checkpoints/{config.exp_name}',
        filename='best-checkpoint',
        save_top_k=config.train.keep_ckpts,
        mode='min',
        save_last=True
    )
    callbacks = [val_checkpoint_callback]
    if config.train.get('save_interval') is not None:
        interval_checkpoint_callback = pl.callbacks.ModelCheckpoint(
            every_n_epochs=config.train.save_interval,
            dirpath=f'checkpoints/{config.exp_name}',
            filename='interval-checkpoint-{epoch:04d}',
            save_top_k=-1
        )
        callbacks.append(interval_checkpoint_callback)

    trainer = pl.Trainer(
        logger=logger,
        accelerator='gpu',
        precision='bf16-mixed',
        max_steps=config.train.get('max_steps', 160000),
        callbacks=callbacks,
        check_val_every_n_epoch=config.train.get('val_interval', 1),
        #val_check_interval=2,
        log_every_n_steps=config.train.get('log_interval', 50),
    )
    trainer.fit(training_module, CombinedLoader({
        "train": train_dataloader,
        "anchor": anchor_dataloader
    }, mode='max_size_cycle'), val_dataloader, ckpt_path=args.resume_from)
