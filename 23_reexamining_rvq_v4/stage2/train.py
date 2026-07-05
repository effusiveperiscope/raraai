import os
import warnings
import librosa
import logging
warnings.filterwarnings('error', category=RuntimeWarning)
warnings.simplefilter('ignore', category=UserWarning) # pyworld spams the log with messages
logging.getLogger('numba').setLevel(logging.WARNING)

import numpy as np
import torch
from torch.distributions import Beta
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
import random
import ultimate_xc
import math

from modeling.vits.models import SynthesizerTrn
# from svc_helper.svc.rvc.lib.infer_pack.models import MultiPeriodDiscriminatorV2
from modeling.vits_decoder.discriminator import Discriminator
from rvc_losses import generator_loss, feature_loss, discriminator_loss
from vits_extend.stft_loss import STFTLoss
from modeling.vits.losses import kl_loss
from modeling.vits import commons
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

from lightning.pytorch.callbacks import WeightAveraging
from torch.optim.swa_utils import get_ema_avg_fn

class EMAWeightAveraging(WeightAveraging):
    def __init__(self):
        super().__init__(avg_fn=get_ema_avg_fn())

    def should_update(self, step_idx=None, epoch_idx=None):
        # Start after 100 steps.
        return (step_idx is not None) and (step_idx >= 100)

import warnings
import numpy as np
import librosa
import logging
warnings.filterwarnings('error', category=RuntimeWarning)
logging.getLogger('numba').setLevel(logging.WARNING)

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

class TrainingModule(L.LightningModule):
    def __init__(self,
        net_g : SynthesizerTrn, 
        net_d: Discriminator, 
        codec : VevoRepCodec,
        config : OmegaConf):
        super().__init__()
        self.net_g = net_g
        self.net_d = net_d
        self.codec = codec
        del self.codec.decoder
        self.config = config
        self.automatic_optimization = False

        self.use_adv = True # use adversarial losses
        self.spk_index = torch.load(self.config.train.spk_index)

    def setup(self, stage=None):
        hp = self.config
        # Goal is to sample mostly quantized 
        self.alpha_dist = Beta(torch.tensor([1.5]), torch.tensor([1.0]))
        self.stft = TacotronSTFT(filter_length=hp.data.filter_length,
                            hop_length=hp.data.hop_length,
                            win_length=hp.data.win_length,
                            n_mel_channels=hp.data.mel_channels,
                            sampling_rate=hp.data.sampling_rate,
                            mel_fmin=hp.data.mel_fmin,
                            mel_fmax=hp.data.mel_fmax,
                            center=False,
                            device=self.device)
        self.stft_criterion = MultiResolutionSTFTLoss(self.device, eval(hp.mrd.resolutions),
            unvoiced_weight = self.config.train.get('c_unvoiced', 0.8))
        self.spkc_criterion = nn.CosineEmbeddingLoss()
        self.test_dataset = dataset(self.config.train.test_filelist, is_test=True)
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

                ppg_zq, ppg_z = self.codec.forward_encode(ppg_interp)
                ppg_zq = rearrange(ppg_zq, "b c t -> b t c")
                ppg_z = rearrange(ppg_z, "b c t -> b t c")

                f0 = batch['f0'].to(self.dtype).to(self.device)
                ppg_interp = ppg_interp[:,:f0.shape[1],:]
                ppg_zq = ppg_zq[:,:f0.shape[1],:]
                ppg_z = ppg_z[:,:f0.shape[1],:]
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

                # rand. sample alpha
                ppg_alpha = self.alpha_dist.sample().to(ppg_zq.device)
                out_audio = self.net_g.infer(ppg_zq, ppg_z, f0, spk, ppg_len, sid=sid, noise_scale = 
                    self.config.train.get('test_noise_scale', 0.34), pitch_extras=pitch_extras,
                    ppg_alpha=ppg_alpha) 

            for i, audio in enumerate(out_audio):
                audio = audio.squeeze(0).cpu().numpy()
                audio = audio[:int(ppg_len[i] * self.config.data.hop_length)]
                self.logger.experiment.add_audio(
                    tag=f'test_prior_{i}_{j}_{ppg_alpha}',
                    snd_tensor=audio,
                    global_step=self.global_step,
                    sample_rate=self.config.data.sampling_rate
                )
            self.logger.experiment.flush()

    def on_train_start(self):
        self.test()

    def on_train_epoch_start(self):
        self.update_stage()
        if self.current_epoch < self.config.train.get('disc_only', 0):
            for param in self.net_d.parameters():
                param.requires_grad = False
            for param in self.net_g.parameters():
                param.requires_grad = True
        else:
            for param in self.net_d.parameters():
                param.requires_grad = True
            for param in self.net_g.parameters():
                param.requires_grad = True
        if self.config.train.get('freeze_dec', False):
            for param in self.net_g.dec.parameters():
                param.requires_grad = False

    def on_train_epoch_end(self):
        self.test()
        return super().on_train_epoch_end()

    def update_stage(self):
        pass

    def configure_optimizers(self):
        flow = self.net_g.flow
        flow_ids = list(map(id, flow.parameters()))
        base_params = [p for p in self.net_g.parameters() if id(p) not in flow_ids]
        if self.config.train.get('train_codec', False):
            base_params.extend([p for p in self.codec.parameters()])
        flow_params = flow.parameters()

        gen_optim = torch.optim.AdamW(
            params=[
                {'params': self.net_g.parameters(), 'weight_decay': self.config.train.weight_decay},
            ],
            lr=self.config.train.lr, betas=self.config.train.betas)
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

    def step(self, batch, batch_idx, is_train=True):
        ppg = batch['whisper']
        ppg_interp = F.interpolate(rearrange(ppg, 'b t d -> b d t'), scale_factor=2)
        ppg_interp = rearrange(ppg_interp, 'b d t -> b t d')
        ppg_len = batch['whisper_length'] * 2

        with torch.no_grad():
            ppg_zq, ppg_z = self.codec.forward_encode(ppg_interp)
            ppg_zq = rearrange(ppg_zq, "b c t -> b t c")
            ppg_z = rearrange(ppg_z, "b c t -> b t c")

        f0 = batch['f0'].to(ppg_interp.dtype)
        ppg_zq = ppg_zq[:,:f0.shape[1],:]
        ppg_z = ppg_z[:,:f0.shape[1],:]
        ppg = ppg[:,:f0.shape[1],:]
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

        optim_g, optim_d = self.optimizers()
        sched_g, sched_d = self.lr_schedulers()

        hp = self.config
        if random.random() < self.config.train.get('p_pit_extra', 0.5):
            pitch_extras = None # unconditional

        ppg_alpha = self.alpha_dist.sample().to(ppg_zq.device)
        fake_audio, ids_slice, z_mask, \
            (z_f, z_r, z_p, m_p, logs_p, z_q, m_q, 
            logs_q, logdet_f, logdet_r, spk_preds) = self.net_g(
                ppg_zq, ppg_z, f0, spec, spk, ppg_len, spec_len, sid=sid,
                pitch_extras=pitch_extras, ppg_alpha=ppg_alpha) 

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

        # f0 shape: (B, T) — 0 or NaN on unvoiced frames
        f0_sliced = slice_segments_general(f0, ids_slice, hp.data.segment_size // hp.data.hop_length)
        voiced_mask = (f0_sliced > 0).float()  # (B, T), 1=voiced, 0=unvoiced

        # Smooth the mask slightly to avoid hard discontinuities at transitions
        # (optional but helps — unvoiced/voiced boundaries are ambiguous)
        voiced_mask_smooth = F.max_pool1d(
            voiced_mask.unsqueeze(1), kernel_size=5, stride=1, padding=2
        ).squeeze(1)

        # Spk Loss
        if spk_preds is not None and not self.config.train.get('disable_spk', False):
            spk_loss = self.spkc_criterion(spk, spk_preds, torch.Tensor(spk_preds.size(0))
                .to(self.device).fill_(1.0))
        else:
            spk_loss = 0

        # Mel Loss
        mel_fake = self.stft.mel_spectrogram(fake_audio.squeeze(1))
        mel_real = self.stft.mel_spectrogram(audio.squeeze(1))
        mel_loss = weighted_mel_loss(mel_fake, mel_real, voiced_mask_smooth,
            unvoiced_weight = self.config.train.get('c_unvoiced', 0.8)) * hp.train.c_mel

        # # audio is of shape [b, 1, t]
        # dump_batched_audio(audio, prefix="gt_", sr=hp.data.sampling_rate)
        # # mels are of shape [b, 100, t]
        # dump_batched_spectrogram(mel_real, prefix="mel_real_")

        # Multi-Resolution STFT Loss
        sc_loss, mag_loss = self.stft_criterion(
            fake_audio.squeeze(1), audio.squeeze(1), voiced_mask_mel=voiced_mask_smooth)
        stft_loss = (sc_loss + mag_loss) * hp.train.c_stft 

        if self.use_adv:
            # Generator Loss
            disc_fake = self.net_d(fake_audio)
            score_loss = 0.0
            for (_, score_fake) in disc_fake:
                score_loss += torch.mean(torch.pow(score_fake - 1.0, 2))
            score_loss = score_loss / len(disc_fake) * hp.train.get('c_score')

            # Feature Loss
            disc_real = self.net_d(audio)
            feat_loss = 0.0
            for (feat_fake, _), (feat_real, _) in zip(disc_fake, disc_real):
                for fake, real in zip(feat_fake, feat_real):
                    feat_loss += torch.mean(torch.abs(fake - real))
            feat_loss = feat_loss / len(disc_fake)
            feat_loss = feat_loss * 2
        else:
            score_loss = torch.tensor(0.0).to(self.device)
            feat_loss = torch.tensor(0.0).to(self.device)

        # Kl Loss
        loss_kl_f = kl_loss(z_f, logs_q, m_p, logs_p, logdet_f, z_mask) * hp.train.c_kl
        loss_kl_r = kl_loss(z_r, logs_p, m_q, logs_q, logdet_r, z_mask) * hp.train.c_kl

        # Loss
        loss_g = score_loss + feat_loss + mel_loss + stft_loss + loss_kl_f + \
            loss_kl_r * 0.5 + spk_loss * 2

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
            disc_fake = self.net_d(fake_audio.detach())
            disc_real = self.net_d(audio)
            for (_, score_fake), (_, score_real) in zip(disc_fake, disc_real):
                loss_d += torch.mean(torch.pow(score_real - 1.0, 2))
                loss_d += torch.mean(torch.pow(score_fake, 2))
            loss_d = loss_d / len(disc_fake)
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
        return {'loss_g': loss_g, 'loss_d': loss_d, 'g_norm': g_norm, 'd_norm': d_norm,
                'score_loss': score_loss, 'feat_loss': feat_loss, 'mel_loss': mel_loss,
                'stft_loss': stft_loss, 'loss_kl_f': loss_kl_f, 'loss_kl_r': loss_kl_r,
                'spk_loss': spk_loss
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

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base_linux.yaml')

    parser.add_argument('--svc5_ckpt', type=str, default=None)
    parser.add_argument('--rvc_gen_ckpt', type=str, default=None)
    parser.add_argument('--disc_ckpt', type=str, default=None)
    parser.add_argument('--codec_ckpt', type=str, default=None)
    parser.add_argument('--reset_prior', type=int, default=0)

    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--transfer_from', type=str, default=None)
    parser.add_argument('--decoder_from', type=str, default=None)

    return parser.parse_args()


def train(args):
    """Run training. ``args`` may be a dict or any object with attribute access
    (e.g. an argparse.Namespace).  When a plain dict is supplied it is
    converted to a SimpleNamespace so the rest of the function can use
    attribute-style access uniformly.

    Recognised keys / attributes (all optional except ``config``):
        config          – path to the OmegaConf YAML config file (default: 'configs/base.yaml')
        svc5_ckpt       – path to SVC5 generator checkpoint
        rvc_gen_ckpt    – path to RVC generator checkpoint
        disc_ckpt       – path to SVC5 discriminator checkpoint
        codec_ckpt      – path to codec checkpoint
        reset_prior     – number of prior layers to reset (default: 0)
        resume_from     – path to a Lightning checkpoint to resume from
        transfer_from   – path to a Lightning checkpoint to transfer weights from
    """
    from types import SimpleNamespace
    if isinstance(args, dict):
        # Fill in defaults for any keys that were not provided
        defaults = dict(
            config='configs/base.yaml',
            svc5_ckpt=None,
            rvc_gen_ckpt=None,
            disc_ckpt=None,
            codec_ckpt=None,
            reset_prior=0,
            resume_from=None,
            transfer_from=None,
        )
        defaults.update(args)
        args = SimpleNamespace(**defaults)

    config = OmegaConf.load(args.config)
    hp = config

    net_g = SynthesizerTrn(
        spec_channels=hp.data.filter_length // 2 + 1,
        segment_size=hp.data.segment_size // hp.data.hop_length,
        hp=hp
    )
    net_d = Discriminator(hp=hp)
    codec = VevoRepCodec(
        input_channels=hp.codec.whisper_dim,
        output_channels=hp.codec.get('out_dim', hp.codec.whisper_dim),
        encode_channels=hp.codec.hidden_dim,
        decode_channels=hp.codec.hidden_dim,
        code_dim=hp.codec.get('code_dim', hp.codec.whisper_dim),
        codebook_num=1,
        codebook_size=hp.codec.codebook_size
    )

    if os.path.exists(f'checkpoints/{config.exp_name}/last.ckpt'):
        print('Detected interrupted training - resuming from last.ckpt')
        args.resume_from = f'checkpoints/{config.exp_name}/last.ckpt'
    else:
        if args.svc5_ckpt is not None:
            print("Loading SVC5 checkpoint: {}".format(args.svc5_ckpt))
            state_dict = torch.load(args.svc5_ckpt, map_location='cpu')['model_g']
            load_state_dict_mismatch(net_g, state_dict)

            state_dict = torch.load(args.svc5_ckpt, map_location='cpu')['model_d']
            load_state_dict_mismatch(net_d, state_dict)

            if args.codec_ckpt is None:
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('No codec file provided')
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            # assert args.codec_ckpt is not None

        if args.resume_from is not None:
            print('Resuming from lightning checkpoint: {}'.format(args.resume_from))
        elif args.transfer_from is not None:
            print('Transferring from lightning checkpoint: {}'.format(args.transfer_from))
            state = torch.load(args.transfer_from, map_location='cpu', weights_only=False)['state_dict']
            load_submodule_prefix(net_g, 'net_g.', state)
            load_submodule_prefix(codec, 'codec.', state) # oops lol
            load_submodule_prefix(net_d, 'net_d.', state)
        else:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('!!! No checkpoint file found - starting from scratch !!!')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        if args.decoder_from is not None:
            print('Transferring decoder from lightning checkpoint: {}'.format(args.decoder_from))
            state = torch.load(args.decoder_from, map_location='cpu', weights_only=False)['state_dict']
            load_submodule_prefix(net_g.dec, 'net_g.dec.', state)

        if args.codec_ckpt is not None:
            print("Loading codec checkpoint: {}".format(args.codec_ckpt))
            state_dict = torch.load(args.codec_ckpt, map_location='cpu', weights_only=False)['state_dict']
            load_submodule_prefix(codec, 'model.', state_dict)

        if args.disc_ckpt is not None:
            print("Loading SVC5 discriminator checkpoint: {}".format(args.disc_ckpt))
            state_dict = torch.load(args.disc_ckpt, map_location='cpu')['model_d']
            load_state_dict_mismatch(net_d, state_dict)

    if args.reset_prior > 0:
        print(f"Resetting prior {args.reset_prior} layers")
        net_g.enc_p.reset_layers(args.reset_prior)

    training_module = TrainingModule(
        net_g=net_g, net_d=net_d, codec=codec, config=config)
    logger = L.pytorch.loggers.tensorboard.TensorBoardLogger(
        config.train.get('log_dir', 'logs'), name=config.exp_name,
        version=config.get('tensorboard_version', 0)
    )
    print("Loading data...")
    train_dataset = dataset(config.train.train_filelist, is_test=False)
    val_dataset = dataset(config.train.val_filelist, is_test=False)
    print("Creating dataloaders...")
    num_workers = config.train.get('num_workers', 4)
    # num_workers = 0
    train_dataloader = train_dataset.loader(
        batch_size=config.train.batch_size, shuffle=True, num_workers=num_workers,
            persistent_workers=num_workers > 0)
    val_dataloader = val_dataset.loader(
        batch_size=config.train.batch_size, shuffle=False, num_workers=num_workers, 
            persistent_workers=num_workers > 0)
    print("Done")

    val_checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'checkpoints/{config.exp_name}',
        filename='best-checkpoint',
        save_top_k=config.train.keep_ckpts,
        mode='min',
    )
    last_callback = L.pytorch.callbacks.ModelCheckpoint( # just save last
        dirpath=f'checkpoints/{config.exp_name}',
        filename='last',
        save_last=True
    )

    callbacks = [val_checkpoint_callback, last_callback]
    if config.train.get('save_interval') is not None:
        interval_checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
            every_n_epochs=config.train.save_interval,
            dirpath=f'checkpoints/{config.exp_name}',
            filename='interval-checkpoint-{epoch:04d}', save_top_k=-1
        )
        callbacks.append(interval_checkpoint_callback)

    if config.train.get('use_ema'):
        callbacks.append(EMAWeightAveraging())

    trainer = L.Trainer(
        logger=logger,
        accelerator='gpu',
        precision='bf16-mixed',
        max_steps=config.train.get('max_steps', 160000),
        max_epochs=10000000,
        callbacks=callbacks,
        check_val_every_n_epoch=config.train.get('val_interval', 1),
        #val_check_interval=2,
        log_every_n_steps=config.train.get('log_interval', 50),
    )
    trainer.fit(training_module, train_dataloader, val_dataloader, ckpt_path=args.resume_from, weights_only=False)


if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True) # This is needed on Linux

    train(parse_args())
