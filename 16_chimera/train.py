import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
import random
import ultimate_xc

from modeling.vits.models import SynthesizerTrn
from modeling.vits.losses import kl_loss
from modeling.vits import commons
from vits_extend.stft import TacotronSTFT
from vits_extend.stft_loss import MultiResolutionSTFTLoss
from svc_helper.svc.rvc.lib.infer_pack.models import MultiPeriodDiscriminatorV2
from rvc_losses import generator_loss, discriminator_loss, feature_loss
from dataset import dataset
from commons import load_state_dict_mismatch, load_submodule_prefix, slice_segments_general
from utils import dump_batched_audio, dump_batched_spectrogram

class TrainingModule(pl.LightningModule):
    def __init__(self,
        net_g : SynthesizerTrn, 
        net_d: MultiPeriodDiscriminatorV2, 
        config : OmegaConf):
        super().__init__()
        self.net_g = net_g
        self.net_d = net_d
        self.config = config
        self.automatic_optimization = False

        self.stage = 1
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
        self.spkc_criterion = nn.CosineEmbeddingLoss()
        self.test_dataset = dataset(self.config.train.test_filelist, is_train=False)
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
                vec = batch['hubert'].to(self.dtype).to(self.device)
                pit = batch['f0'].to(self.dtype).to(self.device)
                pit = pit * (2 ** (self.config.train.get('test_transpose', 0) / 12))
                spk = self.spk_index['0'].to(self.dtype).to(self.device).unsqueeze(0)
                spec = batch['spec'].to(self.dtype).to(self.device).transpose(1,2)
                ppg_len = batch['whisper_length']
                out_audio = self.net_g.infer(ppg, vec, pit, spk, ppg_len, noise_scale = 
                    self.config.train.get('test_noise_scale', 0.34))
                out_audio_post = self.net_g.posterior_test(spec, ppg_len, pit, spk)
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

    def on_train_epoch_end(self):
        self.test()
        return super().on_train_epoch_end()

    def update_stage(self):
        if self.current_epoch >= self.config.train.stage2_epoch:
            self.stage = 2
            self.use_adv = True
        else:
            self.stage = 1
            self.use_adv = False # don't use adversarial training for stage 1
            # (we only rely on reconstruction/kl losses at this point)

    def on_train_batch_start(self, batch, batch_idx):
        for param in self.net_d.parameters():
            param.requires_grad = True
        for param in self.net_g.parameters():
            param.requires_grad = True

        # for param in self.net_d.parameters():
        #     param.requires_grad = True
        # for param in self.net_g.parameters():
        #     param.requires_grad = False

        # for param in self.net_g.enc_q.parameters():
        #     param.requires_grad = True
        # for param in self.net_g.dec.parameters():
        #     param.requires_grad = True
        # for param in self.net_g.dec.adapter.parameters():
        #     param.requires_grad = False

    def configure_optimizers(self):
        gen_optim = torch.optim.AdamW(
            self.net_g.parameters(), lr=self.config.train.lr, betas=self.config.train.betas,
            weight_decay=self.config.train.weight_decay)
        disc_optim = torch.optim.AdamW(
            self.net_d.parameters(), lr=self.config.train.lr, betas=self.config.train.betas,
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
        ppg_len = batch['whisper_length']
        vec = batch['hubert']
        pit = batch['f0'].to(ppg.dtype)
        spec = batch['spec']
        spec = rearrange(spec, "b t c -> b c t") # channel first is expected for some reason
        spec_len = batch['spec_length']
        spk = batch['spk']
        wave = batch['wave']

        optim_g, optim_d = self.optimizers()
        sched_g, sched_d = self.lr_schedulers()

        # augmentation is baked into the model for so-vits-svc 5.0, 
        # so we don't perform any ourselves
        hp = self.config
        fake_audio, ids_slice, z_mask, \
            (z_f, z_r, z_p, m_p, logs_p, z_q, m_q, 
            logs_q, logdet_f, logdet_r), spk_preds = self.net_g(
                ppg, vec, pit, spec, spk, ppg_len, spec_len)

        audio = slice_segments_general(
            wave.unsqueeze(1), ids_slice * hp.data.hop_length, hp.data.segment_size)  # slice
        # Spk Loss
        spk_loss = self.spkc_criterion(spk, spk_preds, torch.Tensor(spk_preds.size(0))
                            .to(self.device).fill_(1.0))
        
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
        # exit(0)

        # Multi-Resolution STFT Loss
        sc_loss, mag_loss = self.stft_criterion(fake_audio.squeeze(1), audio.squeeze(1))
        stft_loss = (sc_loss + mag_loss) * hp.train.c_stft

        # Generator Loss
        if self.use_adv:
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(audio, fake_audio)
            score_loss, _ = generator_loss(y_d_hat_g)
            # # Feature Loss
            feat_loss = feature_loss(fmap_r, fmap_g)
        else:
            score_loss = torch.tensor(0.0).to(self.device)
            feat_loss = torch.tensor(0.0).to(self.device)

        # Kl Loss
        loss_kl_f = kl_loss(z_f, logs_q, m_p, logs_p, logdet_f, z_mask) * hp.train.c_kl
        loss_kl_r = kl_loss(z_r, logs_p, m_q, logs_q, logdet_r, z_mask) * hp.train.c_kl

        # Loss
        loss_g = score_loss + feat_loss + mel_loss + stft_loss + loss_kl_f + loss_kl_r * 0.5 + spk_loss * 2

        if loss_g.requires_grad:
            optim_g.zero_grad()
            self.manual_backward(loss_g)

            g_norm = torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), hp.train.grad_clip_thresh)
            optim_g.step()
            sched_g.step()
        else:
            g_norm = None

        if self.use_adv:
            y_d_hat_r, y_d_hat_g, _, _ = self.net_d(audio, fake_audio.detach())
            loss_d, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
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
                'spk_loss': spk_loss}

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/svc5_base.yaml')

    parser.add_argument('--svc5_ckpt', type=str, default=None)
    parser.add_argument('--rvc_gen_ckpt', type=str, default=None)
    parser.add_argument('--rvc_disc_ckpt', type=str, default=None) # RVC D_ checkpoint

    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--transfer_from', type=str, default=None)
    parser.add_argument('--version', type=int, default=None, help='tensorboard log version')

    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    hp = config

    net_g = SynthesizerTrn(
        spec_channels=hp.data.filter_length // 2 + 1,
        segment_size=hp.data.segment_size // hp.data.hop_length,
        hp=hp
    )
    net_d = MultiPeriodDiscriminatorV2(use_spectral_norm=False)

    if args.svc5_ckpt is not None:
        print("Loading SVC5 checkpoint: {}".format(args.svc5_ckpt))
        state_dict = torch.load(args.svc5_ckpt, map_location='cpu')['model_g']
        load_state_dict_mismatch(net_g, state_dict)
        assert args.rvc_gen_ckpt is not None
        print("Loading RVC generator checkpoint: {}".format(args.rvc_gen_ckpt))
        state_dict = torch.load(args.rvc_gen_ckpt, map_location='cpu')['model']
        load_submodule_prefix(net_g.dec, 'dec.', state_dict)
        assert args.rvc_disc_ckpt is not None
        print("Loading RVC discriminator checkpoint: {}".format(args.rvc_disc_ckpt))
        state_dict = torch.load(args.rvc_disc_ckpt, map_location='cpu')['model']
        load_state_dict_mismatch(net_d, state_dict)
    elif args.resume_from is not None:
        print('Resuming from lightning checkpoint: {}'.format(args.resume_from))
    elif args.transfer_from is not None:
        print('Transferring from lightning checkpoint: {}'.format(args.transfer_from))
        state = torch.load(args.transfer_from, map_location='cpu', weights_only=False)['state_dict']
        load_submodule_prefix(net_g, 'net_g.', state)
        load_submodule_prefix(net_d, 'net_d.', state)
    else:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!! No checkpoint file found - starting from scratch !!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    training_module = TrainingModule(net_g=net_g, net_d=net_d, config=config)
    logger = pl.loggers.TensorBoardLogger(
        config.train.get('log_dir', 'logs'), name=config.exp_name,
        version=args.version
    )
    print("Loading data...")
    train_dataset = dataset(config.train.train_filelist, is_train=True)
    val_dataset = dataset(config.train.val_filelist, is_train=False)
    print("Creating dataloaders...")
    train_dataloader = train_dataset.loader(
        batch_size=config.train.batch_size, shuffle=True)
    val_dataloader = val_dataset.loader(
        batch_size=config.train.batch_size)
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
    )
    trainer.fit(training_module, train_dataloader, val_dataloader, ckpt_path=args.resume_from)