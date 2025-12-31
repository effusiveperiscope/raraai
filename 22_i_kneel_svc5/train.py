import itertools
import random
from einops import rearrange
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
from modeling.vits import commons
from modeling.vits.losses import kl_loss
from modeling.vits.models import SynthesizerTrn
from modeling.vits_decoder.discriminator import Discriminator
from dataset import dataset

from vits_extend.stft_loss import STFTLoss
from vits_extend.stft import TacotronSTFT
from vits_extend.stft_loss import MultiResolutionSTFTLoss
import torch.nn as nn
import torch.nn.functional as F
from commons import load_state_dict_mismatch, load_submodule_prefix, slice_segments_general
import ultimate_xc

class TrainModule(pl.LightningModule):
    def __init__(self,
        net_g : SynthesizerTrn,
        net_d : Discriminator,
        config : OmegaConf):
        super().__init__()
        self.automatic_optimization = False

        self.net_g = net_g
        self.net_d = net_d
        self.config = config

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
        self.test_dataset = dataset(self.config.train.test_filelist, is_train=False)
        self.test_dataloader = self.test_dataset.loader()
        self.spk_index = torch.load(self.config.train.spk_index)

    def configure_optimizers(self):
        gen_optim = torch.optim.AdamW(
            self.net_g.parameters(), 
            lr=self.config.train.lr,
            betas=self.config.train.betas,
            eps=self.config.train.eps)
        disc_optim = torch.optim.AdamW(
            self.net_d.parameters(),
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
        return [gen_optim, disc_optim], [gen_scheduler, disc_scheduler]

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

        for j,batch in enumerate(self.test_dataloader):
            with torch.no_grad():
                for k, x in batch.items():
                    if k in {'whisper', 'hubert', 'f0',
                        'f0_inharm', 'f0_subharm', 'f0_confidence', 'spec', 'spk'}:
                        batch[k] = x.to(self.dtype).to(self.device)
                ppg = batch['whisper']
                ppg_len = batch['whisper_length']
                vec = batch['hubert']
                pit = batch['f0']
                pitch_extras = {
                    'confidence': batch['f0_confidence'],
                    'subharmonic': batch['f0_subharm'],
                    'inharmonic': batch['f0_inharm']
                }
                spec = batch['spec']
                spec = rearrange(spec, "b t c -> b c t") # channel first is expected for some reason
                spk = batch['spk']

                pit = pit * (2 ** (self.config.train.get('test_transpose', 0) / 12))
                key = self.config.train.get('test_sid', 0)
                if not key in self.spk_index:
                    key = str(key)
                spk = self.spk_index[key].to(self.dtype).to(self.device).unsqueeze(0)

                out_audio = self.net_g.infer(ppg, vec, pit, spk, ppg_len, pitch_extras=pitch_extras)

            for i, audio in enumerate(out_audio):
                audio = audio.squeeze(0).cpu().numpy()
                audio = audio[:int(ppg_len[i] * self.config.data.hop_length)]
                self.logger.experiment.add_audio(
                    tag=f'test_prior_{i}_{j}',
                    snd_tensor=audio,
                    global_step=self.global_step,
                    sample_rate=self.config.data.sampling_rate
                )
            self.logger.experiment.flush()

    def step(self, batch, batch_idx, is_train=True):
        for k, x in batch.items():
            if k in {'whisper', 'hubert', 'f0',
                'f0_inharm', 'f0_subharm', 'f0_confidence', 'spec', 'spk'}:
                batch[k] = x.to(self.dtype).to(self.device)
        ppg = batch['whisper']
        ppg_len = batch['whisper_length']
        vec = batch['hubert']
        pit = batch['f0']
        pitch_extras = {
            'confidence': batch['f0_confidence'],
            'subharmonic': batch['f0_subharm'],
            'inharmonic': batch['f0_inharm']
        }
        spec = batch['spec']
        spec = rearrange(spec, "b t c -> b c t") # channel first is expected for some reason
        spec_len = batch['spec_length']
        spk = batch['spk']
        audio = batch['wave']

        optim_g, optim_d = self.optimizers()
        sched_g, sched_d = self.lr_schedulers()

        hp = self.config
        if random.random() < self.config.train.get('p_pit_extra', 0.5):
            pitch_extras = None # unconditional

        # generator
        fake_audio, ids_slice, z_mask, \
            (z_f, z_r, z_p, m_p, logs_p, z_q, m_q, logs_q, logdet_f, logdet_r), spk_preds = self.net_g(
                ppg, vec, pit, spec, spk, ppg_len, spec_len, pitch_extras=pitch_extras)
        audio = commons.slice_segments(
            audio.unsqueeze(1), ids_slice * hp.data.hop_length, hp.data.segment_size)  # slice
        # Spk Loss
        spk_loss = self.spkc_criterion(spk, spk_preds, torch.Tensor(spk_preds.size(0))
            .to(self.device).fill_(1.0))

        # Mel Loss
        mel_fake = self.stft.mel_spectrogram(fake_audio.squeeze(1))
        mel_real = self.stft.mel_spectrogram(audio.squeeze(1))
        mel_loss = F.l1_loss(mel_fake, mel_real) * hp.train.c_mel

        # multi-resolution stft loss
        sc_loss, mag_loss = self.stft_criterion(fake_audio.squeeze(1), audio.squeeze(1))
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

        # Kl Loss
        loss_kl_f = kl_loss(z_f, logs_q, m_p, logs_p, logdet_f, z_mask) * hp.train.c_kl
        loss_kl_r = kl_loss(z_r, logs_p, m_q, logs_q, logdet_r, z_mask) * hp.train.c_kl

        # Loss
        loss_g = score_loss + feat_loss + mel_loss + stft_loss + loss_kl_f + loss_kl_r * 0.5 + spk_loss * 2

        if loss_g.requires_grad:
            optim_g.zero_grad()
            self.manual_backward(loss_g)
            optim_g.step()
            sched_g.step()

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

        return {'loss_g': loss_g, 'loss_d': loss_d, 'spk_loss': spk_loss, 
                'mel_loss': mel_loss, 'stft_loss': stft_loss, 'score_loss': score_loss,
                  'feat_loss': feat_loss, 'loss_kl_f': loss_kl_f, 'loss_kl_r': loss_kl_r}

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
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    parser.add_argument('--svc5_ckpt', type=str, default=None)
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
    net_d = Discriminator(hp)

    if args.svc5_ckpt is not None:
        print("Loading SVC5 checkpoint: {}".format(args.svc5_ckpt))
        state_dict = torch.load(args.svc5_ckpt, map_location='cpu')['model_g']
        load_state_dict_mismatch(net_g, state_dict)
        state_dict = torch.load(args.svc5_ckpt, map_location='cpu')['model_d']
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

    training_module = TrainModule(
        net_g=net_g, net_d=net_d, config=config)
    logger = pl.loggers.TensorBoardLogger(
        config.train.get('log_dir', 'logs'), name=config.exp_name,
        version=config.get('tensorboard_version', 0)
    )
    print("Loading data...")
    train_dataset = dataset(config.train.train_filelist, is_train=True)
    val_dataset = dataset(config.train.val_filelist, is_train=False)
    print("Creating dataloaders...")
    num_workers = config.train.get('num_workers', 4)
    train_dataloader = train_dataset.loader(
        batch_size=config.train.batch_size, shuffle=True, num_workers=num_workers,
            persistent_workers=num_workers > 0)
    val_dataloader = val_dataset.loader(
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
        max_steps=config.train.get('max_steps', 3000000),
        callbacks=callbacks,
        check_val_every_n_epoch=config.train.get('val_interval', 1),
        #val_check_interval=2,
        log_every_n_steps=config.train.get('log_interval', 50),
    )
    trainer.fit(training_module, train_dataloader, val_dataloader, ckpt_path=args.resume_from)