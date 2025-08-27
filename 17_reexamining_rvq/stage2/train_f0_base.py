import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from commons import load_submodule_prefix, matplotlib_to_tensorboard
from vits_extend.stft_loss import STFTLoss
import matplotlib.pyplot as plt
import ultimate_xc

from modeling.vits.models import F0Predictor2
from modeling.rvc.f0_predictor import F0Discriminator
from svc_helper.pitch.utils import nonzero_mean, discretize_f0_log
from modeling.vits import commons
from dataset import dataset

# Train base F0 predictor model
class TrainingModule(pl.LightningModule):
    def __init__(self,
        net_g: F0Predictor2,
        net_d: F0Discriminator,
        config: OmegaConf
    ):
        super().__init__()
        self.net_g = net_g
        self.net_d = net_d
        self.config = config
        self.automatic_optimization = False

    # def on_train_start(self):
    #     self.reset_disc_weights()

    def setup(self, stage=None):
        self.f0_stft = STFTLoss(self.device, 
            fft_size=512, shift_size=50, win_length=200, window="hann_window")

    def reset_disc_weights(self):
        def reset_weights(layer):
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.net_d.apply(reset_weights)

    def step(self, batch, batch_idx, is_train=True):
        f0 = batch['f0'].to(self.dtype)
        np_pit = f0.detach().cpu().numpy()
        ppg = batch['whisper']
        ppg_len = batch['whisper_length']
        # Iterate over batch and apply functions individually
        target_f0_mean = []
        quant_pitch = []

        optim_g, optim_d = self.optimizers()

        for pit_sample in np_pit:
            # These functions expect a 1D array
            target_f0_mean.append(nonzero_mean(pit_sample))
            quant_pitch.append(
                discretize_f0_log(
                    f0=pit_sample,
                    n_voiced_bins=self.config.vits.get('pitch_quant_dim', 8),
                    hold_length=10
                )
            )

        # Convert to numpy/tensor
        target_f0_mean = torch.from_numpy(np.array(target_f0_mean)).float().to(self.device)  # shape: (B,)
        quant_pitch = torch.from_numpy(np.stack(quant_pitch)).long().to(self.device)  # shape: (B, T)

        mask = commons.sequence_mask(ppg_len, quant_pitch.size(1))

        target = torch.log(f0 + 1)
        f0_pred = self.net_g(quant_pitch, target_f0_mean, ppg, mask).squeeze(-1)
        
        vuv_mask = (quant_pitch != 0).unsqueeze(-1)

        # Don't calculate discriminator loss on silent frames
        f0_fake_disc = self.net_d(f0_pred.unsqueeze(-1))[vuv_mask]
        disc_label = self.config.train.get('disc_label', 1.0)

        f0_gen_loss = torch.mean((disc_label - f0_fake_disc) ** 2)
        l1_loss = F.l1_loss(f0_pred, target)
        sc_loss, mag_loss = self.f0_stft(f0_pred.squeeze(-1), target.squeeze(-1))
        f0_recon_loss = l1_loss + sc_loss + mag_loss
        # f0_recon_loss = 
        gen_loss = f0_gen_loss + f0_recon_loss * self.config.train.get('c_recon', 1.0)

        if gen_loss.requires_grad and not gen_loss.isnan().any() and not gen_loss.isinf().any():
            optim_g.zero_grad()
            self.manual_backward(gen_loss)
            g_norm = torch.nn.utils.clip_grad_norm_(
                self.net_g.parameters(), self.config.train.grad_clip_thresh)
            optim_g.step()
        else:
            if gen_loss.isnan().any():
                print("gen_loss is nan")
            else:
                print("gen_loss is inf")
            g_norm = None

        f0_fake_disc = self.net_d(f0_pred.unsqueeze(-1).detach())[vuv_mask]
        f0_real_disc = self.net_d(target.unsqueeze(-1))[vuv_mask]
        f0_real_loss = torch.mean((disc_label - f0_real_disc) ** 2)
        f0_fake_loss = torch.mean(f0_fake_disc ** 2)
        disc_loss = f0_real_loss + f0_fake_loss

        if disc_loss.requires_grad and not disc_loss.isnan().any() and not disc_loss.isinf().any():
            optim_d.zero_grad()
            self.manual_backward(disc_loss)
            d_norm = torch.nn.utils.clip_grad_norm_(
                self.net_d.parameters(), self.config.train.grad_clip_thresh)
            optim_d.step()
        else:
            if disc_loss.isnan().any():
                print("disc_loss is nan")
            else: 
                print("disc_loss is inf")
            d_norm = None

        if is_train and self.global_step % 100 == 0:
            fig = self.plot_f0_curves(
                target.detach().cpu().numpy(), 
                f0_pred.detach().cpu().numpy())
            self.logger.experiment.add_image(
                'f0_curves', fig, global_step=self.global_step)
            plt.close(fig)

        return {
            'loss': gen_loss + disc_loss,
            'gen_loss': gen_loss,
            'disc_loss': disc_loss,
            'recon_loss': f0_recon_loss,
            'g_norm': g_norm,
            'd_norm': d_norm
        }

    def plot_f0_curves(self, real : np.ndarray, fake : np.ndarray):
        fig = plt.figure(figsize=(6, 8))
        ax = fig.add_subplot(111)
        ax.plot(real[0], label='real')
        ax.plot(fake[0], label='fake')
        ax.legend()
        return matplotlib_to_tensorboard(fig)

    def training_step(self, batch, batch_idx):
        ret = self.step(batch, batch_idx, is_train=True)
        if ret is None:
            return None
        
        prog_bar_set = {'gen_loss', 'disc_loss'}
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

        return ret['loss']

    def configure_optimizers(self):
        gen_optim = torch.optim.AdamW(
            self.net_g.parameters(), 
                lr=self.config.train.lr * self.config.train.get('lr_coef_gen', 1),
                betas=self.config.train.betas,
            weight_decay=self.config.train.weight_decay)
        disc_optim = torch.optim.AdamW(
            self.net_d.parameters(), 
            lr=self.config.train.lr * self.config.train.get('lr_coef_disc', 1),
            betas=self.config.train.betas,
            weight_decay=self.config.train.weight_decay)
        return [gen_optim, disc_optim], []

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/f0_base.yaml')
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--transfer_from', type=str, default=None)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    hp = config

    net_g = F0Predictor2(
        speech_dim=hp.codec.whisper_dim,
        hidden_dim=hp.vits.hidden_channels
    )
    net_d = F0Discriminator()
    training_module = TrainingModule(net_g, net_d, config)

    if args.transfer_from is not None:
        print('Transferring from lightning checkpoint: {}'.format(args.transfer_from))
        state = torch.load(args.transfer_from, map_location='cpu', weights_only=False)['state_dict']
        load_submodule_prefix(net_g, 'net_g.', state)
        load_submodule_prefix(net_d, 'net_d.', state)

    logger = pl.loggers.TensorBoardLogger(
        config.train.get('log_dir', 'logs/f0/'), name=config.exp_name,
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
        dirpath=f'checkpoints/f0/{config.exp_name}',
        filename='best-checkpoint',
        save_top_k=config.train.keep_ckpts,
        mode='min',
        save_last=True
    )
    callbacks = [val_checkpoint_callback]
    if config.train.get('save_interval') is not None:
        interval_checkpoint_callback = pl.callbacks.ModelCheckpoint(
            every_n_epochs=config.train.save_interval,
            dirpath=f'checkpoints/f0/{config.exp_name}',
            filename='interval-checkpoint-{epoch:04d}',
            save_top_k=-1
        )
        callbacks.append(interval_checkpoint_callback)

    trainer = pl.Trainer(
        logger=logger,
        accelerator='gpu',
        precision='bf16-mixed',
        max_steps=config.train.get('max_steps', -1),
        callbacks=callbacks,
        check_val_every_n_epoch=config.train.get('val_interval', 1),
        # detect_anomaly=True
    )
    trainer.fit(training_module, train_dataloader, val_dataloader, ckpt_path=args.resume_from)