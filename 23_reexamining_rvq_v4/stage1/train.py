import os
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import sys

from commons import load_submodule_prefix
sys.path.append('..')
from rvq.vevo_repcodec import VevoRepCodec
import ultimate_xc
from dataset import dataset
from omegaconf import OmegaConf

class TrainingModule(pl.LightningModule):
    def __init__(self, model: VevoRepCodec, config: OmegaConf):
        super().__init__()
        self.model = model
        self.config = config
        self._val_indices: list[torch.Tensor] = []

    def step(self, batch, batch_idx, is_test=False):
        whisper = batch['whisper']
        whisper_len = batch['whisper_length']
        yq, y, zq, z, vqloss, perplexity = self.model(whisper)

        # Create a mask based on lengths [B, T, 1]
        B, T, C = whisper.shape
        mask = torch.arange(T, device=whisper.device)[None, :] < whisper_len[:, None]  # [B, T]
        mask = mask.unsqueeze(-1).float()  # [B, T, 1]

        # Apply mask and compute mean over valid positions only
        q_recon_loss = (F.mse_loss(yq, whisper, reduction='none') * mask).sum() / (mask.sum() * C)
        recon_loss = (F.mse_loss(y, whisper, reduction='none') * mask).sum() / (mask.sum() * C)

        loss = q_recon_loss + vqloss * self.config.c_vqloss + recon_loss * self.config.c_reconloss
        od = {
            'loss': loss,
            'q_recon_loss': q_recon_loss,
            'recon_loss': recon_loss,
            'vqloss': vqloss,
            'perplexity': perplexity
        }
        if is_test:
            indices = self.model.forward_index(whisper)
            od['indices'] = indices  # [1, batch, time]
        return od

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

    def training_step(self, batch, batch_idx):
        ret = self.step(batch, batch_idx)
        prog_bar_set = {'loss', 'perplexity'}
        for k, v in ret.items():
            self.log(k, v, prog_bar=k in prog_bar_set, logger=True)
        return ret

    def validation_step(self, batch, batch_idx):
        ret = self.step(batch, batch_idx, is_test=True)
        for k, v in ret.items():
            if k == 'indices':
                continue
            self.log('val_' + k, v, logger=True)

        # Accumulate indices — move to CPU immediately to avoid holding GPU memory
        self._val_indices.append(ret['indices'].detach().cpu())

        return ret['loss']

    def on_validation_epoch_start(self):
        self._val_indices.clear()

    def on_validation_epoch_end(self):
        if not self._val_indices:
            return

        # Concatenate all indices: [1, batch, time] -> flatten to [N]
        all_indices = torch.cat(
            [idx.flatten() for idx in self._val_indices]
        )  # [N]

        codebook_size = self.config.codebook_size

        counts = torch.bincount(all_indices, minlength=codebook_size).float()  # [codebook_size]

        # --- Derived stats ---
        probs = counts / counts.sum()
        entropy = -(probs * (probs + 1e-10).log()).sum()           # codebook entropy (nats)
        active_codes = (counts > 0).sum()                          # codebook utilisation
        most_used_pct = counts.max() / counts.sum() * 100          # dominance of top code

        self.log('val_codebook/entropy',      entropy,      logger=True)
        self.log('val_codebook/active_codes', active_codes.float(), logger=True)
        self.log('val_codebook/most_used_pct', most_used_pct, logger=True)

        # --- Full histogram to TensorBoard / W&B ---
        if self.logger is not None:
            experiment = getattr(self.logger, 'experiment', None)
            if experiment is not None:
                # TensorBoard
                if hasattr(experiment, 'add_histogram'):
                    experiment.add_histogram(
                        'val_codebook/index_hist',
                        all_indices,
                        global_step=self.current_epoch
                    )

        self._val_indices.clear()

if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True) # This is needed on Linux

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--transfer_from', type=str, default=None)

    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    model = VevoRepCodec(
        input_channels=config.whisper_dim,
        output_channels=config.whisper_dim,
        encode_channels=config.hidden_dim,
        decode_channels=config.hidden_dim,
        code_dim=config.code_dim,
        codebook_num=1,
        codebook_size=config.codebook_size
    )

    if os.path.exists(f'checkpoints/{config.exp_name}/last.ckpt'):
        print('Detected interrupted training - resuming from last.ckpt')
        args.resume_from = f'checkpoints/{config.exp_name}/last.ckpt'
    elif args.transfer_from is not None:
            print('Transferring from lightning checkpoint: {}'.format(args.transfer_from))
            state = torch.load(args.transfer_from, map_location='cpu', weights_only=False)['state_dict']
            load_submodule_prefix(model, 'model.', state)
    else:
        if args.resume_from is not None:
            print(f'Resuming from {args.resume_from}')

    training_module = TrainingModule(model, config)
    train_dataset = dataset(config.train_filelist, is_train=True)
    val_dataset = dataset(config.val_filelist, is_train=False)
    train_dataloader = train_dataset.loader(
        batch_size=config.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    val_dataloader = val_dataset.loader(
         batch_size=config.batch_size, shuffle=False, num_workers=4, persistent_workers=True)
    
    val_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'checkpoints/{config.exp_name}',
        filename='best-checkpoint',
        save_top_k=config.keep_ckpts,
        mode='min',
        save_last=True
    )
    callbacks = [val_checkpoint_callback]
    logger = pl.loggers.TensorBoardLogger(
        config.get('log_dir', 'logs'), name=config.exp_name,
        version=0
    )
    trainer = pl.Trainer(
        logger=logger,
        accelerator='gpu',
        precision='bf16-mixed',
        callbacks=callbacks,
        check_val_every_n_epoch=config.get('val_interval', 1),
    )
    trainer.fit(training_module, train_dataloader, val_dataloader, ckpt_path=args.resume_from)
