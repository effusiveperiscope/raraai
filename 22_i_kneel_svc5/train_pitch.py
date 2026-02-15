import pytorch_lightning as pl
from commons import load_state_dict_mismatch, load_submodule_prefix, slice_segments_general
from modeling.pitch_predictor import (
    PitchPredictorV0, get_masked_mean, get_masked_std)
from omegaconf import OmegaConf
from dataset import dataset_f0
import torch

class TrainModule(pl.LightningModule):
    def __init__(self,
        net: PitchPredictorV0,
        config: OmegaConf):
        super().__init__()

        self.net = net
        self.config = config

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.net_g.parameters(),
            lr=self.config.train.lr,
            betas=self.config.train.betas,
            eps=self.config.train.eps
        )
        return [optim], []

    def step(self, batch):
        f0 = batch['f0']

        x_mask = (f0 != 0)
        x_1 = f0
        t = torch.rand(x_1.shape[0], device=self.device).unsqueeze(-1)

        loss = self.net.compute_loss(f0=x_1, t=t, x_mask=x_mask)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch)

    def validation_step(self, batch, batch_idx):
        return self.step(batch)

if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True) # This is needed on Linux

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--transfer_from', type=str, default=None)

    net = PitchPredictorV0()

    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    resume_from = args.resume_from
    transfer_from = args.transfer_from

    if resume_from is not None:
        print('Resuming from lightning checkpoint: {}'.format(resume_from))
        d = torch.load(resume_from, map_location='cpu', weights_only=False)
        if d['global_step'] >= config.train.get('max_steps', 3000000):
            print('Maximum steps reached. Exiting.')
    elif transfer_from is not None:
        print('Transferring from lightning checkpoint: {}'.format(transfer_from))
        state = torch.load(transfer_from, map_location='cpu', weights_only=False)['state_dict']
        load_submodule_prefix(net, 'net.', state)
    else:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!! No checkpoint file found - starting from scratch !!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    
    val_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'checkpoints/{config.exp_name}',
        filename='best-checkpoint',
        save_top_k=config.train.keep_ckpts,
        mode='min',
    )
    training_module = TrainModule(net, config)
    logger = pl.loggers.TensorBoardLogger(
        config.train.get('log_dir', 'logs'), name=config.exp_name,
        version=config.get('tensorboard_version', 0)
    )
    trainer = pl.Trainer(
        logger=logger,
        accelerator='gpu',
        precision='bf16-mixed',
        max_steps=config.train.get('max_steps', 3000000),
        callbacks=[val_checkpoint_callback],
        check_val_every_n_epoch=config.train.get('val_interval', 1),
        #val_check_interval=2,
        log_every_n_steps=config.train.get('log_interval', 50),
    )
    print("Loading data...")
    train_dataset = dataset_f0(config.train.train_filelist, is_train=True)
    val_dataset = dataset_f0(config.train.val_filelist, is_train=False)
    print("Creating dataloaders...")
    num_workers = config.train.get('num_workers', 4)
    train_dataloader = train_dataset.loader(
        batch_size=config.train.batch_size, shuffle=True, num_workers=num_workers,
            persistent_workers=num_workers > 0)
    val_dataloader = val_dataset.loader(
        batch_size=config.train.batch_size, shuffle=False, num_workers=num_workers, 
            persistent_workers=num_workers > 0)
    trainer.fit(training_module, train_dataloader, val_dataloader, ckpt_path=resume_from)
