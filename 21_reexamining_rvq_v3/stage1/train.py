import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import sys
sys.path.append('..')
from rvq.vevo_repcodec import VevoRepCodec
import ultimate_xc
from dataset import dataset
from omegaconf import OmegaConf
from features import PedalboardContext, WhisperContext
from commons import load_submodule_prefix

def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

class TrainingModule(pl.LightningModule):
    def __init__(self, model : VevoRepCodec, config : OmegaConf):
        super().__init__()
        self.model = model
        self.config = config
        self.pedalboard_context = PedalboardContext()
        self.whisper_context = WhisperContext()

    def step(self, batch, batch_idx):
        wave = batch['wave']
        wave_lengths = batch['wave_length']

        wave_np = wave.detach().cpu().numpy()
        waves_processed = []
        for i, wave in enumerate(wave_np):
            waves_processed.append(self.pedalboard_context.process_wave(
                wave[:wave_lengths[i]]))
        whisper_features, feature_len = self.whisper_context.extract_features_batched(
            waves_processed)
        whisper_features = whisper_features.to(self.dtype).to(self.device)[
            :, :feature_len.max().item() // 2, :]

        y, zq, z, vqloss, perplexity = self.model(whisper_features)
        recon_loss = F.mse_loss(y, whisper_features)
        loss = recon_loss + vqloss * self.config.c_vqloss

        return { 
            'loss' : loss,
            'recon_loss' : recon_loss,
            'vqloss' : vqloss,
            'perplexity' : perplexity
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

    def training_step(self, batch, batch_idx):
        ret = self.step(batch, batch_idx)
        prog_bar_set = {'loss', 'perplexity'}
        for k,v in ret.items():
            self.log(k, v, prog_bar=k in prog_bar_set, logger=True)
        return ret

    def validation_step(self, batch, batch_idx):
        ret = self.step(batch, batch_idx)
        for k,v in ret.items():
            self.log('val_' + k, v, logger=True)
        val_loss = ret['loss']
        return val_loss

if __name__ == '__main__':
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
        encode_channels=config.whisper_dim,
        decode_channels=config.whisper_dim,
        code_dim=config.whisper_dim,
        codebook_num=1,
        codebook_size=config.codebook_size
    )

    if args.resume_from is not None:
        print(f'Resuming from {args.resume_from}')
    if args.transfer_from is not None:
        print("Transfer from checkpoint: {}".format(args.transfer_from))
        state_dict = torch.load(args.transfer_from, map_location='cpu')['state_dict']
        load_submodule_prefix(model, 'model.', state_dict)

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