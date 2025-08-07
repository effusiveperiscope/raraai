from omegaconf import OmegaConf
import torch
from commons import load_submodule_prefix
from gather2 import process_criteria, Criterion
from modeling.vits.models import SynthesizerTrn
from svc_helper.svc.rvc.lib.infer_pack.models import MultiPeriodDiscriminatorV2
from preprocess import process_filelist
import pytorch_lightning as pl
from train import TrainingModule
from dataset import dataset
import os

PPP_FILELIST = 'filelists/ppp_filelist.txt'
PPP_SING_FILELIST = 'filelists/ppp_sing_filelist.txt'
BASE_CHECKPOINT = 'checkpoints/base.ckpt'
BASE_CONFIG = 'config/fs.yaml'
sources = [
    ('Applejack_Sing', PPP_SING_FILELIST),
    # ('Fluttershy_Sing', PPP_SING_FILELIST), # Done
    ('Pinkie_Sing', PPP_SING_FILELIST),
    ('Rarity_Sing', PPP_SING_FILELIST),
    ('Twilight_Sing', PPP_SING_FILELIST),
    ('Rainbow_Sing', PPP_SING_FILELIST),
    ('Applejack', PPP_FILELIST),
    ('Fluttershy', PPP_FILELIST),
    # ('Pinkie', PPP_FILELIST), # Done
    ('Rarity', PPP_FILELIST),
    ('Twilight', PPP_FILELIST),
    ('Rainbow', PPP_FILELIST),
]

for char, src_filelist in sources:
    out_filelist = os.path.join('filelists', f'{char}.txt')

    # 1. Collect filelist using gather2.py
    process_criteria(
        [
            Criterion(char=char, filelist=src_filelist,
            excl_terms=['_Very Noisy_', 'CAUTION']),
        ], out_file=out_filelist
    )

    # 2. Preprocess filelist using preprocess.py
    data_dir = f'data/{char}'
    process_filelist(filelist_path=out_filelist, 
        output_dir=data_dir)

    # 3. Setup config
    hp = OmegaConf.load(BASE_CONFIG)
    hp.exp_name = f'{char}'
    hp.train_filelist = os.path.join(data_dir, 'train.txt')
    hp.val_filelist = os.path.join(data_dir, 'val.txt')
    hp.spk_index = os.path.join(data_dir, 'sid_avgs.pt')

    # Train
    net_g = SynthesizerTrn(
        spec_channels=hp.data.filter_length // 2 + 1,
        segment_size=hp.data.segment_size // hp.data.hop_length,
        hp=hp
    )
    net_d = MultiPeriodDiscriminatorV2(use_spectral_norm=False)
    training_module = TrainingModule(net_g=net_g, net_d=net_d, config=hp)
    state = torch.load(
        BASE_CHECKPOINT, map_location='cpu', weights_only=False)['state_dict']
    load_submodule_prefix(net_g, 'net_g.', state)
    load_submodule_prefix(net_d, 'net_d.', state)
    logger = pl.loggers.TensorBoardLogger(
        hp.train.get('log_dir', 'logs'), name=hp.exp_name,
        version=0
    )
    train_dataset = dataset(hp.train.train_filelist, is_train=True)
    val_dataset = dataset(hp.train.val_filelist, is_train=False)
    train_dataloader = train_dataset.loader(
        batch_size=hp.train.batch_size, shuffle=True)
    val_dataloader = val_dataset.loader(
        batch_size=hp.train.batch_size)
    val_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'checkpoints/{hp.exp_name}',
        filename='best-checkpoint',
        save_top_k=hp.train.keep_ckpts,
        mode='min',
        save_last=True
    )
    callbacks = [val_checkpoint_callback]
    if hp.train.get('save_interval') is not None:
        interval_checkpoint_callback = pl.callbacks.ModelCheckpoint(
            every_n_epochs=hp.train.save_interval,
            dirpath=f'checkpoints/{hp.exp_name}',
            filename='interval-checkpoint-{epoch:04d}',
            save_top_k=-1
        )
        callbacks.append(interval_checkpoint_callback)

    trainer = pl.Trainer(
        logger=logger,
        accelerator='gpu',
        precision='bf16-mixed',
        max_steps=hp.train.get('max_steps', 160000),
        callbacks=callbacks,
        check_val_every_n_epoch=hp.train.get('val_interval', 1),
    )
    trainer.fit(training_module, train_dataloader, val_dataloader)