import os
from preprocess import process_filelist
import torch
from omegaconf import OmegaConf
from modeling.vits.models import SynthesizerTrn
from svc_helper.svc.rvc.lib.infer_pack.models import MultiPeriodDiscriminatorV2
import sys
sys.path.append('..')
from rvq.vevo_repcodec import VevoRepCodec
from commons import load_submodule_prefix
from dataset import dataset
import pytorch_lightning as pl
from train import TrainingModule

import logging
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('fsspec').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)

import torch.multiprocessing as mp

def main():
    SOURCE_FILELISTS_DIR = '/mnt/data/Code/MasterDataset/pony_reduced'
    BASE_MODEL = 'pretrain/2ac.ckpt'
    SRC_CONFIG = 'configs/char.yaml'
    for filelist in os.listdir(SOURCE_FILELISTS_DIR):
        abs_path = os.path.join(os.path.abspath(SOURCE_FILELISTS_DIR), filelist)
        basename = os.path.basename(filelist).split('.')[0]
        data_dir = os.path.join('data', basename)

        print(f'Processing {filelist}')
        process_filelist(
            abs_path, val_fraction=0.00,
            output_dir=data_dir, skip_exists=True,
            filepath_regex_pattern=r"D:\\",
            filepath_regex_rep="/mnt/data/")

        config = OmegaConf.load(SRC_CONFIG)
        config.exp_name = basename
        config.train.test_filelist = 'data/test/val_linux.txt'
        hp = config

        config.train.train_filelist = os.path.join(data_dir, 'train.txt')
        config.train.val_filelist = os.path.join(data_dir, 'val.txt')
        config.train.spk_index = os.path.join(data_dir, 'sid_avgs.pt')

        net_g = SynthesizerTrn(
            spec_channels=hp.data.filter_length // 2 + 1,
            segment_size=hp.data.segment_size // hp.data.hop_length,
            hp=hp
        )
        net_d = MultiPeriodDiscriminatorV2(use_spectral_norm=False)
        codec = VevoRepCodec(
            input_channels=hp.codec.whisper_dim,
            output_channels=hp.codec.whisper_dim,
            code_dim=hp.codec.whisper_dim,
            codebook_num=1,
            codebook_size=hp.codec.codebook_size
        )

        resume_ckpt = os.path.join('checkpoints', basename, 'last.ckpt')

        with open(config.train.train_filelist) as f:
            line_count = len(f.readlines())
        len_dataset = line_count
        steps_factor = 40
        max_steps = 40000 + (len_dataset * steps_factor)
        config.train.test_interval = int(4200 * 2 / len_dataset)
        
        if not os.path.exists(resume_ckpt):
            print('Found no checkpoint, transfering from base')
            # transfer learn from base
            state = torch.load(
                BASE_MODEL, map_location='cpu', weights_only=False)['state_dict']
            load_submodule_prefix(net_g, 'net_g.', state)
            load_submodule_prefix(codec, 'codec.', state) # oops lol
            load_submodule_prefix(net_d, 'net_d.', state)
            resume_ckpt = None
        else:
            resume_ckpt_state = torch.load(resume_ckpt, map_location='cpu', weights_only=False)
            global_step = resume_ckpt_state['global_step']
            if global_step >= max_steps:
                print(f'Checkpoint at {resume_ckpt} exceeds or has reached max steps {max_steps}, skipping')
                continue
            print(f'Resuming from {resume_ckpt}')

        print("Loading data...")
        train_dataset = dataset(config.train.train_filelist, is_train=True)
        test_sample = train_dataset[0]
        print("Creating dataloaders...")
        if len_dataset < 100:
            num_workers = 0 # Short datasets will incur too much overhead with num_workers > 0
        else:
            num_workers = config.train.get('num_workers', 4)
        train_dataloader = train_dataset.loader(
            batch_size=config.train.batch_size, shuffle=True, num_workers=num_workers,
                persistent_workers=num_workers > 0)

        training_module = TrainingModule(
            net_g=net_g, net_d=net_d, test_sample=test_sample, codec=codec, config=config)
        logger = pl.loggers.TensorBoardLogger(
            config.train.get('log_dir', 'logs'), name=config.exp_name,
            version=config.get('tensorboard_version', 0)
        )
        print("Done")

        callbacks = [
            pl.callbacks.ModelCheckpoint( # just save last
                dirpath=f'checkpoints/{config.exp_name}',
                save_last=True
            )
        ]

        trainer = pl.Trainer(
            logger=logger,
            accelerator='gpu',
            precision='bf16-mixed',
            max_steps=max_steps,
            callbacks=callbacks,
            val_check_interval=0, # no validation
            limit_val_batches=0,
            log_every_n_steps=config.train.get('log_interval', 50),
        )
        trainer.fit(training_module, train_dataloader, ckpt_path = resume_ckpt)

        del net_g
        del net_d
        del codec
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()