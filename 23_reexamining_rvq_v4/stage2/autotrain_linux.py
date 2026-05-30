import os
from preprocess import process_filelist
import torch
from omegaconf import OmegaConf
from modeling.vits.models import SynthesizerTrn
# from svc_helper.svc.rvc.lib.infer_pack.models import MultiPeriodDiscriminatorV2
from modeling.vits_decoder.discriminator import Discriminator
from modeling.intensity import IntensityModel
import sys
sys.path.append('..')
from rvq.vevo_repcodec import VevoRepCodec
from commons import load_submodule_prefix
from dataset import dataset
from train import TrainingModule
import lightning as L

import logging
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('fsspec').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)

import torch.multiprocessing as mp

def main():
    SOURCE_FILELISTS_DIR = '/mnt/data/Code/MasterDataset/temp'
    BASE_MODEL = 'pretrain/mlp_base_large.ckpt'
    SRC_CONFIG = 'configs/base_linux.yaml'
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
        config.exp_name = basename + '_large_halflr_mlpbase'
        config.train.test_filelist = 'data/test/val.txt'
        config.train.lr = config.train.lr / 2
        config.train.c_unvoiced = 0.2
        print(f"using lr {config.train.lr}")
        hp = config

        config.train.train_filelist = os.path.join(data_dir, 'train.txt')
        config.train.val_filelist = os.path.join(data_dir, 'val.txt')
        config.train.spk_index = os.path.join(data_dir, 'sid_avgs.pt')

        config.train.use_ema = True

        net_g = SynthesizerTrn(
            spec_channels=hp.data.filter_length // 2 + 1,
            segment_size=hp.data.segment_size // hp.data.hop_length,
            hp=hp
        )
        net_d = Discriminator(hp=hp)
        codec = VevoRepCodec(
            input_channels=hp.codec.whisper_dim,
            output_channels=hp.codec.whisper_dim,
            encode_channels=hp.codec.whisper_dim,
            decode_channels=hp.codec.whisper_dim,
            code_dim=hp.codec.code_dim,
            codebook_num=1,
            codebook_size=hp.codec.codebook_size
        )

        resume_ckpt = os.path.join('checkpoints', config.exp_name, 'last.ckpt')

        with open(config.train.train_filelist) as f:
            line_count = len(f.readlines())
        len_dataset = line_count
        config.train.test_interval = int(4200 * 2 / len_dataset)

        print("Loading data...")
        train_dataset = dataset(config.train.train_filelist, is_train=True)
        print("Creating dataloaders...")
        if len_dataset < 100:
            num_workers = 0 # Short datasets will incur too much overhead with num_workers > 0
        else:
            num_workers = config.train.get('num_workers', 4)
        train_dataloader = train_dataset.loader(
            batch_size=config.train.batch_size, shuffle=True, num_workers=num_workers,
                persistent_workers=num_workers > 0)
        print("Done")
        
        max_steps = 250000
        # max_steps = min(
            # len_dataset * 2000 / config.train.batch_size * 2,  # not sure why need x2
            # 250000) # 2000 epochs or 250k steps whichever is fewer
        est_epochs = max_steps / len_dataset

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

        training_module = TrainingModule(
            net_g=net_g, net_d=net_d, codec=codec, config=config)
        logger = L.pytorch.loggers.tensorboard.TensorBoardLogger(
            config.train.get('log_dir', 'logs'), name=config.exp_name,
            version=config.get('tensorboard_version', 0)
        )

        callbacks = [
            L.pytorch.callbacks.ModelCheckpoint( # just save last
                every_n_epochs=(1 if len_dataset > 100 else 20),
                dirpath=f'checkpoints/{config.exp_name}',
                filename='last'
            )
        ]

        interval_checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
            every_n_epochs=est_epochs // 4,
            dirpath=f'checkpoints/{config.exp_name}',
            filename='interval-checkpoint-{epoch:04d}', save_top_k=-1
        )
        callbacks.append(interval_checkpoint_callback)

        trainer = L.Trainer(
            logger=logger,
            accelerator='gpu',
            precision='bf16-mixed',
            max_steps=max_steps,
            callbacks=callbacks,
            val_check_interval=0, # no validation
            limit_val_batches=0,
            log_every_n_steps=config.train.get('log_interval', 50),
        )
        trainer.fit(training_module, train_dataloader, ckpt_path = resume_ckpt, weights_only=False)

        del net_g
        del net_d
        del codec

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
