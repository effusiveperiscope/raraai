import torch
from dataclasses import dataclass
from svc_helper.svc.rvc.lib.infer_pack.models import MultiPeriodDiscriminatorV2
from commons import load_submodule_prefix
from dataset import FeatureCollator, FeatureDataset
from modeling.v08.rvc import V08Synthesizer
from train_v08_stage2 import V08TrainingModule
import pytorch_lightning as pl
import os
import shutil
import subprocess
from omegaconf import OmegaConf

@dataclass
class TrainSpec:
    exp_name: str
    src_filelist: str

# py preprocess.py --filelist filelists\fluttershy_test.txt --output_dir data\fluttershy_test
# py train_v08_stage2.py --config configs\finetune.yaml --transfer_from checkpoints\teacher\v08_test02\last.ckpt

def process_spec(
    train_spec : TrainSpec,
    transfer_ckpt : str = 'checkpoints/teacher/v08_test02/last.ckpt',
    base_config : str = 'configs/finetune.yaml'):

    if not os.path.exists(os.path.join('data', train_spec.exp_name)):
        subprocess.run(["py", "preprocess.py", 
            "--filelist", train_spec.src_filelist, 
            "--output_dir", "data\\" + train_spec.exp_name])

    config = OmegaConf.load(base_config)
    config.exp_name = train_spec.exp_name
    config.filelist = os.path.join('data', train_spec.exp_name, 'train.txt')
    config.val_filelist = os.path.join('data', train_spec.exp_name, 'val.txt')

    net_g = V08Synthesizer(**config.model, is_half=True,
        use_pitch_predictor=config.model.get('use_pitch_predictor', False),
        pitch_quant_dim=config.model.get('pitch_quant_dim', 8))
    net_d = MultiPeriodDiscriminatorV2(use_spectral_norm=False)

    state = torch.load(transfer_ckpt, map_location='cpu')['state_dict']
    load_submodule_prefix(net_g, 'net_g.', state)
    load_submodule_prefix(net_d, 'net_d.', state)

    training_module = V08TrainingModule(net_g, net_d, config)
        
    logger = pl.loggers.TensorBoardLogger(
        config.train.get('log_dir', 'logs'), name=config.exp_name,
    )
    train_dataset = FeatureDataset(config, is_train=True)
    val_dataset = FeatureDataset(config, is_train=False)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = config.train.batch_size,
        shuffle = True,
        collate_fn = FeatureCollator(),
        num_workers=4,
        persistent_workers=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = config.train.batch_size,
        shuffle = False,
        collate_fn = FeatureCollator(),
        num_workers=4,
        persistent_workers=True
    )

    val_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'checkpoints/teacher/{config.exp_name}',
        filename='best-checkpoint',
        save_top_k=2,
        mode='min',
        save_last=True
    )
    callbacks = [val_checkpoint_callback]
    if config.train.get('save_every_n_epochs'):
        interval_checkpoint_callback = pl.callbacks.ModelCheckpoint(
            every_n_epochs=config.train.save_every_n_epochs,
            dirpath=f'checkpoints/teacher/{config.exp_name}',
            filename='interval-checkpoint-{epoch:04d}',
            save_top_k=-1
        )
        callbacks.append(interval_checkpoint_callback)

    trainer = pl.Trainer(
        logger=logger,
        accelerator='gpu',
        precision='bf16-mixed',
        max_epochs=config.train.epochs,
        callbacks=callbacks,
    )
    trainer.fit(training_module, train_dataloader, val_dataloader)

train_specs = [
    TrainSpec(exp_name='twilight_anchor',
        src_filelist='filelists/twilight_anchor.txt')
]