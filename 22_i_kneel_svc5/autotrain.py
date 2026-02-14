import os
import re
from train import train
from preprocess import process_filelist
from omegaconf import OmegaConf

import warnings
import librosa
import logging
warnings.filterwarnings('error', category=RuntimeWarning)
warnings.simplefilter('ignore', category=UserWarning) 
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('fsspec').setLevel(logging.WARNING)

SOURCE_FILELIST_PATH = "/mnt/data/Code/MasterDataset/pony_enhanced/"
TMP_FILELIST = "filelists/tmp.txt"
CONFIG = "configs/char.yaml"
TRANSFER_FROM = "pretrain/titan_last.ckpt"

def linux_filelist_line(line):
    if os.name == 'nt': return line
    regex_pattern = r'D:\\'
    regex_rep = '/mnt/data/'
    line = line.replace('\\\\?\\', "")
    line = re.sub(regex_pattern, regex_rep, line).replace('\\','/')
    return line

if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True) # This is needed on Linux

    for filelist in os.listdir(SOURCE_FILELIST_PATH):
        exp_name = os.path.basename(filelist).split('.')[0] + '_v2'

        with open(SOURCE_FILELIST_PATH+filelist, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]
            lines = [linux_filelist_line(line) for line in lines]
            line_count = len(lines)

        if not os.path.exists(f'logs/{exp_name}'): 
            with open(TMP_FILELIST, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))

            print("Processing", exp_name)
            process_filelist(
                TMP_FILELIST, val_fraction=0, output_dir=f'data/{exp_name}', shuffle_seed=42,
                skip_if_one_exists=True)
            

        len_dataset = line_count
        steps_factor = 30
        max_steps = 280000

        config = OmegaConf.load(CONFIG)
        OmegaConf.set_struct(config, True)
        config.exp_name = exp_name
        config.train.train_filelist = os.path.join('data', exp_name, 'train.txt')
        config.train.val_filelist = os.path.join('data', exp_name, 'val.txt')
        config.train.spk_index = os.path.join('data', exp_name, 'sid_avgs.pt')
        config.train.max_steps = max_steps
        if line_count < 1000:
            config.train.val_interval = 2000 // line_count
            config.train.test_interval = 2000 // line_count
        else:
            config.train.val_interval = 2
            config.train.test_interval = 2
        print("Training", exp_name, "for", max_steps, "steps")
        if not os.path.exists(f'logs/{exp_name}'): 
            train(config,
                resume_from=None,
                transfer_from=TRANSFER_FROM,
                svc5_ckpt=None,
                rvc_disc_ckpt=None,
                prior_ckpt=None)
        else:
            print("Resuming", exp_name)
            last_ckpt_files = [f'checkpoints/{exp_name}/last.ckpt', f'checkpoints/{exp_name}/last-v1.ckpt']
            last_ckpt = max((file for file in last_ckpt_files if os.path.exists(file)), key=os.path.getmtime, default=None)
            train(config,
                resume_from=last_ckpt if os.path.exists(last_ckpt) else None,
                transfer_from=None,
                svc5_ckpt=None,
                rvc_disc_ckpt=None,
                prior_ckpt=None)