import itertools

import librosa
from omegaconf import OmegaConf
import torch
from features import FeatureExtractor
from utils import win_longpath
from modeling.vits.utils import f0_to_coarse
import ultimate_xc
import random
import os
import argparse
from tqdm import tqdm

def process_filelist(filelist_path,
    val_fraction=0.05, output_dir='data/preprocessed', shuffle_seed=42,
    skip_if_one_exists=False):

    with open(filelist_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]

    os.makedirs(output_dir, exist_ok=True)

    random.seed(shuffle_seed)
    random.shuffle(lines)

    hp = OmegaConf.load("configs/base.yaml")
    fe = FeatureExtractor(hp=hp)

    new_lines = []

    for line in tqdm(lines, total=len(lines), desc='Preprocessing'):
        if not line:
            continue
        if 'longform' in line:
            continue
        split = line.split('|')
        line = split[0]
        sid = split[1]
        transcr = split[2]

        base_line = line
        line = win_longpath(line)
        if not os.path.exists(line):
            print(f'File not found: {line}')
            continue

        wav_16k, _ = librosa.load(line, sr=16000)
        wav_48k, _ = librosa.load(line, sr=48000)
        savepaths = []
        try:

            expected_keys = ['wave', 'spec', 'f0']
            if all(os.path.exists(os.path.join(output_dir,
                os.path.basename(base_line) + '.' + key)) for key in expected_keys) and skip_if_one_exists:
                print('File already exists, skipping...')
                return

            feats, _ = fe.extract(wav_16k, transcription=transcr)
            feats_ac = fe.extract_features_ac(wav_48k)
            phone_ids = list(itertools.chain.from_iterable(
                [f['phone_ids'] for f in feats]))

            feats_n = {
                'phone_ids': torch.tensor(phone_ids),  
                'spec': feats_ac['spec'],
                'f0': f0_to_coarse(feats_ac['f0']), # f0_to_coarse?
            }
            # phone_ids [n_phone]
            # spec [n_spec, channels]
            # f0 [n_spec]
            # post_ids [n_spec // 4, n_codes, n_]
            for key, value in feats_n.items():
                savepath = os.path.join(output_dir,
                    os.path.basename(base_line) + '.' + key)
                savepath = win_longpath(savepath)
                torch.save(value, savepath)
                savepaths.append(savepath)

        except Exception as e:
            print(f'Error processing {line}: {e}')
            raise e
            continue

        newline = '|'.join(savepaths)+f'|{sid}'
        new_lines.append(newline)

    if val_fraction > 0:
        val_size = int(len(lines) * val_fraction)
        val_lines = new_lines[-val_size:]
        train_lines = new_lines[:-val_size]
    else:
        val_lines = [new_lines[0]] # ensure at least one item in val for checkpointing
        train_lines = new_lines

    with open(os.path.join(output_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_lines))
    with open(os.path.join(output_dir, 'val.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_lines))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', type=str, help='path to filelist', required=True)
    parser.add_argument('--val_fraction', type=float, default=0.05)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--shuffle_seed', type=int, default=42)

    args = parser.parse_args()

    process_filelist(
        filelist_path=args.filelist,
        val_fraction=args.val_fraction,
        output_dir=args.output_dir,
        shuffle_seed=args.shuffle_seed,
    )