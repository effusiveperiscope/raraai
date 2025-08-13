# file: preprocess.py

import argparse
import os
import random
import torch
from tqdm import tqdm
from features import MyFeatures

def process_filelist(filelist_path, config='configs/base.yaml', val_fraction=0.05,
                     output_dir='output', shuffle_seed=42, regen_filelist=False):
    with open(filelist_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]

    os.makedirs(output_dir, exist_ok=True)

    random.seed(shuffle_seed)
    random.shuffle(lines)

    if not regen_filelist:
        extractor = MyFeatures(config=config)

    is_multispk = False
    new_lines = []
    sid_avgs = {}
    sid_sums = {}

    for line in tqdm(lines, total=len(lines), desc='Preprocessing'):
        if 'longform' in line:
            continue

        if '|' in line:
            if not is_multispk:
                print('=== Multispeaker filelist detected! ===')
            is_multispk = True
            split = line.split('|')
            line = split[0]
            sid = split[1]
        else:
            sid = 0

        if not os.path.exists(line):
            print(f'File not found: {line}')
            continue

        if not regen_filelist:
            feats = extractor.extract_features(line)
            savepaths = {}
            for key, value in feats.items():
                savepath = os.path.join(output_dir, os.path.basename(line) + '.' + key)
                savepaths[key] = savepath
                torch.save(value, savepath)

            newline = '|'.join([
                savepaths['whisper'],
                savepaths['f0'],
                savepaths['spec'],
                savepaths['spk'],
                savepaths['wave'],
            ])
            new_lines.append(newline)
            if sid not in sid_avgs:
                sid_avgs[sid] = torch.zeros_like(feats['spk'])
                sid_sums[sid] = 0
            sid_avgs[sid] += feats['spk']
            sid_sums[sid] += 1
        else:
            savepaths = {}
            for key in ['whisper', 'f0', 'spec', 'spk', 'wave']:
                savepaths[key] = os.path.join(output_dir, os.path.basename(line) + '.' + key)
            newline = '|'.join(
                [savepaths['whisper'],
                 savepaths['f0'],
                  savepaths['spec'],
                   savepaths['spk'],
                    savepaths['wave'], str(sid)])
            new_lines.append(newline)


    if val_fraction > 0:
        val_size = int(len(lines) * val_fraction)
        val_lines = new_lines[-val_size:]
        train_lines = new_lines[:-val_size]
    else:
        val_lines = []
        train_lines = new_lines

    if not regen_filelist:
        for sid, avg in sid_avgs.items():
            sid_avgs[sid] = sid_avgs[sid] / sid_sums[sid]
        torch.save(sid_avgs, os.path.join(output_dir, 'sid_avgs.pt'))

    with open(os.path.join(output_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_lines))
    with open(os.path.join(output_dir, 'val.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_lines))

    if not regen_filelist:
        del extractor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', type=str, required=True, help='path to filelist')
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    parser.add_argument('--val_fraction', type=float, default=0.05)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--shuffle_seed', type=int, default=42)
    parser.add_argument('--regen_filelist', action='store_true')

    args = parser.parse_args()

    process_filelist(
        filelist_path=args.filelist,
        config=args.config,
        val_fraction=args.val_fraction,
        output_dir=args.output_dir,
        shuffle_seed=args.shuffle_seed,
        regen_filelist=args.regen_filelist
    )
