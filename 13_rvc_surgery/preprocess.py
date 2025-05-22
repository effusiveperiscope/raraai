import argparse
import os
import random
import torch

import librosa
from tqdm import tqdm
from features import MyFeatures

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', type=str, required=True, help='path to filelist')
    parser.add_argument('--val_fraction', type=float, default=0.05)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--shuffle_seed', type=int, default=42)

    args = parser.parse_args()
    with open(args.filelist, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.shuffle_seed)
    random.shuffle(lines)

    val_lines = lines[-int(len(lines) * args.val_fraction):]
    train_lines = lines[:-int(len(lines) * args.val_fraction)]

    my_feats = MyFeatures()

    for line in tqdm(lines, total=len(lines), desc='Preprocessing'):
        if not os.path.exists(line):
            print(f'File not found: {line}')
            continue
        data, _ = librosa.load(line, sr=16000)
        features = my_feats.get_features(data)

        basename = os.path.basename(line)
        torch.save(
            features['rvc_feat'],
            os.path.join(args.output_dir, basename+'.rvc_feat'))
        torch.save(
            features['whisp_feat'],
            os.path.join(args.output_dir, basename+'.whisp_feat'))
        torch.save(
            features['pitch'],
            os.path.join(args.output_dir, basename+'.pitch'))
        torch.save(
            features['pitch_fine'],
            os.path.join(args.output_dir, basename+'.pitch_fine'))

    with open(os.path.join(args.output_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_lines))
    with open(os.path.join(args.output_dir, 'val.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_lines))