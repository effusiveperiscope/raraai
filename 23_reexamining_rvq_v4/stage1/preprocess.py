import argparse
import os
import random
import torch
from tqdm import tqdm
from features import FeatureExtractor

def process_filelist(
    filelist_path, val_fraction=0.05,
    output_dir='data/preprocessed', shuffle_seed=42):

    with open(filelist_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]

    os.makedirs(output_dir, exist_ok=True)

    random.seed(shuffle_seed)
    random.shuffle(lines)

    extractor = FeatureExtractor()

    out_lines = []
    is_multispk = False
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

        feats = extractor.extract_features(line)

        out_filepath = os.path.join(output_dir, f'{os.path.basename(line)}.whisper')
        torch.save(feats, out_filepath)

        out_lines.append(out_filepath)

    if val_fraction > 0:
        val_size = max(int(len(lines) * val_fraction), 1)
        val_lines = out_lines[-val_size:]
        train_lines = out_lines[:-val_size]
    else:
        val_lines = [out_lines[0]]
        train_lines = out_lines

    with open(os.path.join(output_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_lines))
    with open(os.path.join(output_dir, 'val.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', type=str, required=True, help='path to filelist')
    parser.add_argument('--val_fraction', type=float, default=0.05)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--shuffle_seed', type=int, default=42)

    args = parser.parse_args()

    process_filelist(
        filelist_path=args.filelist,
        val_fraction=args.val_fraction,
        output_dir=args.output_dir,
        shuffle_seed=args.shuffle_seed
    )
