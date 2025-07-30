import argparse
import os
import random
import torch
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

    extractor = MyFeatures()

    is_multispk = False
    new_lines = []
    for line in tqdm(lines, total=len(lines), desc='Preprocessing'):
        orig_line = line
        if 'longform' in line:
            # These are for longform in Expresso - will cause OOM
            continue
        if '|' in line:
            if not is_multispk:
                print('=== Multispeaker filelist detected! ===')
            is_multispk = True
            split = line.split('|')
            line = split[0]
        if not os.path.exists(line):
            print(f'File not found: {line}')
            continue
        new_lines.append(line)

        feats = extractor.extract_features(line)
        for key, value in feats.items():
            torch.save(value, os.path.join(args.output_dir, os.path.basename(line) + '.' + key))

    if args.val_fraction > 0:
        val_lines = new_lines[-int(len(lines) * args.val_fraction):]
        train_lines = new_lines[:-int(len(lines) * args.val_fraction)]
    else:
        val_lines = []
        train_lines = new_lines

    with open(os.path.join(args.output_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_lines))
    with open(os.path.join(args.output_dir, 'val.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_lines))