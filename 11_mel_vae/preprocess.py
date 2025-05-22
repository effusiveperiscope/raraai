import argparse
import os
import torch
from tqdm import tqdm
from features import MyFeatures

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", type=str, required=True)
    parser.add_argument("--val_list", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_list = []
    val_list = []
    with open(args.train_list, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        train_list = [line.strip() for line in lines]
    with open(args.val_list, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        val_list = [line.strip() for line in lines]

    myfeatures = MyFeatures()
    
    output_list = []
    for line in tqdm(train_list + val_list, total=len(train_list) + len(val_list), desc='Preprocessing'):
        basename = os.path.basename(line)
        mel_spec, f0 = myfeatures.extract_features(line)
        # mel - [1, n_mels, T]
        # f0 - [1, T]
        mel_path = os.path.abspath(os.path.join(args.output_dir, f'{basename}.mel'))
        f0_path = os.path.abspath(os.path.join(args.output_dir, f'{basename}.f0'))
        torch.save(mel_spec, mel_path)
        torch.save(f0, f0_path)
        output_list.append(f"{mel_path}|{f0_path}")

    with open(os.path.join(args.output_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_list[:len(train_list)]))
    with open(os.path.join(args.output_dir, 'val.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_list[len(train_list):]))