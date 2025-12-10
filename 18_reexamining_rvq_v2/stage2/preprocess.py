# file: preprocess.py

import argparse
import os
import random
import torch
from tqdm import tqdm
from features import MyFeatures
import ultimate_xc
import re
import sys

def win_longpath(path):
    return '\\\\?\\' + os.path.abspath(path)

def process_filelist(filelist_path, config='configs/base.yaml', val_fraction=0.05,
                     output_dir='output', shuffle_seed=42, 
                     feats_to_extract=None,
                     regen_filelist=False,
                     skip_exists=False,
                     filepath_regex_pattern=None,
                     filepath_regex_rep=None):
    with open(filelist_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]

    os.makedirs(output_dir, exist_ok=True)

    random.seed(shuffle_seed)
    random.shuffle(lines)

    if filepath_regex_pattern is not None:
        lines = [re.sub(
            filepath_regex_pattern, filepath_regex_rep, line).replace(
                '\\', '/') for line in lines]
    lines = list(set(lines)) # cull potential duplicate lines created by regex/replacement

    if not regen_filelist:
        if feats_to_extract is None:
            extractor = MyFeatures(config=config)
        else:
            extractor = MyFeatures(
                config=config,
                feats_to_extract=feats_to_extract
            )

    is_multispk = False
    new_lines = []
    sid_avgs = {}
    sid_sums = {}
    skip_flag = False

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

        if sys.platform == 'win32':
            line = win_longpath(line)

        if not os.path.exists(line):
            print(f'File not found: {line}')
            continue

        if not regen_filelist:
            expected_keys = [
                'whisper', 'f0',
                'f0_confidence', 'f0_subharmonic', 'f0_inharmonic',
                'spec', 'wave']
            savepaths = {}
            for key in expected_keys:
                savepath = os.path.join(output_dir, os.path.basename(line) + '.' + key)
                if sys.platform == 'win32':
                    savepath = win_longpath(savepath)
                savepaths[key] = savepath
            if all(os.path.exists(savepath) for key, savepath in savepaths.items()) and skip_exists:
                skip_flag = True
                # print("skip_flag triggered on line ", line)
                newline = '|'.join(
                    [savepaths['whisper'],
                    savepaths['f0'],
                    savepaths['f0_confidence'],
                    savepaths['f0_subharmonic'],
                    savepaths['f0_inharmonic'],
                    savepaths['spec'],
                        savepaths['wave'], 
                        str(sid)])
                new_lines.append(newline)
                continue

            feats = extractor.extract_features(line)

            for key in expected_keys:
                if key not in feats:
                    feats[key] = None

            for key, value in feats.items():
                savepath = os.path.join(output_dir, os.path.basename(line) + '.' + key)
                if sys.platform == 'win32':
                    savepath = win_longpath(savepath)
                if value is not None:
                    torch.save(value, savepath)

            newline = '|'.join([
                savepaths['whisper'],
                savepaths['f0'],
                savepaths['f0_confidence'],
                savepaths['f0_subharmonic'],
                savepaths['f0_inharmonic'],
                savepaths['spec'],
                savepaths['wave'],
                str(sid)
            ])
            new_lines.append(newline)
        else:
            savepaths = {}
            for key in expected_keys:
                savepaths[key] = os.path.join(output_dir, os.path.basename(line) + '.' + key)
            newline = '|'.join(
                [savepaths['whisper'],
                 savepaths['f0'],
                  savepaths['f0_confidence'],
                  savepaths['f0_subharmonic'],
                  savepaths['f0_inharmonic'],
                  savepaths['spec'],
                    savepaths['wave'], 
                    str(sid)])
            new_lines.append(newline)


    if val_fraction > 0:
        val_size = int(len(lines) * val_fraction)
        val_lines = new_lines[-val_size:]
        train_lines = new_lines[:-val_size]
    else:
        val_lines = []
        train_lines = new_lines

    # if not regen_filelist and not skip_flag: # can't regen sid_avgs if skipped any
    #     if feats_to_extract is None or 'sid' in feats_to_extract:
    #         print('Saving sid_avgs...')
    #         for sid, avg in sid_avgs.items():
    #             sid_avgs[sid] = sid_avgs[sid] / sid_sums[sid]
    #         torch.save(sid_avgs, os.path.join(output_dir, 'sid_avgs.pt'))
    # else:
    #     if regen_filelist:
    #         print('Skipped saving sid_avgs because regen_filelist was True')
    #     elif skip_flag:
    #         print('Skipped saving sid_avgs because skip_flag was triggered')

    with open(os.path.join(output_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_lines))
    with open(os.path.join(output_dir, 'val.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_lines))

    if not regen_filelist:
        del extractor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', type=str, help='path to filelist')
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    parser.add_argument('--val_fraction', type=float, default=0.05)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--shuffle_seed', type=int, default=42)
    parser.add_argument('--regen_filelist', action='store_true')
    parser.add_argument('--skip_exists', action='store_true')

    args = parser.parse_args()

    process_filelist(
        filelist_path=args.filelist,
        config=args.config,
        val_fraction=args.val_fraction,
        output_dir=args.output_dir,
        shuffle_seed=args.shuffle_seed,
        regen_filelist=args.regen_filelist,
        #feats_to_extract={'f0'},
        skip_exists=args.skip_exists
    )
