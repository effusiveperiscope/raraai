import argparse
import os
import random
import torch
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

def win_longpath(path):
    if os.name != 'nt':
        return path
    return '\\\\?\\' + os.path.abspath(path)

def resample_audio(audio, orig_sr, target_sr=16000):
    """Resample audio to target sample rate using librosa."""
    if orig_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr,
            res_type='fft')
    return audio

def process_filelist(
    filelist_path, val_fraction=0.05,
    output_dir='data/preprocessed', shuffle_seed=42):

    with open(filelist_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]

    os.makedirs(output_dir, exist_ok=True)

    random.seed(shuffle_seed)
    random.shuffle(lines)

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

        line = win_longpath(line)

        if not os.path.exists(line):
            print(f'File not found: {line}')
            continue

        audio, orig_sr = sf.read(line)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        audio = resample_audio(audio, orig_sr, 16000)
        audio_tensor = torch.from_numpy(audio)

        out_filepath = os.path.join(output_dir, f'{os.path.basename(line)}.pt')
        out_filepath = win_longpath(out_filepath)
        torch.save(audio_tensor, out_filepath)

        out_lines.append(out_filepath)

    if val_fraction > 0:
        val_size = int(len(lines) * val_fraction)
        val_lines = out_lines[-val_size:]
        train_lines = out_lines[:-val_size]
    else:
        val_lines = []
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
