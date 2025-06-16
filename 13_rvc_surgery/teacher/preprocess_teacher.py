import argparse
import os
import random
import torch
import librosa
from tqdm import tqdm
from features import MyFeatures
from svc_helper.sfeatures.models import SVC5TextEncoderFullModel
from svc_helper.speaker.models import SVC5SpeakerEncoder
from einops import rearrange

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

    # We will use whisper-base.en for the student
    my_feats = MyFeatures(
        extract_hubert=False, extract_whisper=True, extract_vevo=False)
    text_encoder = SVC5TextEncoderFullModel(device='cuda')
    spk_encoder = SVC5SpeakerEncoder(device='cuda')

    is_multispk = False

    new_lines = [] # May need to filter out too short lines
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
            sid = split[1]
        if not os.path.exists(line):
            print(f'File not found: {line}')
            continue

        basename = os.path.basename(line)

        data, _ = librosa.load(line, sr=16000)
        data_48k, _ = librosa.load(line, sr=48000)
        features = None
        if not all(os.path.exists(
            os.path.join(args.output_dir, basename + ext)) for ext in ['.whisp_feat', '.pitch_fine']):
            features = my_feats.get_features(data, data_48k)
            if features['whisp_feat'].shape[1] < 32:
                print(f'Skipping short file: {line}')
                continue
            torch.save( # Student speech feature
                features['whisp_feat'],
                os.path.join(args.output_dir, basename+'.whisp_feat'))
            torch.save( # 
                features['pitch_fine'],
                os.path.join(args.output_dir, basename+'.pitch_fine'))

        if not os.path.exists(os.path.join(args.output_dir, basename + '.wave')):
            features = my_feats.get_features(data, data_48k)
            torch.save(
                features['spec'],
                os.path.join(args.output_dir, basename+'.spec'))
            torch.save(
                features['wave'],
                os.path.join(args.output_dir, basename+'.wave'))

        if not os.path.exists(os.path.join(args.output_dir, basename + '.svc5_feat')):
            teacher_features = text_encoder.extract_features(data)
            teacher_features = rearrange(teacher_features, "b c t -> b t c")

            torch.save(
                teacher_features,
                os.path.join(args.output_dir, basename+'.svc5_feat'))
        
        spk_features = spk_encoder.extract_feature(line)
        torch.save(
            spk_features,
            os.path.join(args.output_dir, basename+'.spk_feat'))

        new_lines.append(orig_line)

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