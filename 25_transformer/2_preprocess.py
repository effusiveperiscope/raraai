import argparse
import itertools
import os
import random

import librosa
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

import speaker_map as speaker_map_lib
from commons import load_state_dict_mismatch, load_submodule_prefix
from features import FeatureExtractor
from modeling.generator import Generator
from utils import win_longpath

import logging
# Something above this is setting DEBUG but I don't know what
logging.getLogger().setLevel(logging.WARNING)


def load_generator(checkpoint_path, config, device):
    net_g = Generator(hp=config)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    if any(k.startswith('net_g.') for k in state.keys()):
        load_submodule_prefix(net_g, 'net_g.', state)
    else:
        load_state_dict_mismatch(net_g, state)
    net_g.to(device)
    net_g.eval()
    return net_g

def process_filelist(filelist_path,
    val_fraction=0.05, output_dir='data/preprocessed', shuffle_seed=42,
    skip_if_one_exists=False, checkpoint_path=None, config_path='configs/base.yaml',
    device=None, speaker_map_path=None):
    if checkpoint_path is None:
        raise ValueError('--checkpoint is required: path to a 1_training checkpoint')

    with open(filelist_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]

    os.makedirs(output_dir, exist_ok=True)

    random.seed(shuffle_seed)
    random.shuffle(lines)

    hp = OmegaConf.load(config_path)
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    fe = FeatureExtractor(hp=hp)
    net_g = load_generator(checkpoint_path, hp, device)

    # Optional canonicalisation of speaker ids into the stage-1 space so a
    # single id works for both the stage-2 transformer and the frozen
    # Generator's speaker embedding (see speaker_map.py).
    smap = None
    if speaker_map_path is not None:
        smap = speaker_map_lib.load_speaker_map(speaker_map_path)
        print(f'Using speaker map {speaker_map_path}')

    new_lines = []
    n_skipped_speaker = 0

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

        if smap is not None:
            resolved = speaker_map_lib.resolve_sid(
                smap, path=base_line, raw_sid=int(sid))
            if resolved is None:
                n_skipped_speaker += 1
                continue
            sid = resolved
        else:
            sid = int(sid)

        wav_16k, _ = librosa.load(line, sr=16000)
        wav_48k, _ = librosa.load(line, sr=48000)
        savepaths = []
        try:

            expected_keys = ['phone_ids', 'spec_codes', 'pitch_codes']
            if all(os.path.exists(os.path.join(output_dir,
                os.path.basename(base_line) + '.' + key)) for key in expected_keys) and skip_if_one_exists:
                print('File already exists, skipping...')
                return

            feats, _ = fe.extract(wav_16k, transcription=transcr)
            feats_ac = fe.extract_features_ac(wav_48k)
            phone_ids = list(itertools.chain.from_iterable(
                [f['phone_ids'] for f in feats]))

            # Encode raw spec/f0 into discrete RVQ codes with the 1_training
            # Generator checkpoint (channel-first spec, frame-rate f0).
            spec = feats_ac['spec']  # [T, C]
            f0 = feats_ac['f0'].float()  # [T]
            spec_b = spec.transpose(0, 1).unsqueeze(0).to(device)  # [1, C, T]
            spec_l = torch.tensor([spec.shape[0]], device=device)
            f0_b = f0.unsqueeze(0).to(device)  # [1, T]
            sid_b = torch.tensor([int(sid)], device=device)
            with torch.no_grad():
                spec_codes, pitch_codes = net_g.encode_codes(
                    spec_b, spec_l, f0_b, sid_b)
            # encode_codes returns (Q, B, T_ds); drop batch dim -> (Q, T_ds).
            spec_codes = spec_codes.squeeze(1).cpu()
            pitch_codes = pitch_codes.squeeze(1).cpu()

            feats_n = {
                'phone_ids': torch.tensor(phone_ids),
                'spec_codes': spec_codes,
                'pitch_codes': pitch_codes,
            }
            # phone_ids [n_phone]
            # spec_codes [Q_spec, T_codes] (Q_spec = codec.codebook_num)
            # pitch_codes [Q_pitch, T_codes] (Q_pitch = pitch_codec.codebook_num)
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

    if smap is not None and n_skipped_speaker:
        print(f'Skipped {n_skipped_speaker} file(s) with no speaker-map match')

    if val_fraction > 0:
        val_size = int(len(lines) * val_fraction)
        val_lines = new_lines[-val_size:]
        train_lines = new_lines[:-val_size]
    else:
        val_lines = [new_lines[0]] # ensure at least one item in val for checkpointing
        train_lines = new_lines

    with open(os.path.join(output_dir, 'train_stage2.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_lines))
    with open(os.path.join(output_dir, 'val_stage2.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_lines))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', type=str, help='path to filelist', required=True)
    parser.add_argument('--val_fraction', type=float, default=0.05)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--shuffle_seed', type=int, default=42)
    parser.add_argument('--checkpoint', type=str, required=True,
        help='path to 1_training (lightning or Generator) checkpoint')
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--speaker_map', type=str, default=None,
        help='JSON map canonicalising speaker ids to the stage-1 space '
             '(see speaker_map.py)')

    args = parser.parse_args()

    process_filelist(
        filelist_path=args.filelist,
        val_fraction=args.val_fraction,
        output_dir=args.output_dir,
        shuffle_seed=args.shuffle_seed,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        speaker_map_path=args.speaker_map,
    )