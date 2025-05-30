# filelist format:
# path/to/audios/1.wav|spk_name|text
import os
import argparse

import librosa
import torch
from features import MyFeatures
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--filelist", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()

my_feats = MyFeatures()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

spk_map = {}

output_filelist = []
with open(args.filelist, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in tqdm(lines, total=len(lines), desc="Preprocessing"):
    line = line.strip()
    path, spk_name, text = line.split("|")
    if spk_name not in spk_map:
        spk_map[spk_name] = len(spk_map)
    spk_id = spk_map[spk_name]

    def make_filename(suffix):
        return os.path.join(args.output_dir, f"{spk_name}_{os.path.basename(path).split('.')[0]}_{suffix}.npy")
    output_filelist.append(
        f"{make_filename('whisper')}|{make_filename('phones')}|{spk_id}\n"
    )
    if os.path.exists(make_filename("whisper")) and os.path.exists(make_filename("phones")):
        continue

    audio, _ = librosa.load(path, sr=MyFeatures.expected_sample_rate)
    with torch.no_grad():
        whisper_features = my_feats.get_whisper_features(audio).cpu().numpy()
        phones_ids = my_feats.get_phonemes_ids(text)

    np.save(
        make_filename("whisper"), whisper_features)
    np.save(
        make_filename("phones"), phones_ids)

with open(os.path.join(args.output_dir, "filelist.txt"), "w", encoding="utf-8") as f:
    f.writelines(output_filelist)