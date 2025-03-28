from svc_helper.sfeatures.models import RVCHubertModel
from svc_helper.pitch.rmvpe import RMVPEModel
import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import librosa
from torch.utils.data import Dataset, DataLoader

def win_longpath(path):
    return '\\\\?\\' + os.path.abspath(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filelist", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    rvc = RVCHubertModel(device=device)
    rmvpe = RMVPEModel(device=device)

    wav_infos = []
    with open(args.filelist, "r", encoding='utf-8') as f:
        for line in f:
            wav_path = line.strip()
            if '|' in wav_path:
                wav_path = wav_path.split('|')[0]
            wav_path = win_longpath(wav_path)
            wav_infos.append({
                "wav_path": wav_path,
                "wav_name": os.path.basename(wav_path)})

    filelist = ""
    for wav_info in tqdm(wav_infos):
        wav_path = wav_info["wav_path"]
        wav_name = wav_info["wav_name"]

        wav, sr = librosa.load(wav_path, sr=16000)
        f0 = rmvpe.extract_pitch(torch.from_numpy(wav))
        feat = rvc.extract_features(torch.from_numpy(wav))

        os.makedirs(os.path.join(args.output_dir, os.path.dirname(wav_name)), exist_ok=True)
        
        f0_out_path = os.path.join(args.output_dir, wav_name.replace(".wav", ".f0.npy"))
        feat_out_path = os.path.join(args.output_dir, wav_name.replace(".wav", ".npy"))

        filelist += f"{wav_name}|{f0_out_path}|{feat_out_path}\n"

        torch.save(torch.from_numpy(f0), f0_out_path)
        torch.save(feat, feat_out_path)

    with open(os.path.join(args.output_dir, "filelist.list"), "w", encoding='utf-8') as f:
        f.write(filelist)