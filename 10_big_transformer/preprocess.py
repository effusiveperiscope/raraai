import os
from omegaconf import OmegaConf
import torch
from features import MyFeatures
import argparse
import librosa
from tqdm import tqdm

ABBREV_LEN = 60
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filelist", help="Line separated filelist to source audio")
    parser.add_argument("--output_dir", help="output dir", default="testdata")
    parser.add_argument("--config", help="config", default="configs/common.yaml")

    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    out_filelist = {}
    lines = []
    with open(args.filelist, "r", encoding="utf-8") as f:
        lines = f.readlines()

    original_want = config.features.want

    # If we try to extract all features at once, it will run out of memory
    print("Stage 1 - Content/Hubert/Pitch")
    config.features.want = ["content_tokens", "content_interp_pitch"]
    myfeatures = MyFeatures(config, "cuda")
    def extract_features():
        for i, line in tqdm(enumerate(lines), desc="Extracting features", total=len(lines)):
            abbrev_name = os.path.splitext(
                os.path.basename(line.strip()))[0][:ABBREV_LEN]
            wav_path = line.strip()

            want_features = config.features.want
            feat_dict = myfeatures.extract_features(wav_path)

            if not abbrev_name in out_filelist:
                out_filelist[abbrev_name] = {
                    'wav_path': wav_path
                }
            for w in want_features:
                feat_path = f"{args.output_dir}/{abbrev_name}_{w}.pt"
                torch.save(feat_dict[w], feat_path)
                out_filelist[abbrev_name][w] = os.path.abspath(feat_path)
    extract_features()

    print("Stage 2 - Acoustic")
    del myfeatures
    config.features.want = ["acoustic_codes"]
    myfeatures = MyFeatures(config, "cuda")
    extract_features()

    with open(f"{args.output_dir}/filelist.txt", "w", encoding="utf-8") as f:
        for (k, v) in out_filelist.items():
            line = f'{v["wav_path"]}|'
            for w in original_want:
                line += f"{v[w]}|"
            line = line[:-1]
            f.write(line + "\n")