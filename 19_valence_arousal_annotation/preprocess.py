import json
import os
import numpy as np
import torch
from features import FeatureExtractor
from tqdm import tqdm
import argparse

def win_longpath(path):
    return '\\\\?\\' + os.path.abspath(path)

class Preprocessor:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()

    def preprocess_multi_json(self, json_paths, output_dir, val_split=0.05):
        filelist = []
        os.makedirs(output_dir, exist_ok=True)
        for json_path in json_paths:
            filelist.extend(self.preprocess_annotations_json(json_path, output_dir))
        np.random.shuffle(filelist)
        split_idx = int(len(filelist) * (1 - val_split))
        train_filelist = filelist[:split_idx]
        val_filelist = filelist[split_idx:]
        with open(os.path.join(output_dir, "train_filelist.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(train_filelist))
        with open(os.path.join(output_dir, "val_filelist.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(val_filelist))
        
    def preprocess_annotations_json(self, json_path, output_dir):
        with open(json_path, "r", encoding="utf-8") as f:
            data : list[dict] = json.load(f)

        filelist_entries = []
        for item in tqdm(data, desc=f"Preprocessing {json_path}"):
            basename = os.path.basename(item["filepath"])
            path = win_longpath(item["filepath"])
            if not os.path.exists(path):
                continue
            if item["valence"] is None or item["arousal"] is None:
                continue

            whisper_path = os.path.join(output_dir, f"{basename}.whisper")
            if os.path.exists(whisper_path):
                filelist_path = f"{whisper_path}|{item['valence']}|{item['arousal']}"
                filelist_entries.append(filelist_path)
                continue

            whisper_features = self.feature_extractor.extract_features(path).squeeze(0)
            
            torch.save(whisper_features, whisper_path)

            filelist_path = f"{whisper_path}|{item['valence']}|{item['arousal']}"
            filelist_entries.append(filelist_path)
        return filelist_entries

if __name__ == '__main__':
    preprocessor = Preprocessor()

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("json_paths", type=str, nargs="+")
    args = parser.parse_args()

    preprocessor.preprocess_multi_json(args.json_paths, args.output_dir)