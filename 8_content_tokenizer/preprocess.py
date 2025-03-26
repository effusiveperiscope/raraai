# Input: filelist pointing to wavs
# Output:
# - Target RVC Features for padded audio
# - Content tokens for padded audio
import argparse
import os
from content import ContentTokenizer
from svc_helper.sfeatures.models import RVCHubertModel
from tqdm import tqdm
import librosa
import torch

def win_longpath(path):
    return '\\\\?\\' + os.path.abspath(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filelist", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--multispk", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.filelist):
        raise ValueError(f"{args.filelist} does not exist")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    wav_infos = []
    with open(args.filelist, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not args.multispk:
                wav_path = line
                wav_name = os.path.basename(wav_path)
                wav_infos.append({
                    "wav_path": wav_path,
                    "wav_name": wav_name
                })
            else:
                wav_path, spk = line.split("|")
                wav_name = os.path.basename(wav_path)
                wav_infos.append({
                    "wav_path": wav_path,
                    "wav_name": wav_name,
                    "spk": spk
                })

    print(f"Found {len(wav_infos)} wavs")

    model = ContentTokenizer()
    rvc_model = RVCHubertModel(device="cuda" if torch.cuda.is_available() else "cpu")

    out_filelist = ""
    for wav_info in tqdm(wav_infos, total=len(wav_infos), desc="Preprocessing"):
        data, rate = librosa.load(
            win_longpath(wav_info["wav_path"]), sr=16000) # [T]
        data_padded = rvc_model.pad_audio(data)

        embed_out_path = win_longpath(os.path.join(args.output_dir, wav_info["wav_name"] + ".embed.pt"))
        hubert_feat_out_path = win_longpath(os.path.join(args.output_dir, wav_info["wav_name"] + ".feat.pt"))
        if os.path.exists(embed_out_path) and os.path.exists(hubert_feat_out_path):
            continue

        with torch.no_grad():
            embed, _, _ = model.extract_hubert_codes(torch.from_numpy(data_padded)
                .unsqueeze(0).to(model.device).to(torch.float32))
            hubert_feat = rvc_model.extract_features(torch.from_numpy(data_padded))

        torch.save(embed, embed_out_path)
        torch.save(hubert_feat, hubert_feat_out_path)

        if args.multispk:
            out_filelist += f"{wav_info['wav_path']}|{wav_info['spk']}|{embed_out_path}|{hubert_feat_out_path}\n"
        else:
            out_filelist += f"{wav_info['wav_path']}|{embed_out_path}|{hubert_feat_out_path}\n"
    
    with open(os.path.join(args.output_dir, "filelist.list"), "w", encoding="utf-8") as f:
        f.write(out_filelist)

    print("Done")