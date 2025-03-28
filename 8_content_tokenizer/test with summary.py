import argparse
from svc_helper.svc.rvc import RVCModel
from svc_helper.sfeatures.models import RVCHubertModel
from utils import subsample_features
from huggingface_hub import hf_hub_download
from model import TokenConvertModel
from omegaconf import OmegaConf
from preprocess import ContentTokenizer
import argparse
import soundfile as sf
import torch
import os
import librosa

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="config", default='config.yaml')
    parser.add_argument("--output_files", help="Whether to output test files", action="store_true")
    parser.add_argument("--transpose", help="Pitch transpose", default=0)
    parser.add_argument("--sid", help="speaker id", default=None)
    parser.add_argument("--model", help="model", default='checkpoints/test-rarity-with-summary/test-rarity-with-summary_75.pth')
    parser.add_argument("--ref_subsample", help="Randomly subsample style reference", default=False)
    parser.add_argument("input_file", help="input file")
    parser.add_argument("ref_file", help="audio reference for summary")
    args = parser.parse_args()
    if not os.path.exists(args.input_file):
        raise ValueError(f"{args.input_file} does not exist")
    test_model_path = hf_hub_download(repo_id='therealvul/RVCv2',
        filename='Rarity-Titan/RarityTitan.pth')
    test_index_path = hf_hub_download(repo_id='therealvul/RVCv2',
        filename='Rarity-Titan/added_IVF3933_Flat_nprobe_1_RarityTitan_v2.index')
    config = OmegaConf.load(args.config)
    model = TokenConvertModel()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.load_state_dict(torch.load(args.model)['model'])
    content_tokenizer = ContentTokenizer()
    rvc_model = RVCModel()
    rvc_model.load_model(model_path = test_model_path,
        index_path = test_index_path)
    if not os.path.exists('tests'):
        os.makedirs('tests')

    rvc_hubert = RVCHubertModel()

    ref_audio, sr = librosa.load(args.ref_file, sr=16000)
    ref_features = rvc_hubert.extract_features(ref_audio)
    if args.ref_subsample:
        ref_features = subsample_features(ref_features, 
            config.train.feat_summary_subsample)

    file_basename = os.path.basename(args.input_file).split('.')[0]

    # 0. Base case
    opt = rvc_model.infer_file(args.input_file, transpose=args.transpose)
    if args.output_files:
        sf.write(f'tests/{file_basename}_base_case.wav', opt, rvc_model.output_sample_rate())

    sid_to_use = (torch.tensor(int(args.sid)).to(content_tokenizer.device).unsqueeze(0)
        if args.sid is not None else None)

    # 1. Conversion test
    def feature_override(padded_audio):
        with torch.no_grad():
            embed, _, masks = content_tokenizer.extract_hubert_codes(
                padded_audio.unsqueeze(0).to(content_tokenizer.device)
                .to(torch.float32))
            summary = model.summarize(ref_features.to(content_tokenizer.device))
            feats = model(embed, masks, sid=sid_to_use, summary=summary)
        return feats

    opt = rvc_model.infer_file(
        args.input_file, transpose=args.transpose, extra_hooks={
            'feature_override': feature_override
        })
    if args.output_files:
        sf.write(f'tests/{file_basename}_conversion.wav', opt, rvc_model.output_sample_rate())
if __name__ == "__main__":    
    test()