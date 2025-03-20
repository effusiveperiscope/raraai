from svc_helper.svc.rvc import RVCModel
from huggingface_hub import hf_hub_download
from model import create_model
import argparse
import yaml
import soundfile as sf
import torch
import os

def test():
    if not os.path.exists('tests'):
        os.makedirs('tests')
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="config")
    parser.add_argument("--cycle", help="RVC model", action="store_true")
    parser.add_argument("--output_files", help="Whether to output test files", action="store_true")
    parser.add_argument("--transpose", help="Pitch transpose", default=0)
    parser.add_argument("input_file", help="input file")
    args = parser.parse_args()

    test_model_path = hf_hub_download(repo_id='therealvul/RVCv2',
        filename='Rarity-Titan/RarityTitan.pth')
    test_index_path = hf_hub_download(repo_id='therealvul/RVCv2',
        filename='Rarity-Titan/added_IVF3933_Flat_nprobe_1_RarityTitan_v2.index')
    spk_id = 2 # Rarity

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_path = 'best_voice_conversion_model2.pt'
    model = create_model(config)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.load_state_dict(torch.load(model_path)) # LOL I never loaded the weights

    rvc_model = RVCModel()
    rvc_model.load_model(model_path = test_model_path,
        index_path = test_index_path)

    def feature_transform(t):
        orig_dtype = t.dtype
        t = t.to(torch.float) # Use full precision for model for now
        print(t.shape)
        encoded = model.encode(t)
        #import pdb; pdb.set_trace()
        output = model.decode(encoded, speaker_id=torch.tensor(spk_id).unsqueeze(0).to(device))
        output = output.to(orig_dtype)

        mae_loss = torch.mean(torch.abs(t - output))
        print(f"MAE Loss: {mae_loss.item()}")

        return output

    def feature_cycle(t):
        orig_dtype = t.dtype
        t = t.to(torch.float)
        encoded = model.encode(t)
        output = model.decode(encoded, speaker_id=torch.tensor(0).unsqueeze(0).to(device))
        encoded = model.encode(output)
        output = model.decode(encoded, speaker_id=torch.tensor(spk_id).unsqueeze(0).to(device))
        output = output.to(orig_dtype)
        return output

    # 0. Base case
    opt = rvc_model.infer_file(args.input_file, transpose=args.transpose)
    if args.output_files:
        sf.write('tests/test_base_case.wav', opt, rvc_model.output_sample_rate())

    # 1. Reconstruction/conversion test
    opt = rvc_model.infer_file(args.input_file, transpose=args.transpose,
        extra_hooks={
            'feature_transform': feature_transform,
        })
    if args.output_files:
        sf.write('tests/test_reconstruction.wav', opt, rvc_model.output_sample_rate())

    # 2. Cycle test
    if args.cycle:
        opt = rvc_model.infer_file(args.input_file, transpose=args.transpose,
            extra_hooks={
                'feature_transform': feature_cycle,
            })
        if args.output_files:
            sf.write('tests/test_cycle.wav', opt, rvc_model.output_sample_rate())

if __name__ == '__main__':
    test()