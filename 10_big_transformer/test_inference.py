from omegaconf import OmegaConf
import torch
from model import MyModel
import os


import sys
import pdb
import dac
def info(type, value, tb):
    # Automatically start pdb on any uncaught exception
    import traceback
    traceback.print_exception(type, value, tb)
    pdb.post_mortem(tb)

sys.excepthook = info


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/common.yaml')
    parser.add_argument("--ckpt" , type=str, default=None, required=True)
    parser.add_argument("--input_audio" , type=str, default='test.wav')
    args = parser.parse_args()

    if not os.path.exists(args.input_audio):
        raise Exception(f"File {args.input_audio} does not exist")

    config = OmegaConf.load(args.config)

    model = MyModel(config)
    model.eval()
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
    model.to("cuda")

    from features import MyFeatures
    config.features.want = ["content_tokens", "content_interp_pitch"]
    myfeatures = MyFeatures(config, "cuda")
    features = myfeatures.extract_features(args.input_audio)
    # features["content_tokens"]
    # features["content_interp_pitch"]

    with torch.no_grad():
        import librosa
        wav_16k, _ = librosa.load(args.input_audio, sr=16000)
        len_wav = wav_16k.shape[0] / 16000
        predicted_token_count = len_wav * (360 / 3.2335)

        predicted = model.generate(
            features["content_tokens"].to("cuda").unsqueeze(2), # Content tokens from VeVo
            features["content_interp_pitch"].to("cuda").unsqueeze(2), # Pitch from RMVPE interpolated to content dim
            torch.Tensor([features["content_tokens"].shape[1]]).long().to("cuda").unsqueeze(0), # Content sequence length
            max_len=int(predicted_token_count)
        )

        model_path = dac.utils.download(model_type="44khz")
        dac_model = dac.DAC.load(model_path)
        dac_model = dac_model.eval().cuda()
        latents, _, _ = dac_model.quantizer.from_codes(predicted.unsqueeze(0).long()) # the docstring for this method is wrong
        pred_audio = dac_model.decode(latents)

        import soundfile as sf
        sf.write("pred.wav", pred_audio[0].cpu().squeeze().numpy(), 44100)