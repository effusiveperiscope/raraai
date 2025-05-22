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
    parser.add_argument("--input_audio" , type=str, default='test_id.wav')
    parser.add_argument("--try_generate", action="store_true", default=False)
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
    config.features.want = ["content_tokens", "content_interp_pitch", "acoustic_codes"]
    myfeatures = MyFeatures(config, "cuda")
    features = myfeatures.extract_features(args.input_audio)

    with torch.no_grad():
        import librosa
        wav_16k, _ = librosa.load(args.input_audio, sr=16000)
        len_wav = wav_16k.shape[0] / 16000
        predicted_token_count = len_wav * (360 / 3.2335)

        output = model(
            content_tokens=features["content_tokens"].to("cuda").unsqueeze(2), # Content tokens from VeVo
            content_interp_pitch=features["content_interp_pitch"].to("cuda").unsqueeze(2), # Pitch from RMVPE interpolated to content dim
            content_seq_lens=torch.Tensor([features["content_tokens"].shape[1]]).long().to("cuda").unsqueeze(0), # Content sequence length
            acoustic_codes=features["acoustic_codes"].to("cuda"),
            acoustic_codes_lens=torch.Tensor([features["acoustic_codes"].shape[1]]).long().to("cuda"),
        )
        print("loss:", output.loss.cpu().item())

        # sanity check: try with meaningless content
        output = model(
            content_tokens=torch.zeros_like(features["content_tokens"].to("cuda").unsqueeze(2)), # Content tokens from VeVo
            content_interp_pitch=torch.zeros_like(features["content_interp_pitch"].to("cuda").unsqueeze(2)), # Pitch from RMVPE interpolated to content dim
            content_seq_lens=torch.Tensor([features["content_tokens"].shape[1]]).long().to("cuda").unsqueeze(0), # Content sequence length
            acoustic_codes=features["acoustic_codes"].to("cuda"),
            acoustic_codes_lens=torch.Tensor([features["acoustic_codes"].shape[1]]).long().to("cuda"),
        )
        print("sanity check loss:", output.loss.cpu().item())

        if args.try_generate:
            predicted = model.generate(
                features["content_tokens"].to("cuda").unsqueeze(2), # Content tokens from VeVo
                features["content_interp_pitch"].to("cuda").unsqueeze(2), # Pitch from RMVPE interpolated to content dim
                torch.Tensor([features["content_tokens"].shape[1]]).long().to("cuda").unsqueeze(0), # Content sequence length
                max_len=int(predicted_token_count),
            ) # ((b d), t)
            predicted = predicted.transpose(0, 1).cpu() # (t, (b, d))

            gt = features["acoustic_codes"].squeeze(0).cpu()
            min_seq_len = min(predicted.shape[0], gt.shape[0])
            predicted = predicted[:min_seq_len, :]
            gt = gt[:min_seq_len, :]

            import matplotlib.pyplot as plt
            import seaborn as sns
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            sns.heatmap(gt, ax=ax[0], cmap="viridis")
            ax[0].set_title("Ground truth")
            sns.heatmap(predicted, ax=ax[1], cmap="viridis")
            ax[1].set_title("Predicted")
            plt.show()

            # sanity check: try with meaningless content
            # sanity = model.generate(
                # torch.zeros_like(features["content_tokens"].to("cuda").unsqueeze(2)), # Content tokens from VeVo
                # torch.zeros_like(features["content_interp_pitch"].to("cuda").unsqueeze(2)), # Pitch from RMVPE interpolated to content dim
                # torch.Tensor([features["content_tokens"].shape[1]]).long().to("cuda").unsqueeze(0), # Content sequence length
                # max_len=int(predicted_token_count),
            # ) # ((b d), t)
            # sanity = sanity.transpose(0, 1).cpu() # (t, (b, d))
            # 
            # sanity = sanity[:min_seq_len, :]
            # import matplotlib.pyplot as plt
            # import seaborn as sns
            # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            # sns.heatmap(predicted, ax=ax[0], cmap="viridis")
            # ax[0].set_title("Predicted")
            # sns.heatmap(sanity, ax=ax[1], cmap="viridis")
            # ax[1].set_title("Sanity check")
            # plt.show()