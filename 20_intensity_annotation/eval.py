import torch
from dataset import WhisperContext
from model import IntensityModel
from omegaconf import OmegaConf
import argparse
import librosa
import numpy as np
from commons import load_submodule_prefix, sequence_mask

if __name__ == '__main__':
    context = WhisperContext()
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_file', type=str)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--config', type=str, default='config/base.yaml')
    args = parser.parse_args()
    
    # Load audio
    audio, sr = librosa.load(args.audio_file, sr=16000)
    audio = librosa.util.normalize(audio)

    use_dtype = torch.bfloat16
    
    # Process through model
    feats, feat_lens = context.extract_features_batched([audio])
    feats = feats.to(use_dtype).to('cuda').unsqueeze(0)
    interp_feats = context.interp2(feats)
    feat_mask = sequence_mask(feat_lens).to(torch.long).to('cuda')
    interp_feats = interp_feats[:, :feat_lens.max(), :]

    config = OmegaConf.load(args.config)
    model = IntensityModel(**config.model).to('cuda').to(use_dtype)
    state = torch.load(args.ckpt, map_location='cpu', weights_only=False)['state_dict']
    load_submodule_prefix(model, 'model.', state)
    model.eval()

    with torch.no_grad():
        import pdb; pdb.set_trace()
        intensity_pred, attn = model(interp_feats, feat_mask)
    
    attn_feats = (intensity_pred * attn).cpu().detach().numpy()
    total_pred = (attn_feats.sum(axis=1)) * 8 + 1

    # Create time axes
    audio_time = np.arange(len(audio)) / sr
    feat_time = np.linspace(0, len(audio) / sr, attn_feats.shape[1])
    
    # Plotting
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot audio waveform
    ax1.plot(audio_time, audio, color='steelblue', linewidth=0.5, alpha=0.7)
    ax1.set_ylabel('Amplitude', fontsize=11)
    ax1.set_title('Audio Waveform', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, audio_time[-1])
    
    # Plot output features
    ax2.plot(feat_time, attn_feats.squeeze(), color='orangered', linewidth=2)
    ax2.set_xlabel('Time (seconds)', fontsize=11)
    ax2.set_ylabel('Intensity', fontsize=11)
    ax2.set_title(f'Predicted Subjective Intensity (total={total_pred})', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, feat_time[-1])

    # Plot attention features
    ax3 = ax2.twinx()
    ax3.plot(feat_time, attn.cpu().numpy().squeeze(), color='g', linewidth=2)
    ax3.set_ylabel('Attention', fontsize=11, color='g')
    ax3.tick_params('y', colors='g')

    plt.tight_layout()
    plt.show()