"""Shared model/feature loading for inference.

The important bit: `in_channels` is read from the checkpoint instead of trusted
from the config, and the matching whisper model is picked from that width. This
is what prevents the trap where an eval script builds a whisper-small (768)
model around a whisper-base (512) checkpoint, silently skips `in_proj`
(load_state_dict_mismatch only prints) and then reports noise.
"""
import torch
from omegaconf import OmegaConf

from commons import infer_in_channels, load_submodule_prefix
from dataset import WhisperContext, normalize_wave
from model import IntensityModel

WHISPER_BY_DIM = {
    512: 'openai/whisper-base',
    768: 'openai/whisper-small',
    1024: 'openai/whisper-medium',
    1280: 'openai/whisper-large-v2',
}

def load_intensity_checkpoint(ckpt_path, map_location='cpu'):
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    if 'state_dict' not in ckpt:
        raise ValueError(f'{ckpt_path} has no state_dict (not a lightning checkpoint?)')
    return ckpt

def load_model_and_context(ckpt_path, config_path='config/base.yaml',
                          whisper_model=None, device=None, dtype=torch.bfloat16,
                          verbose=True):
    """Load (model, whisper_context) with shapes guaranteed to match the ckpt."""
    config = OmegaConf.load(config_path) if isinstance(config_path, str) else config_path
    ckpt = load_intensity_checkpoint(ckpt_path)
    state = ckpt['state_dict']

    in_channels = infer_in_channels(state)
    configured = int(config.model.get('in_channels', in_channels))
    if configured != in_channels:
        print(f"[inference] NOTE: config says in_channels={configured} but the checkpoint "
              f"was trained with {in_channels}; using the checkpoint value.")
    config.model.in_channels = in_channels

    if whisper_model is None:
        whisper_model = WHISPER_BY_DIM.get(in_channels)
        if whisper_model is None:
            raise ValueError(f'No known whisper model for in_channels={in_channels}; '
                             f'pass whisper_model explicitly.')
    if verbose:
        print(f"[inference] features={whisper_model} (d_model={in_channels}), "
              f"ckpt={ckpt_path} (epoch={ckpt.get('epoch')}, step={ckpt.get('global_step')})")

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    context = WhisperContext(whisper_model)
    d_model = int(context.model.config.d_model)
    if d_model != in_channels:
        raise ValueError(
            f'{whisper_model} produces {d_model}-dim features but the checkpoint '
            f'expects {in_channels}; pass the matching --whisper_model')
    model = IntensityModel(**config.model).to(device)
    load_submodule_prefix(model, 'model.', state, strict=True)  # raises on mismatch
    model = model.to(dtype).eval()

    return model, context, config

def interp2(context, feats):
    """Time-upsample whisper encoder features to 10 ms frames."""
    return context.interp2(feats)

def load_audio(path, sr=16000):
    """Load a mono float32 waveform at `sr`.

    Accepts preprocessed .pt tensors (as written by preprocess.py) as well as
    anything librosa can read.
    """
    import numpy as np
    if str(path).lower().endswith('.pt'):
        wave = torch.load(path, map_location='cpu', weights_only=False)
        wave = np.asarray(wave, dtype=np.float32).squeeze()
        return wave, sr
    import librosa
    wave, _ = librosa.load(path, sr=sr)
    if wave.ndim > 1:
        wave = wave.mean(-1)
    return wave, sr

def extract_interp_features(context, waves, device=None, dtype=torch.bfloat16):
    """waves: list of mono np arrays at 16 kHz (already normalised).

    Returns (features (B, T_max, D) at 10 ms/frame, mask (B, T_max) long).
    """
    from commons import sequence_mask
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    feats, feature_len = context.extract_features_batched(waves)
    feats = feats.to(device)
    feats = context.interp2(feats.to(dtype))
    mask = sequence_mask(feature_len).to(torch.long).to(device)
    # feature_len is already in 10 ms frames == length after interp2
    feats = feats[:, :int(feature_len.max()), :]
    mask = mask[:, :feats.shape[1]]
    return feats, mask

def score_features(model, features, mask):
    """features (B, T, D) 10 ms frames, mask (B, T) -> pooled intensity in 0..1,
    per-frame intensity (B, T), attention (B, T)."""
    with torch.no_grad():
        intensity, attn = model(features, mask)
    intensity = intensity[..., 0]
    attn = attn[..., 0]
    return (intensity * attn).sum(-1), intensity, attn
