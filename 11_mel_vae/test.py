from typing import OrderedDict
from features import MyFeatures
from models.model_flowvaegan import MelFlowVAEGAN
from omegaconf import OmegaConf
from einops import rearrange
import torch
import torch.nn.functional as F
import soundfile as sf

from nsfhifigan.models import Generator, AttrDict
from nsfhifigan.config_utils import read_full_config
from nsfhifigan.wav2mel import PitchAdjustableMelSpectrogram
from pathlib import Path
import common

test_audio = "pretests/unseen2.flac"
test_audio_basename = Path(test_audio).stem+"gan"
config = "configs/vaegan_v2.yaml"
model_ckpt = "checkpoints/last.ckpt"

nsfhifigan_model_ckpt = r"D:\Code\SingingVocoders\pc_nsf_hifigan_44.1k_hop512_128bin_2025.02\model.ckpt"

config = OmegaConf.load(config)
myfeatures = MyFeatures(config)

nsfhifigan_config = myfeatures.nsfhifigan_config
nsfh_model_cfg = nsfhifigan_config['model_args']
nsfh_model_cfg.update({
    'sampling_rate': nsfhifigan_config['audio_sample_rate'],
    'num_mels': nsfhifigan_config['audio_num_mel_bins'],
    'hop_size': nsfhifigan_config['hop_size'],
})
generator = Generator(AttrDict(nsfh_model_cfg)).to("cuda")
generator.load_state_dict(
    torch.load(nsfhifigan_model_ckpt, map_location="cuda")["generator"])
generator.eval()

model = MelFlowVAEGAN(config)
state_dict = torch.load(model_ckpt, weights_only=False)['state_dict']
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if k.startswith("model."):
        new_key = k[len("model."):]
        new_state_dict[new_key] = v
model.load_state_dict(new_state_dict)

mel_spec, f0 = myfeatures.extract_features(test_audio)
mel_spec = rearrange(mel_spec, 'b c n -> b n c')
f0 = f0.to(mel_spec.dtype)


# 1. Pad mel_spec to a sequence multiple of config.model.sampling_ratio
seq_len = mel_spec.shape[1]
sampling_ratio = config.model.sampling_ratio
pad_len = (sampling_ratio - seq_len % sampling_ratio) % sampling_ratio  # no padding if already a multiple

if pad_len > 0:
    mel_spec = F.pad(mel_spec, (0, 0, 0, pad_len), mode='constant', value=0)
    f0 = F.pad(f0, (0, pad_len), mode='constant', value=0)

# 2. Create an appropriate sequence mask
seq_mask = torch.arange(mel_spec.shape[1])[None, :].to(mel_spec.device) < seq_len
log_mel_spec = PitchAdjustableMelSpectrogram.dynamic_range_compression_torch(mel_spec)
norm_mel_spec = common.normalize(config, log_mel_spec)
with torch.no_grad():
    x_recon, z_mean, z_log_var, z, z_transformed, log_det, z_mask = model(x=
        norm_mel_spec, x_mask=seq_mask.bool(), pitch=f0)
    # outputs log spec - x_recon is already log

# 3. Test against vocoder
x_recon = x_recon.to('cuda')
f0 = f0.to('cuda')

# Denormalize output
x_recon = common.denormalize(config, x_recon)

# takes log specs as input
with torch.no_grad():
    orig_wav = generator(x=rearrange(log_mel_spec, 'b n c -> b c n').to('cuda'), f0=f0)
    recon_wav = generator(x=rearrange(x_recon, 'b n c -> b c n'), f0=f0)

sf.write(f"pretests/{test_audio_basename}_orig.wav", 
    orig_wav.squeeze().cpu().numpy(), nsfhifigan_config['audio_sample_rate'],
    subtype='FLOAT')
sf.write(f"pretests/{test_audio_basename}_recon.wav", 
    recon_wav.squeeze().cpu().numpy(), nsfhifigan_config['audio_sample_rate'],
    subtype='FLOAT')
