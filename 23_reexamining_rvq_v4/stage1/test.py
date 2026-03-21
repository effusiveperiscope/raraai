from einops import rearrange
from omegaconf import OmegaConf
import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import soundfile as sf
sys.path.append('..')
from rvq.vevo_repcodec import VevoRepCodec
from commons import load_submodule_prefix
from features import FeatureExtractor
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/base.yaml')
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('audio', type=str)
args = parser.parse_args()

config = OmegaConf.load(args.config)
model = VevoRepCodec(
    input_channels=config.whisper_dim,
    output_channels=config.whisper_dim,
    encode_channels=config.whisper_dim,
    decode_channels=config.whisper_dim,
    code_dim=config.code_dim,
    codebook_num=1,
    codebook_size=config.codebook_size
)
feats = FeatureExtractor()
state = torch.load(args.ckpt, map_location='cpu', weights_only=False)['state_dict']
load_submodule_prefix(model, 'model.', state)
model.to('cuda')

whisper = feats.extract_features(args.audio)  # [1, T, C]
whisper_interp = F.interpolate(rearrange(whisper, "b t c -> b c t"), scale_factor = 2)
whisper_interp = rearrange(whisper_interp, "b c t -> b t c")
# -- These take INTERPOLATED inputs!
yq, y, _, _, vqloss, perplexity = model(whisper_interp)
q_recon_loss = F.mse_loss(yq, whisper_interp)
recon_loss = F.mse_loss(y, whisper_interp)
indices = model.forward_index(whisper_interp)         # [1, 1, T]

# --- Load raw waveform for visualization ---
waveform, sample_rate = sf.read(args.audio)
if waveform.ndim > 1:
    waveform = waveform.mean(axis=1)  # mix down to mono

# --- Prepare index data ---
idx = indices[0, 0].cpu().numpy()  # [T]
n_frames = len(idx)

# Build time axes
audio_duration = len(waveform) / sample_rate
waveform_times = np.linspace(0, audio_duration, len(waveform))
index_times    = np.linspace(0, audio_duration, n_frames)

# --- Plot ---
# Left column: waveform + indices stacked; right column: histogram
fig = plt.figure(figsize=(16, 7), facecolor='#0d1117')
gs = gridspec.GridSpec(
    2, 2,
    figure=fig,
    hspace=0.45,
    wspace=0.35,
    width_ratios=[3, 1]   # time panels get 3x the width of the histogram
)

# Waveform panel (top-left)
ax_wave = fig.add_subplot(gs[0, 0])
ax_wave.set_facecolor('#0d1117')
ax_wave.plot(waveform_times, waveform, color='#58a6ff', linewidth=0.6, alpha=0.85)
ax_wave.set_xlim(0, audio_duration)
ax_wave.set_ylabel('Amplitude', color='#8b949e', fontsize=10)
ax_wave.set_title('Waveform', color='#e6edf3', fontsize=11, loc='left', pad=6)
ax_wave.tick_params(colors='#8b949e', labelsize=8)
for spine in ax_wave.spines.values():
    spine.set_edgecolor('#30363d')

# Indices-over-time panel (bottom-left)
ax_idx = fig.add_subplot(gs[1, 0], sharex=ax_wave)
ax_idx.set_facecolor('#0d1117')
ax_idx.step(index_times, idx, where='mid', color='#3fb950', linewidth=0.9, alpha=0.9)
ax_idx.scatter(index_times, idx, color='#3fb950', s=2, alpha=0.5, zorder=3)
ax_idx.set_xlim(0, audio_duration)
ax_idx.set_ylim(-5, config.codebook_size + 5)
ax_idx.set_xlabel('Time (s)', color='#8b949e', fontsize=10)
ax_idx.set_ylabel('Codebook Index', color='#8b949e', fontsize=10)
ax_idx.set_title('VQ Codebook Indices', color='#e6edf3', fontsize=11, loc='left', pad=6)
ax_idx.tick_params(colors='#8b949e', labelsize=8)
for spine in ax_idx.spines.values():
    spine.set_edgecolor('#30363d')

# Histogram panel (right column, spans both rows)
ax_hist = fig.add_subplot(gs[:, 1])
ax_hist.set_facecolor('#0d1117')
counts, _, patches = ax_hist.hist(
    idx,
    bins=config.codebook_size,
    range=(0, config.codebook_size),
    orientation='horizontal',
    color='#bc8cff',
    alpha=0.85,
    edgecolor='none',
    rwidth=0.85
)
# Colour bars by frequency: dim unused, highlight hot ones
max_count = counts.max() if counts.max() > 0 else 1
for patch, count in zip(patches, counts):
    intensity = count / max_count
    patch.set_alpha(max(0.15, intensity))  # keep zero-use bins faintly visible

ax_hist.set_ylim(-5, config.codebook_size + 5)
ax_hist.set_xlabel('Count', color='#8b949e', fontsize=10)
ax_hist.set_ylabel('Codebook Index', color='#8b949e', fontsize=10)
ax_hist.set_title('Index Usage', color='#e6edf3', fontsize=11, loc='left', pad=6)
ax_hist.tick_params(colors='#8b949e', labelsize=8)
for spine in ax_hist.spines.values():
    spine.set_edgecolor('#30363d')

# Annotate utilisation rate
n_used = int((counts > 0).sum())
utilisation = 100 * n_used / config.codebook_size
ax_hist.text(
    0.97, 0.02,
    f"{n_used}/{config.codebook_size} used\n({utilisation:.1f}%)'"
    f"\nq_recon_loss: {q_recon_loss: .3f}, recon_loss: {recon_loss: .3f}"
    f"\nperplexity: {perplexity.item(): .3f}",
    transform=ax_hist.transAxes,
    ha='right', va='bottom',
    color='#bc8cff', fontsize=9,
    bbox=dict(facecolor='#161b22', edgecolor='#30363d', boxstyle='round,pad=0.4'),
)


fig.suptitle(args.audio, color='#8b949e', fontsize=9, y=0.99)
plt.savefig('indices_vs_waveform.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("Saved → indices_vs_waveform.png")