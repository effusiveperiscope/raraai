from nsfhifigan.wav2mel import PitchAdjustableMelSpectrogram
from nsfhifigan.config_utils import read_full_config
from pathlib import Path
import librosa
import torch

config = read_full_config(Path('nsfhifigan/ft_hifigan.yaml'))
wav, sr = librosa.load('pretests/test1.wav', sr=44100, mono=True)
mel_spec_transform=PitchAdjustableMelSpectrogram(
    sample_rate=config['audio_sample_rate'],
    n_fft=config['fft_size'],
    win_length=config['win_size'],
    hop_length=config['hop_size'],
    f_min=config['fmin'],
    f_max=config['fmax'],
    n_mels=config['audio_num_mel_bins'],
)
with torch.no_grad():
    spectrogram = mel_spec_transform(torch.FloatTensor(wav).unsqueeze(0).cuda())
    print(spectrogram.shape)