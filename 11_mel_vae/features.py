import os
from pathlib import Path
import librosa
import torch
from nsfhifigan.wav2mel import PitchAdjustableMelSpectrogram
from nsfhifigan.config_utils import read_full_config
from svc_helper.pitch.rmvpe import RMVPEModel
from omegaconf import OmegaConf
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Tensorboard will log many extraneous warnings

class MyFeatures:
    def __init__(self, config=None):
        if config is None:
            config = OmegaConf.load("configs/base.yaml")

        # --- 1. Read NSF-HiFiGAN config ---
        try:
            nsfhifigan_config = read_full_config(
                Path(config.train.nsfhifigan_config))
            print("  NSF-HiFiGAN config loaded successfully.")
        except Exception as e:
            print(f"Error reading NSF-HiFiGAN config {config.train.nsfhifigan_config}: {e}")
            raise # Re-raise the exception as this is critical

        # --- 2. Create Mel Spectrogram Transform ---
        # Ensure parameters exist in the loaded config
        required_keys = ['audio_sample_rate', 'fft_size', 'win_size', 'hop_size', 'fmin', 'fmax', 'audio_num_mel_bins']
        for key in required_keys:
            if key not in nsfhifigan_config:
                raise ValueError(f"Missing required key '{key}' in nsfhifigan config.")

        self.mel_spec_transform = PitchAdjustableMelSpectrogram(
            sample_rate=nsfhifigan_config['audio_sample_rate'],
            n_fft=nsfhifigan_config['fft_size'],
            win_length=nsfhifigan_config['win_size'],
            hop_length=nsfhifigan_config['hop_size'],
            f_min=nsfhifigan_config['fmin'],
            f_max=nsfhifigan_config['fmax'],
            n_mels=nsfhifigan_config['audio_num_mel_bins'],
        )
        print("  Mel Spectrogram Transform created.")

        # --- 3. Initialize RMVPE Model ---
        self.rmvpe_model = RMVPEModel(
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            hop_length=nsfhifigan_config['hop_size'],
        )
        print("  RMVPE Model initialized.")

        self.nsfhifigan_config = nsfhifigan_config
        self.config = config

    def extract_features(self, audio : str | np.ndarray):
        if type(audio) is str:
            audio, sr = librosa.load(audio, sr=
                self.nsfhifigan_config['audio_sample_rate'], mono=True)
        if self.config.train.normalize_audio:
            audio = librosa.util.normalize(audio) * 0.95
        audio = torch.from_numpy(audio).unsqueeze(0).to(torch.float32) # [1, T]

        # Extract mel spectrogram
        mel_spec = self.mel_spec_transform(audio) # [1, D, T]

        # Extract pitch
        f0 = torch.from_numpy(self.rmvpe_model.extract_pitch(audio.squeeze(0))).unsqueeze(0) # [1, T]

        # Scale pitch
        # because this was extracted from a 44.1khz audio but the model expects 16khz:
        f0 = f0 * (self.nsfhifigan_config['audio_sample_rate']/RMVPEModel.expected_sample_rate)

        f0 = f0[:, :mel_spec.shape[-1]]
        mel_spec = mel_spec[:, :, :f0.shape[-1]]

        return mel_spec, f0

if __name__ == '__main__':
    myfeatures = MyFeatures()
    mel_spec, f0 = myfeatures.extract_features("pretests/test1.wav")
    print(mel_spec.shape, f0.shape)
    