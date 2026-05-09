import torch
import librosa
from svc_helper.pitch.rmvpe import RMVPEModel

class MyFeatures:
    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.dtype = torch.float16
        self.load_rmvpe()

    def load_rmvpe(self):
        print("loading rmvpe...")
        rmvpe = RMVPEModel(device=self.device, is_half=self.dtype == torch.float16)
        self.rmvpe = rmvpe

    def extract_pitch(self, wav_16k):
        f0_extracted, extras = self.rmvpe.extract_pitch2(
            torch.from_numpy(wav_16k), return_confidence=True, 
            return_subharmonic_confidence=True, 
            return_inharmonic_confidence=True)
        return torch.from_numpy(f0_extracted), extras

    def extract_features(self, wav, sr):
        wav_16k = librosa.resample(wav, orig_sr=sr, target_sr=16000)

        f0, extras = self.extract_pitch(wav_16k)
        f0_inharm = torch.from_numpy(extras['inharmonic_confidence'])
        f0_subharm = torch.from_numpy(extras['subharmonic_confidence'])
        f0_confidence = torch.from_numpy(extras['confidence'])

        return {
            'wave': torch.from_numpy(wav),
            'f0': f0,
            'f0_inharm': f0_inharm,
            'f0_subharm': f0_subharm,
            'f0_confidence': f0_confidence,
        }

    def expected_keys(self):
        return ['wave', 'f0', 'f0_inharm', 'f0_subharm', 'f0_confidence']