from svc5whisper.model import Whisper, ModelDimensions
from svc5whisper.audio import pad_or_trim, log_mel_spectrogram
from svc5hubert import hubert_model
from svc_helper.pitch.rmvpe import RMVPEModel
from svc_helper.speaker.models import SVC5SpeakerEncoder
from modeling.vits import spectrogram, utils
import torch
import librosa
import io
import soundfile as sf
from omegaconf import OmegaConf

class MyFeatures:
    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.dtype = torch.float16
        self.hps = OmegaConf.load("./configs/base.yaml").data

        self.load_whisper()
        self.load_hubert()
        self.load_rmvpe()
        self.svc5_spk_model = SVC5SpeakerEncoder(device=device)

    def load_whisper(self):
        print("loading whisper...")
        checkpoint = torch.load("pretrain/whisper-large-v2.pt", 
            map_location="cpu", weights_only=False)
        dims = ModelDimensions(**checkpoint["dims"])
        model = Whisper(dims)
        del model.decoder
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.eval()
        model.to(self.dtype)
        model.to(self.device)
        self.whisper_model = model

    def extract_whisper(self, wav_16k):
        assert hasattr(self, "whisper_model")
        self.whisper_model
        
        audln = wav_16k.shape[0]
        ppgln = audln // 320
        audio = pad_or_trim(wav_16k)
        mel = log_mel_spectrogram(audio).to(self.whisper_model.device).to(
            self.dtype)
        with torch.no_grad():
            ppg = self.whisper_model.encoder(
                mel.unsqueeze(0)).squeeze().data.cpu()
            ppg = ppg[:ppgln,]
        return ppg

    def load_hubert(self):
        print("loading hubert...")
        model = hubert_model.hubert_soft("pretrain/hubert-soft-0d54a1f4.pt")
        model.eval()
        model.to(self.dtype)
        model.to(self.device)
        self.hubert_model = model

    def extract_hubert(self, wav_16k):
        feats = torch.from_numpy(wav_16k).to(self.device)
        feats = feats[None, None, :].to(self.dtype)
        with torch.no_grad():
            vec = self.hubert_model.units(feats).squeeze().data.cpu()
        return vec

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

    def extract_spec(self, wav, sr):
        audio_norm = librosa.util.normalize(wav)
        assert sr == self.hps.sampling_rate
        audio_norm = torch.from_numpy(audio_norm).unsqueeze(0)
        n_fft = self.hps.filter_length
        sampling_rate = self.hps.sampling_rate
        hop_size = self.hps.sampling_rate // 100
        win_size = self.hps.win_length
        spec = spectrogram.spectrogram_torch(
            audio_norm, n_fft, sampling_rate, hop_size, win_size, center=False)
        spec = torch.squeeze(spec, 0)
        return spec

    def extract_features(self, wav, sr):
        wav_16k = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        ppg = self.extract_whisper(wav_16k)
        vec = self.extract_hubert(wav_16k)

        f0, extras = self.extract_pitch(wav_16k)
        f0_inharm = torch.from_numpy(extras['inharmonic_confidence'])
        f0_subharm = torch.from_numpy(extras['subharmonic_confidence'])
        f0_confidence = torch.from_numpy(extras['confidence'])

        f0 = f0[:ppg.shape[0]*2]
        f0_inharm = f0_inharm[:ppg.shape[0]*2]
        f0_subharm = f0_subharm[:ppg.shape[0]*2]
        f0_confidence = f0_confidence[:ppg.shape[0]*2]

        spec = self.extract_spec(wav, sr)
        spec = spec.transpose(0, 1)[:ppg.shape[0]*2, :]

        virtfile = io.BytesIO()
        sf.write(virtfile, wav_16k, 16000, format='WAV', subtype='PCM_16')
        virtfile.seek(0)
        spk = self.svc5_spk_model.extract_feature(
            sf.SoundFile(virtfile)
        )

        return {
            'whisper': ppg,
            'hubert': vec,
            'f0': f0,
            'f0_inharm': f0_inharm,
            'f0_subharm': f0_subharm,
            'f0_confidence': f0_confidence,
            'spec': spec,
            'spk': spk,
            'wave': torch.from_numpy(wav)
        }

if __name__ == "__main__":
    feats = MyFeatures()
    wav, sr = librosa.load("test.wav", sr=48000)
    feats = feats.extract_features(wav, sr)
    for key, value in feats.items():
        print(f"{key} shape: {value.shape} {type(value)}")