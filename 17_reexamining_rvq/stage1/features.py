import librosa
import torch
from transformers import WhisperFeatureExtractor, WhisperModel

class FeatureExtractor:
    def __init__(self, whisper_model = 'openai/whisper-base'):
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model)
        self.device = 'cuda'
        self.model = WhisperModel.from_pretrained(whisper_model).to(self.device)
        del self.model.decoder
        self.model.eval()

    def extract_features(self, audio_path):
        audio_16k, sr = librosa.load(audio_path, sr=16000)
        inputs = self.feature_extractor(
            audio_16k, sampling_rate=sr, return_tensors="pt",
            return_attention_mask=True
        )
        input_features = inputs.input_features.to(self.device)
        with torch.no_grad():
            encoder_outputs = self.model.encoder(input_features)
            encoder_features = encoder_outputs.last_hidden_state
            feature_len = inputs.attention_mask.sum(-1).item() / 2 # divide by 2 to get feature length
            feature_len = int(feature_len)
            encoder_features = encoder_features[:, :feature_len, :]
        return encoder_features

if __name__ == '__main__':
    extractor = FeatureExtractor()
    features = extractor.extract_features('test.wav')
    print(features.shape)