from transformers import WhisperFeatureExtractor, WhisperModel
from speechbrain.inference.text import GraphemeToPhoneme
import numpy
import torch

class MyFeatures:
    expected_sample_rate = 16000

    def __init__(self, no_whisper=False):
        if not no_whisper:
            self.init_whisper()
        self.init_g2p()

    def init_whisper(self):
        model_name = "openai/whisper-base"
        feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        model = WhisperModel.from_pretrained(model_name)
        model.eval()
        self.feature_extractor = feature_extractor
        self.device = "cuda"
        self.model = model.to(self.device)
        del self.model.decoder

    def init_g2p(self):
        self.g2p = GraphemeToPhoneme.from_hparams("speechbrain/soundchoice-g2p", savedir="pretrained_models/soundchoice-g2p")
        self.g2p_phones_to_ids = {
            phone: i for i, phone in enumerate(self.g2p.phonemes)
        }

    def all_phonemes(self):
        return self.g2p.phonemes

    def get_phonemes(self, text: str):
        return self.g2p(text)

    def get_phonemes_ids(self, text: str):
        return [self.g2p_phones_to_ids[phone] for phone in self.get_phonemes(text)]

    def ids_to_phonemes(self, ids):
        return [self.g2p.phonemes[i] for i in ids]

    def get_whisper_features(self, audio: numpy.ndarray):
        inputs = self.feature_extractor(audio, 
            sampling_rate=MyFeatures.expected_sample_rate, return_tensors="pt",
            return_attention_mask=True)
        input_features = inputs.input_features.to(self.device)
        with torch.no_grad():
            hidden_states = self.model.encoder(input_features).last_hidden_state
            feature_len = inputs.attention_mask.sum(-1).item() / 2 # divide by 2 to get feature length
            feature_len = int(feature_len)
            hidden_states = hidden_states[:, :feature_len, :]
        return hidden_states

if __name__ == "__main__":
    import librosa
    features = MyFeatures()
    audio, _ = librosa.load("test1.wav", sr=MyFeatures.expected_sample_rate)
    test_text = "You wanted to see me? To give me a test? Equestria. Griffons. Griffonia. Ooh. Oh."
    phones = features.get_phonemes(test_text)
    print(features.get_whisper_features(audio).shape)
    print(phones)
    print(features.get_phonemes_ids(test_text))