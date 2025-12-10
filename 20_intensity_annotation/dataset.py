import datatia as dt
import numpy as np
import torch
import librosa
import torch.nn.functional as F
from einops import rearrange
from transformers import WhisperFeatureExtractor, WhisperModel
from svc_helper.augmentation.pedalboard import PedalboardRandomAugmentor

class LiveDataContext:
    def __init__(self, whisper_model = 'openai/whisper-base'):
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model)
        self.device = 'cuda'
        self.augmentor = PedalboardRandomAugmentor(
            {'base': 0.5,
            'comp_gentle': 0.5, 'comp_hard': 0.5, 'limit': 0.2,
            'resample_8k': 0, 'resample_16k': 0,
            'resample_22k': 0, 'resample_24k': 0,
            'bitcrush_8': 0, 'mp3_vbr2': 0, 'mp3_vbr0': 0}
        )
        self.model = WhisperModel.from_pretrained(whisper_model).to(self.device)
        self.model = self.model.half()
        del self.model.decoder
        self.model.eval()

    def extract_features(self, wave):
        inputs = self.feature_extractor(
            wave, sampling_rate=16000, return_tensors="pt",
            return_attention_mask=True
        )
        input_features = inputs.input_features.to(self.device).half()
        with torch.no_grad():
            encoder_outputs = self.model.encoder(input_features)
            encoder_features = encoder_outputs.last_hidden_state
            feature_len = inputs.attention_mask.sum(-1).item() / 2 # divide by 2 to get feature length
            feature_len = int(feature_len)
            encoder_features = encoder_features[:, :feature_len, :]
        return encoder_features

    def extract_features_batched(self, wave_batch): # TODO output the lengths instead of mask
        inputs = self.feature_extractor(
            wave_batch, sampling_rate=16000, return_tensors="pt",
            return_attention_mask=True
        )
        input_features = inputs.input_features.to(self.device).half()
        with torch.no_grad():
            encoder_outputs = self.model.encoder(input_features)
            encoder_features = encoder_outputs.last_hidden_state # divide by 2 to get feature length
            feature_len = inputs.attention_mask.sum(-1) / 2
        return encoder_features, feature_len

    def process_wave_action(self, row):
        row['wave'] = row['wave'].squeeze().numpy()
        row['wave'] = librosa.util.normalize(row['wave']) # normalize
        row['wave'] = self.augmentor.process(row['wave'], 16000) # augment
        row['wave'] = self.extract_features(
            row['wave'] # problem here?
            ).squeeze(0) # extract feat
        return row

class WhisperContext:
    def __init__(self, whisper_model = 'openai/whisper-base'):
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model)
        self.device = 'cuda'
        self.model = WhisperModel.from_pretrained(whisper_model).to(self.device)
        self.model = self.model.half()
        del self.model.decoder
        self.model.eval()

    def extract_features_batched(self, wave_batch):
        inputs = self.feature_extractor(
            wave_batch, sampling_rate=16000, return_tensors="pt",
            return_attention_mask=True
        )
        input_features = inputs.input_features.to(self.device).half()
        with torch.no_grad():
            encoder_outputs = self.model.encoder(input_features)
            encoder_features = encoder_outputs.last_hidden_state
            feature_len = inputs.attention_mask.sum(-1) 
        return encoder_features, feature_len

    def interp2(self, tensor):
        tensor = tensor.squeeze(0)
        tensor = rearrange(tensor, "b t d -> b d t")
        tensor = F.interpolate(tensor, scale_factor=2)
        tensor = rearrange(tensor, "b d t -> b t d")
        return tensor

class PedalboardContext:
    def __init__(self):
        self.augmentor = PedalboardRandomAugmentor(
            {'base': 0.5,
            'comp_gentle': 0.5, 'comp_hard': 0.5, 'limit': 0.2,
            'resample_8k': 0, 'resample_16k': 0,
            'resample_22k': 0, 'resample_24k': 0,
            'bitcrush_8': 0, 'mp3_vbr2': 0, 'mp3_vbr0': 0}
        )

    def process_wave(self, wave):
        wave = wave.squeeze()
        wave = librosa.util.normalize(wave) # normalize
        wave = self.augmentor.process(wave, 16000) # augment
        return wave


def process_row(row):
    return {'wave': row['wave'].squeeze(), 'intensity': row['intensity']}

def dataset2(filelist, is_train: bool): # Provide raw wave only
    return dt.Dataset(
        filelist=filelist,
        field_specs=[
            dt.FieldSpec(name='wave', datatype=torch.Tensor,
                dim=torch.Size([-1]), 
                keep_in_memory=True, # because this is hand annotated we won't have much
                provide_length=True),
            dt.FieldSpec(name='intensity', datatype=int),
        ],
        actions=[
            dt.LiveMapRow(operation=process_row),
            dt.RandomSubsample(fields=['wave'], dims=[0], length=320*100),
            dt.PadGroup(fields=['wave'], dims=[0], values=[0]),
        ]
    )