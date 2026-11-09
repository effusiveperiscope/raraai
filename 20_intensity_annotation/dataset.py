import datatia as dt
import numpy as np
import torch
import librosa
import torch.nn.functional as F
from einops import rearrange
from transformers import WhisperFeatureExtractor, WhisperModel

def normalize_wave(wave):
    """Peak-normalize. Intensity is deliberately loudness-independent, so the
    model should not see absolute level."""
    return librosa.util.normalize(wave)

class LiveDataContext:
    def __init__(self, whisper_model = 'openai/whisper-base'):
        # imported lazily so inference-only scripts don't need the augmentation stack
        from svc_helper.augmentation.pedalboard import PedalboardRandomAugmentor
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
        row['wave'] = normalize_wave(row['wave']) # normalize
        row['wave'] = self.augmentor.process(row['wave'], 16000) # augment
        row['wave'] = self.extract_features(
            row['wave'] # problem here?
            ).squeeze(0) # extract feat
        return row

class WhisperContext:
    """Frozen whisper encoder used as a feature extractor.

    Length convention: `feature_len` counts **10 ms output frames** (mel frames),
    i.e. it is the length of the feature sequence *after* `interp2` (which
    doubles whisper's 20 ms encoder frames). Use it directly for
    `sequence_mask(feature_len)` on `interp2(...)` output.
    """
    def __init__(self, whisper_model = 'openai/whisper-small'):
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
        # from PyQt5.QtCore import pyqtRemoveInputHook
        # pyqtRemoveInputHook()
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            encoder_outputs = self.model.encoder(input_features)
            encoder_features = encoder_outputs.last_hidden_state
            feature_len = inputs.attention_mask.sum(-1) 
        return encoder_features, feature_len

    def interp2(self, tensor):
        """Upsample time by 2 (20 ms encoder frames -> 10 ms output frames).

        Accepts (T, D), (B, T, D) or a legacy (1, B, T, D) call and returns
        (B, 2T, D), so batch size 1 works.
        """
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        tensor = rearrange(tensor, "b t d -> b d t")
        tensor = F.interpolate(tensor, scale_factor=2)
        tensor = rearrange(tensor, "b d t -> b t d")
        return tensor

class PedalboardContext:
    def __init__(self, augment: bool = True):
        self.augmentor = None
        if augment:
            # imported lazily so inference-only scripts don't need the augmentation stack
            from svc_helper.augmentation.pedalboard import PedalboardRandomAugmentor
            self.augmentor = PedalboardRandomAugmentor(
                {'base': 0.5,
                'comp_gentle': 0.5, 'comp_hard': 0.5, 'limit': 0.2,
                'resample_8k': 0, 'resample_16k': 0,
                'resample_22k': 0, 'resample_24k': 0,
                'bitcrush_8': 0, 'mp3_vbr2': 0, 'mp3_vbr0': 0}
            )

    def process_wave(self, wave, augment: bool = True):
        wave = wave.squeeze()
        wave = normalize_wave(wave) # normalize
        if augment and self.augmentor is not None:
            wave = self.augmentor.process(wave, 16000) # augment
        return wave


def process_row(row):
    # NOTE: datatia's collate_fn requires '__filelist_entry__' to survive this
    # live map, so carry it over instead of building a fresh dict.
    return {
        **{k: v for k, v in row.items() if k == '__filelist_entry__'},
        'wave': row['wave'].squeeze(),
        'intensity': row['intensity'],
    }

def dataset2(filelist, is_train: bool, max_crop_sec: float = 2.0): # Provide raw wave only
    """Clip-level dataset.

    is_train is forwarded to datatia: on validation the random time-subsample is
    replaced by a deterministic crop from index 0 (previously it was ignored, so
    validation was randomly cropped and the val loss was noisy).

    max_crop_sec / training applies a random crop of at most this many seconds;
    train.py further randomises the *length* inside that window so the model
    becomes length-robust (it is scored on 0.5-4 s windows at inference).
    """
    return dt.Dataset(
        filelist=filelist,
        is_train=is_train,
        field_specs=[
            dt.FieldSpec(name='wave', datatype=torch.Tensor,
                dim=torch.Size([-1]), 
                keep_in_memory=True, # because this is hand annotated we won't have much
                provide_length=True),
            dt.FieldSpec(name='intensity', datatype=int),
        ],
        actions=[
            dt.LiveMapRow(operation=process_row),
            dt.RandomSubsample(fields=['wave'], dims=[0], length=int(max_crop_sec*16000)),
            dt.PadGroup(fields=['wave'], dims=[0], values=[0]),
        ]
    )
