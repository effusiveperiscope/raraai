import datatia as dt
import numpy as np
import torch
import librosa
import torch.nn.functional as F
from einops import rearrange
from svc_helper.augmentation.pedalboard import PedalboardRandomAugmentor
from omegaconf import OmegaConf
from modeling.vits import utils, spectrogram

class LiveDataContext:
    def __init__(self, config : OmegaConf):
        self.device = 'cuda'
        self.augmentor = PedalboardRandomAugmentor(
            {'base': 0.5,
            'comp_gentle': 0.5, 'comp_hard': 0.5, 'limit': 0.2,
            'resample_8k': 0, 'resample_16k': 0,
            'resample_22k': 0, 'resample_24k': 0,
            'bitcrush_8': 0, 'mp3_vbr2': 0, 'mp3_vbr0': 0}
        )
        self.config = config

    def extract_spec(self, wave):
        # Mel spec
        hps = self.config.data
        n_fft = hps.filter_length
        sampling_rate = hps.sampling_rate
        hop_size = hps.hop_length
        win_size = hps.win_length

        spec = spectrogram.spectrogram_torch(
            wave, n_fft, sampling_rate, hop_size, win_size, center=False).squeeze(0).transpose(0, 1)
        return spec

    def process_wave_action(self, row):
        row['wave'] = row['wave'].squeeze().numpy()
        row['wave'] = librosa.util.normalize(row['wave']) # normalize
        row['wave'] = self.augmentor.process(row['wave'], 48000) # augment
        row['wave'] = librosa.util.normalize(row['wave']) # normalize
        return row

def row_action(row):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    ctx : LiveDataContext = dataset.live_data_context
    return ctx.process_wave_action(row)

def dataloader(filelist, config, **kwargs): # Provide raw wave only
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        dataset.live_data_context = LiveDataContext(config) 

    return dt.Dataset(
        filelist=filelist,
        field_specs=[
            dt.FieldSpec(name='wave', datatype=torch.Tensor,
                dim=torch.Size([-1]), 
                keep_in_memory=True, # because this is hand annotated we won't have much
                provide_length=True),
        ],
        actions=[
            dt.LiveMapRow(row_action),
            dt.RandomSubsample(fields=['wave'], dims=[0], length=48000),
            dt.PadGroup(fields=['wave'], dims=[0], values=[0]),
        ]).loader(worker_init_fn=worker_init_fn, **kwargs)