import os
import random
from einops import rearrange
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class FeatureDataset(Dataset):
    def __init__(self, cfg : OmegaConf, is_train=True):
        super().__init__()
        self.cfg = cfg

        self.files = []
        with open(os.path.join(cfg.train.filelist), "r", encoding="utf-8") as f:
            self.files = f.readlines()
        self.files = [f.strip() for f in self.files]

        random.seed(cfg.train.seed)
        random.shuffle(self.files)

        if is_train:
            self.files = self.files[int(len(self.files) * cfg.train.val_frac):]
        else:
            self.files = self.files[:int(len(self.files) * cfg.train.val_frac)]
        
        self.is_train = is_train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        whisper, phones, pitch, spk_id = self.files[index].split("|")
        basename = os.path.basename(whisper).removesuffix("_whisper.npy")
        whisper = np.load(whisper) # (1, T, C)
        phones = np.load(phones) # (T2,)
        pitch = np.load(pitch) # (Tp, )

        # RVC expects whisper seq dim to be interpolated up 2x
        whisper = torch.from_numpy(whisper)
        whisper = rearrange(whisper, "1 T C -> 1 C T")
        whisper = F.interpolate(whisper, scale_factor=2)
        whisper = rearrange(whisper, "1 C T -> 1 T C")

        pitch = pitch[:whisper.shape[1]] # Pitch is expected to be same length as whisper
        return whisper, torch.from_numpy(phones), torch.from_numpy(pitch), int(spk_id), basename

class FeatureCollator:
    def __init__(self, config):
        """
        Collator for FeatureDataset that batches and pads whisper features and phone sequences.
        """
        self.config = config
    def __call__(self, batch):
        """
        Collate function for DataLoader.
        
        Args:
            batch: List of tuples (whisper, phones, pitches, spk_id) from FeatureDataset
        
        Returns:
            dict containing:
                - whisper: [B, T_max, C] padded whisper features
                - phones: [B, T2_max] padded phone sequences  
                - spk_ids: [B] speaker IDs
                - whisper_lengths: [B] actual lengths of whisper sequences
                - phones_lengths: [B] actual lengths of phone sequences
                - whisper_mask: [B, T_max] boolean mask (True for valid positions)
                - phones_mask: [B, T2_max] boolean mask (True for valid positions)
        """
        whisper_list, phones_list, pitches_list, spk_ids, basenames = zip(*batch)
        
        # Convert to lists and get original lengths
        whisper_lengths = torch.tensor([w.shape[1] for w in whisper_list], dtype=torch.long)  # T dimension
        phones_lengths = torch.tensor([len(p) for p in phones_list], dtype=torch.long)
        spk_ids = torch.tensor(spk_ids, dtype=torch.long)
        
        # Handle whisper features: [1, T, C] -> [T, C] for padding, then back to [B, T, C]
        whisper_squeezed = [w.squeeze(0) for w in whisper_list]  # Remove batch dim: [T, C]
        whisper_padded = pad_sequence(whisper_squeezed, batch_first=True, padding_value=0)  # [B, T_max, C]
        
        # Pad pitches
        pitches_padded = pad_sequence(pitches_list, batch_first=True, padding_value=0)
        
        # Handle phone sequences: [T2] -> [B, T2_max]
        phones_padded = pad_sequence(phones_list, batch_first=True, padding_value=self.config.model.pad_token_id)  # [B, T2_max]
        
        # Create length masks
        batch_size = len(batch)
        max_whisper_len = whisper_padded.shape[1]
        max_phones_len = phones_padded.shape[1]
        
        # Create masks: True for valid positions, False for padding
        whisper_mask = torch.arange(max_whisper_len).unsqueeze(0) < whisper_lengths.unsqueeze(1)  # [B, T_max]
        phones_mask = torch.arange(max_phones_len).unsqueeze(0) < phones_lengths.unsqueeze(1)    # [B, T2_max]
        
        return {
            'whisper': whisper_padded,           # [B, T_max, C]
            'phones': phones_padded,             # [B, T2_max]
            'pitches': pitches_padded,           # [B, T_max]
            'spk_ids': spk_ids,                  # [B]
            'whisper_lengths': whisper_lengths,  # [B]
            'phones_lengths': phones_lengths,    # [B]
            'whisper_mask': whisper_mask,        # [B, T_max]
            'phones_mask': phones_mask,          # [B, T2_max]
            'basenames': basenames
        }