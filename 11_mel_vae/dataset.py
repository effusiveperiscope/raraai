import math
from pathlib import Path
import random
import librosa
import numpy as np
from nsfhifigan.wav2mel import PitchAdjustableMelSpectrogram
from nsfhifigan.config_utils import read_full_config
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import os # Import os for checking file existence
import torch

class MelDataset(Dataset):
    """
    Dataset class to load audio, apply optional volume augmentation,
    compute Mel spectrograms, and handle length constraints.
    """
    def __init__(self, filelist_path, max_len, mel_spec_transform, target_sr, config_train):
        """
        Args:
            filelist_path (str): Path to the text file containing audio file paths.
            max_len (int): Maximum number of frames allowed for the Mel spectrogram.
            mel_spec_transform (callable): The transformation to compute Mel spectrograms.
            target_sr (int): The sample rate expected by the mel_spec_transform.
            config_train: The training section of the configuration object/dict.
                          Expected keys related to volume augmentation:
                          - config_train.apply_random_volume (bool, optional)
                          - config_train.target_peak_norm (float, optional)
                          - config_train.random_volume_db_range (list[float, float], optional)
        """
        super().__init__()
        self.filelist_path = filelist_path
        self.max_len = max_len
        self.mel_spec_transform = mel_spec_transform
        self.target_sr = target_sr

        # --- Volume Augmentation Parameters ---
        self.apply_random_volume = getattr(config_train, 'apply_random_volume', False)
        if self.apply_random_volume:
            self.target_peak = getattr(config_train, 'target_peak_norm', 0.95)
            self.volume_db_range = getattr(config_train, 'random_volume_db_range', [-3.0, 3.0])
            print(f"  Volume Augmentation Enabled:")
            print(f"    Target Peak Norm: {self.target_peak}")
            print(f"    Random Gain Range (dB): {self.volume_db_range}")
        else:
             print(f"  Volume Augmentation Disabled.")
        # --- End Volume Augmentation ---

        print(f"Initializing MelDataset:")
        print(f"  Filelist: {self.filelist_path}")
        print(f"  Max Mel Length: {self.max_len}")
        print(f"  Target SR: {self.target_sr}")

        try:
            with open(self.filelist_path, 'r', encoding='utf-8') as f:
                self.filelist = [line.strip() for line in f if line.strip()]
            print(f"  Loaded {len(self.filelist)} files from list.")
            if not self.filelist:
                 print("Warning: Filelist is empty.")

        except FileNotFoundError:
            print(f"Error: Filelist not found at {self.filelist_path}")
            self.filelist = []
        except Exception as e:
            print(f"Error reading filelist {self.filelist_path}: {e}")
            self.filelist = []


    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        """
        Loads audio, applies optional volume augmentation, computes Mel spectrogram,
        and applies random cropping if needed.

        Returns:
            torch.Tensor: Mel spectrogram tensor [n_mels, T] or None if loading fails.
        """
        filepath = self.filelist[index]

        if not os.path.exists(filepath):
            print(f"Warning: Audio file not found: {filepath}. Skipping.")
            return None

        try:
            wav, sr = librosa.load(filepath, sr=self.target_sr, mono=True)
        except Exception as e:
            print(f"Error loading audio file {filepath}: {e}. Skipping.")
            return None

        # --- Apply Volume Augmentation ---
        if self.apply_random_volume and len(wav) > 0:
            # 1. Peak normalize the audio
            peak = np.max(np.abs(wav))
            if peak > 1e-5: # Avoid division by zero for silence
                wav = wav / peak * self.target_peak

            # 2. Apply random gain
            gain_db = random.uniform(self.volume_db_range[0], self.volume_db_range[1])
            gain_linear = 10.0 ** (gain_db / 20.0)
            wav = wav * gain_linear

            # 3. Optional: Clip to ensure it stays within a valid range (e.g., [-1, 1])
            #    Clipping after gain helps prevent potential issues with downstream processes,
            #    even though peak normalization was applied earlier.
            wav = np.clip(wav, -1.0, 1.0) # Clip to standard audio range
            # Or clip back to target_peak if preferred:
            # wav = np.clip(wav, -self.target_peak, self.target_peak)
        # --- End Volume Augmentation ---


        # Convert to torch tensor (CPU)
        wav_tensor = torch.FloatTensor(wav).unsqueeze(0) # Shape: [1, num_samples]

        # --- Compute Mel Spectrogram and Crop (rest is the same as before) ---
        try:
            with torch.no_grad():
                mel_spectrogram = self.mel_spec_transform(wav_tensor) # Shape: [1, n_mels, T]

            mel = mel_spectrogram.squeeze(0) # Shape: [n_mels, T]

            mel_len = mel.shape[-1]
            if mel_len > self.max_len:
                max_start = mel_len - self.max_len
                start = random.randint(0, max_start)
                mel = mel[:, start : start + self.max_len]
            elif mel_len == 0:
                print(f"Warning: Zero-length mel spectrogram for {filepath}. Skipping.")
                return None

            return mel

        except Exception as e:
            print(f"Error processing file {filepath} during Mel transform: {e}. Skipping.")
            return None

class MelCollator:
    """
    Collator function to pad Mel spectrograms in a batch to the same length,
    with padding to multiples of a specified value.
    """
    def __init__(self, pad_value=0.0, padding_multiple=16):
        """
        Args:
            pad_value (float): Value used for padding.
            padding_multiple (int): Pad sequences to multiples of this value.
        """
        self.pad_value = pad_value
        self.padding_multiple = padding_multiple
        print(f"MelCollator Initialized: Pad Value = {self.pad_value}, Padding Multiple = {self.padding_multiple}")
        
    def __call__(self, batch):
        """
        Processes a batch of Mel spectrograms.
        
        Args:
            batch (list): A list of Mel spectrogram tensors (output of MelDataset.__getitem__).
                          Items can be None if loading/processing failed for some files.
        
        Returns:
            tuple: Contains:
                - torch.Tensor: A batch of padded Mel spectrograms [batch_size, max_T_in_batch, n_mels].
                - torch.Tensor: Boolean mask indicating valid sequence positions [batch_size, max_T_in_batch].
                  Returns None if the batch is empty after filtering Nones.
        """
        # Filter out None values (from failed file loads/processing)
        batch = [item for item in batch if item is not None]
        
        # If the entire batch failed, return None
        if not batch:
            return None
        
        # Get sequence lengths for mask creation later
        seq_lengths = [mel.shape[1] for mel in batch]
        
        # Round up max length to nearest multiple of padding_multiple
        max_len = max(seq_lengths)
        padded_max_len = math.ceil(max_len / self.padding_multiple) * self.padding_multiple
        
        # Create padded batch with desired dimensions [batch_size, max_T_in_batch, n_mels]
        padded_mels = []
        for mel in batch:
            # Transpose mel from [n_mels, T] to [T, n_mels]
            mel = mel.transpose(0, 1)  # Now [T, n_mels]
            
            # Calculate padding
            pad_len = padded_max_len - mel.shape[0]
            
            # Pad the sequence
            if pad_len > 0:
                # Pad at the end (right) of the sequence dimension
                padded_mel = F.pad(mel, (0, 0, 0, pad_len), mode='constant', value=self.pad_value)
            else:
                padded_mel = mel
                
            padded_mels.append(padded_mel)
        
        # Stack tensors into a batch [batch_size, padded_max_len, n_mels]
        padded_batch = torch.stack(padded_mels)
        
        # Create sequence mask [batch_size, padded_max_len]
        # True for valid positions, False for padded positions
        mask = torch.zeros((len(batch), padded_max_len), dtype=torch.bool, device=padded_batch.device)
        for i, length in enumerate(seq_lengths):
            mask[i, :length] = True
            
        return padded_batch, mask

class MelDataloader(DataLoader):
    """
    DataLoader for Mel spectrograms using MelDataset and MelCollator.
    """
    def __init__(self, config, is_train : bool = True):
        """
        Args:
            config: A configuration object/dict containing necessary parameters.
                    Expected keys:
                    - config.train.filelist (str)
                    - config.train.max_len (int)
                    - config.train.nsfhifigan_config (str)
                    - config.train.batch_size (int)
                    - config.train.shuffle (bool)
                    - config.train.num_workers (int)
                    - config.train.pin_memory (bool)
                    - config.train.mel_pad_value (float, optional, defaults to 0.0)
        """
        print("Initializing MelDataloader...")
        # --- 1. Read NSF-HiFiGAN config ---
        try:
            nsfhifigan_config = read_full_config(Path(config.train.nsfhifigan_config))
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

        # --- 3. Create Dataset ---
        self.dataset = MelDataset(
            filelist_path=config.train.filelist if is_train else config.val.filelist,
            max_len=config.train.max_len,
            mel_spec_transform=self.mel_spec_transform,
            target_sr=nsfhifigan_config['audio_sample_rate'], # Pass target SR to dataset
            config_train=config.train
        )

        # --- 4. Create Collator ---
        # Use getattr for optional parameter with default
        mel_pad_value = getattr(config.train, 'mel_pad_value', 0.0)
        self.collator = MelCollator(pad_value=mel_pad_value)

        # --- 5. Initialize DataLoader ---
        super().__init__(
            dataset=self.dataset,
            batch_size=config.train.batch_size,
            shuffle=config.train.shuffle if is_train else False,
            num_workers=config.train.num_workers,
            pin_memory=config.train.pin_memory,
            collate_fn=self.collator, # Use the custom collator
            drop_last=True # Often useful for training stability, prevents tiny last batch
        )
        print("MelDataloader initialization complete.")
        print(f"  Batch Size: {config.train.batch_size}")
        print(f"  Shuffle: {config.train.shuffle}")
        print(f"  Num Workers: {config.train.num_workers}")
        print(f"  Pin Memory: {config.train.pin_memory}")

if __name__ == '__main__':
    from omegaconf import OmegaConf
    from tqdm import tqdm
    config = OmegaConf.load("config.yaml")
    train_dataloader = MelDataloader(config)
    val_dataloader = MelDataloader(config, is_train=False)
    for b in tqdm(train_dataloader, total=len(train_dataloader.dataset)):
        pass