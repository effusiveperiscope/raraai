import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import random
from omegaconf import OmegaConf  # Assuming OmegaConf is used for config
from einops import rearrange
from nsfhifigan.wav2mel import PitchAdjustableMelSpectrogram

# ------------------- Dataset Class -------------------

class MelF0Dataset(Dataset):
    """
    Dataset class to load preprocessed Mel-spectrogram and F0 pairs.
    Supports normalization, random cropping and random energy augmentation for training.
    """
    def __init__(self, filelist_path: str, config: OmegaConf, random_cropping: bool):
        """
        Initializes the dataset.

        Args:
            filelist_path (str): Path to the filelist (e.g., train.txt or val.txt).
                                 Each line should be in the format "mel_path|f0_path".
            config (OmegaConf): Configuration object containing parameters like:
                                - min_len (int): Minimum length for random cropping.
                                - max_len (int): Maximum length for random cropping.
                                - random_mel_energy_muls (list/tuple): [min_mul, max_mul] for energy augmentation.
            random_cropping (bool): Whether to apply random cropping and augmentation (True for train, False for val).
        """
        super().__init__()
        self.config = config
        self.random_cropping = random_cropping
        self.filelist = self._load_filelist(filelist_path)

        # --- Validation for config ---
        if not hasattr(config.train, 'min_len') or not isinstance(config.train.min_len, int):
            raise ValueError("Config must contain an integer 'min_len' attribute.")
        if not hasattr(config.train, 'max_len') or not isinstance(config.train.max_len, int):
             raise ValueError("Config must contain an integer 'max_len' attribute.")
        if config.train.min_len <= 0:
             raise ValueError("'min_len' must be positive.")
        if config.train.max_len < config.train.min_len:
            raise ValueError("'max_len' must be greater than or equal to 'min_len'.")
        if random_cropping:
             if not hasattr(config.train, 'random_mel_energy_muls') or \
               len(config.train.random_mel_energy_muls) != 2:

               #not isinstance(config.train.random_mel_energy_muls, (list, tuple)) or \ # relax this because it's omegaconf.listconfig.ListConfig
                 raise ValueError("Config must contain 'random_mel_energy_muls' as a list/tuple of length 2 for random cropping.")
             if not all(isinstance(x, (int, float)) for x in config.train.random_mel_energy_muls):
                  raise ValueError("'random_mel_energy_muls' must contain numeric values.")
             if config.train.random_mel_energy_muls[0] < 0 or config.train.random_mel_energy_muls[1] < config.train.random_mel_energy_muls[0]:
                 raise ValueError("'random_mel_energy_muls' must have non-negative values with min <= max.")
        # --- End Validation ---


        print(f"Loaded {len(self.filelist)} samples from {filelist_path}. Random cropping: {random_cropping}")

    def _load_filelist(self, filelist_path: str) -> list[tuple[str, str]]:
        """Loads the filelist where each line is 'mel_path|f0_path'."""
        if not os.path.exists(filelist_path):
            raise FileNotFoundError(f"Filelist not found: {filelist_path}")
        filepaths = []
        with open(filelist_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('|')
                if len(parts) != 2:
                    print(f"Skipping malformed line in {filelist_path}: {line}")
                    continue
                mel_path, f0_path = parts
                if not os.path.exists(mel_path):
                     print(f"Warning: Mel file not found: {mel_path} (referenced in {filelist_path})")
                     continue
                if not os.path.exists(f0_path):
                     print(f"Warning: F0 file not found: {f0_path} (referenced in {filelist_path})")
                     continue
                filepaths.append((mel_path, f0_path))
        if not filepaths:
            raise ValueError(f"No valid samples found in filelist: {filelist_path}")
        return filepaths

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.filelist)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Loads, processes, and returns a single data sample.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - mel_spec (torch.Tensor): Mel spectrogram tensor [n_mels, T_proc].
                - f0 (torch.Tensor): F0 contour tensor [T_proc].
        """
        mel_path, f0_path = self.filelist[index]

        try:
            # Load tensors saved by the preprocessing script
            # Expected shapes from script: mel [1, n_mels, T], f0 [1, T]
            mel_spec = torch.load(mel_path, map_location='cpu')
            f0 = torch.load(f0_path, map_location='cpu')

            # Remove the singleton batch dimension
            if mel_spec.dim() == 3 and mel_spec.shape[0] == 1:
                 mel_spec = mel_spec.squeeze(0) # Shape becomes [n_mels, T]
            if f0.dim() == 2 and f0.shape[0] == 1:
                 f0 = f0.squeeze(0) # Shape becomes [T]

            # Validate shapes after loading and squeezing
            if mel_spec.dim() != 2:
                 raise ValueError(f"Unexpected mel dimensions: {mel_spec.dim()} in {mel_path}")
            if f0.dim() != 1:
                 raise ValueError(f"Unexpected f0 dimensions: {f0.dim()} in {f0_path}")
            if mel_spec.shape[1] != f0.shape[0]:
                raise ValueError(f"Mel length ({mel_spec.shape[1]}) != F0 length ({f0.shape[0]}) in files {mel_path}, {f0_path}")

            seq_len = mel_spec.shape[1]

            if self.random_cropping:
                # --- Random Cropping ---
                if seq_len < self.config.train.min_len:
                    # If sequence is too short, pad it? Or skip? For now, let's use it as is.
                    # This case should ideally be handled during preprocessing or data filtering.
                    # If we must crop, we have to crop to seq_len.
                    crop_len = seq_len
                    start_idx = 0
                    # print(f"Warning: Sequence {index} length ({seq_len}) is less than min_len ({self.config.train.min_len}). Using full sequence.")
                else:
                    # Determine the maximum possible crop length for this specific sample
                    max_possible_crop_len = min(seq_len, self.config.train.max_len)
                    # Choose a random crop length between min_len and max_possible_crop_len
                    # Ensure min_len isn't greater than max_possible_crop_len before sampling
                    actual_min_len = min(self.config.train.min_len, max_possible_crop_len)
                    crop_len = random.randint(actual_min_len, max_possible_crop_len)

                    # Choose a random start index
                    start_idx = random.randint(0, seq_len - crop_len)

                # Perform the crop
                mel_spec = mel_spec[:, start_idx : start_idx + crop_len]
                f0 = f0[start_idx : start_idx + crop_len]

                # --- Random Energy Augmentation ---
                min_mul, max_mul = self.config.train.random_mel_energy_muls
                # Use torch.rand for potentially better randomness within range
                multiplier = torch.rand(1).item() * (max_mul - min_mul) + min_mul
                # Apply augmentation only to mel spectrogram
                # Add a small epsilon to prevent log(0) if mel values can be zero
                mel_spec = mel_spec * multiplier
                # Or, if working in log-domain typically:
                # mel_spec = mel_spec + torch.log(torch.tensor(multiplier) + 1e-6)

            else:
                # --- Validation Set Handling ---
                # Optionally truncate validation sequences to max_len if needed
                if seq_len > self.config.train.max_len:
                    # print(f"Warning: Val sequence {index} length ({seq_len}) > max_len ({self.config.train.max_len}). Truncating.")
                    mel_spec = mel_spec[:, :self.config.train.max_len]
                    f0 = f0[:self.config.train.max_len]
                # No cropping or augmentation for validation

            return mel_spec, f0

        except Exception as e:
            print(f"Error processing sample {index} ({mel_path}, {f0_path}): {e}")
            # Return None or raise error, depending on how the training loop handles it
            # Returning None might require filtering in the collator or dataloader loop
            # For simplicity here, let's re-raise after logging
            raise e

# ------------------- Collator Class -------------------

class MelF0Collator:
    """
    Collator class to pad batches of Mel/F0 data to a multiple of `padding_multiple`.
    """
    def __init__(self, config: OmegaConf, pad_value: float = 0.0, padding_multiple: int = 16):
        """
        Initializes the collator.

        Args:
            pad_value (float): Value used for padding. Defaults to 0.0.
            padding_multiple (int): The sequence length will be padded to be a
                                     multiple of this value. Defaults to 16.
        """
        self.pad_value = pad_value
        if not isinstance(padding_multiple, int) or padding_multiple <= 0:
            raise ValueError("padding_multiple must be a positive integer.")
        self.padding_multiple = padding_multiple
        print(f"Initialized MelF0Collator with pad_value={pad_value}, padding_multiple={padding_multiple}")


    def _ceil_to_multiple(self, length: int, multiple: int) -> int:
        """Calculates the smallest multiple of `multiple` greater than or equal to `length`."""
        return ((length + multiple - 1) // multiple) * multiple

    def __call__(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Processes a batch of samples from MelF0Dataset.

        Args:
            batch (list[tuple[torch.Tensor, torch.Tensor]]): A list where each element
                is a tuple (mel_spec, f0) returned by MelF0Dataset.__getitem__.
                mel_spec shape: [n_mels, T_i]
                f0 shape: [T_i]

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - padded_mel (torch.Tensor): Padded batch of mel spectrograms
                                             [B, T_pad, n_mels].
                - padded_f0 (torch.Tensor): Padded batch of F0 contours
                                            [B, T_pad].
                - mask (torch.Tensor): Boolean mask indicating valid (True) vs. padded (False)
                                       positions [B, T_pad].
        """
        # Filter out None entries if __getitem__ can return None on error
        batch = [item for item in batch if item is not None]
        if not batch:
            # Handle empty batch case if necessary
            return torch.empty(0), torch.empty(0), torch.empty(0)

        # Find the number of mel bins (should be consistent across batch)
        n_mels = batch[0][0].shape[0]
        # Find the maximum sequence length in the batch
        max_len = max(mel.shape[1] for mel, f0 in batch)

        # Calculate the target padded length
        padded_len = self._ceil_to_multiple(max_len, self.padding_multiple)

        # Initialize tensors for the padded batch and the mask
        # Assuming B = len(batch)
        B = len(batch)
        # Shape: [B, n_mels, T_pad]
        padded_mel = torch.full((B, n_mels, padded_len), self.pad_value, dtype=batch[0][0].dtype)
        # Shape: [B, T_pad]
        padded_f0 = torch.full((B, padded_len), self.pad_value, dtype=batch[0][1].dtype)
        # Shape: [B, T_pad]
        mask = torch.zeros((B, padded_len), dtype=torch.bool)

        # Fill the tensors
        for i, (mel, f0) in enumerate(batch):
            current_len = mel.shape[1]
            if mel.shape[0] != n_mels: # Sanity check
                 raise ValueError(f"Inconsistent number of mel bins in batch! Expected {n_mels}, got {mel.shape[0]}")
            if f0.shape[0] != current_len: # Sanity check
                 raise ValueError(f"Inconsistent mel/f0 length within sample! Mel: {current_len}, F0: {f0.shape[0]}")

            padded_mel[i, :, :current_len] = mel
            padded_f0[i, :current_len] = f0
            mask[i, :current_len] = True # Mark the valid (non-padded) positions

        padded_mel = rearrange(padded_mel, 'b c n -> b n c')
        log_padded_mel = PitchAdjustableMelSpectrogram.dynamic_range_compression_torch(padded_mel, clip_val=1e-5)

        return log_padded_mel, padded_f0.to(padded_mel.dtype), mask