import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import random

class AudioFeatureDataset(Dataset):
    def __init__(self, filelist_path):
        """
        Initialize the dataset by reading the filelist
        
        Args:
            filelist_path (str): Path to the filelist containing wav names and feature paths
        """
        self.data = []
        with open(filelist_path, 'r', encoding='utf-8') as f:
            for line in f:
                wav_name, f0_path, feat_path = line.strip().split('|')
                self.data.append({
                    'wav_name': wav_name,
                    'f0_path': f0_path,
                    'feat_path': feat_path
                })
    
    def __len__(self):
        """Return the total number of samples in the dataset"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Load and return a single sample's features
        
        Args:
            idx (int): Index of the sample
        
        Returns:
            dict: Dictionary containing f0, features, and metadata
        """
        item = self.data[idx]
        
        # Load f0 and features
        f0 = torch.load(item['f0_path'], weights_only=True)
        features = torch.load(item['feat_path'], weights_only=True)
        
        return {
            'f0': f0,
            'features': features,
            'wav_name': item['wav_name']
        }

class AudioFeatureCollator:
    def __init__(self, max_len=None):
        """
        Initialize the collator with optional max length for padding
        
        Args:
            max_len (int, optional): Maximum sequence length to pad to. 
                                     If None, uses the longest sequence in the batch.
        """
        self.max_len = max_len
    
    def __call__(self, batch):
        """
        Collate and pad a batch of samples
        
        Args:
            batch (list): List of samples from the dataset
        
        Returns:
            dict: Batched and padded tensors
        """
        # Extract components from batch
        f0_list = [item['f0'] for item in batch]
        features_list = [item['features'] for item in batch]
        wav_names = [item['wav_name'] for item in batch]
        
        # Determine max length
        max_f0_len = self.max_len or max(f0.shape[0] for f0 in f0_list)
        max_feat_len = self.max_len or max(feat.shape[1] for feat in features_list)
        
        # Pad f0 sequences
        f0_padded = torch.zeros(len(f0_list), max_f0_len, dtype=f0_list[0].dtype)
        f0_mask = torch.zeros(len(f0_list), max_f0_len, dtype=torch.bool)
        for i, f0 in enumerate(f0_list):
            f0_padded[i, :f0.shape[0]] = f0
            f0_mask[i, :f0.shape[0]] = 1
        
        # Pad feature sequences
        features_padded = torch.zeros(len(features_list), max_feat_len, features_list[0].shape[2], 
                                      dtype=features_list[0].dtype)
        features_mask = torch.zeros(len(features_list), max_feat_len, dtype=torch.bool)
        for i, feat in enumerate(features_list):
            features_padded[i, :feat.shape[1], :] = feat
            features_mask[i, :feat.shape[1]] = 1
        
        return {
            'f0': f0_padded,
            'f0_mask': f0_mask,
            'features': features_padded,
            'features_mask': features_mask,
            'wav_names': wav_names
        }

def train_val_split(dataset, val_split=0.2, random_seed=42):
    """
    Splits a dataset into training and validation sets.

    Args:
        dataset (Dataset): The dataset to split.
        val_split (float): The fraction of the dataset to use for validation.
        random_seed (int): Random seed for reproducibility.

        Returns:
        tuple: (train_dataset, val_dataset)
    """
    dataset_size = len(dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size

    indices = list(range(dataset_size))
    random.seed(random_seed)
    random.shuffle(indices)

    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset


def create_dataloader(filelist_path, batch_size=16, max_len=None):
    """
    Create a DataLoader for the audio feature dataset

    Args:
        filelist_path (str): Path to the filelist
        batch_size (int): Batch size for DataLoader
        max_len (int, optional): Maximum sequence length for padding

    Returns:
        torch.utils.data.DataLoader: DataLoader for the dataset
    """
    dataset = AudioFeatureDataset(filelist_path)
    collator = AudioFeatureCollator(max_len)

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collator
    )

    return dataloader

if __name__ == "__main__":
    # Assuming you have a filelist.list in the output directory from your previous script
    dataloader = create_dataloader('TestTwilight/filelist.list')
    
    for batch in dataloader:
        print("Batch F0 shape:", batch['f0'].shape)
        print("Batch Features shape:", batch['features'].shape)
        print("Wav Names:", batch['wav_names'])
        break  # Just print the first batch