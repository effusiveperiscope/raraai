import random
from einops import rearrange
from torch.utils.data import Dataset, Subset
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
import numpy as np
from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(self, config : OmegaConf, filelist_path):
        self.data = []
        want_features = config.features.want
        with open(filelist_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Loading features"):
                line = line.strip().split("|")[1:] # First column is wav path
                if len(line) == 0:
                    continue
                data = {}

                for i, feat_name in enumerate(want_features):
                    feat_path = line[i]
                    data[feat_name] = torch.load(feat_path)
                self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

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

class Collator:
    def __init__(self, config):
        self.config = config
        self.max_seq_len = config.train.max_seq_len

    def collate(self, batch):
        decoder_pad_token = self.config.decoder.pad_token_id

        # Find max sequence lengths in the batch
        max_content_len = max(item['content_tokens'].shape[1] for item in batch)
        max_acoustic_len = min(
            self.max_seq_len,
            max(item['acoustic_codes'].shape[1] for item in batch))
        
        # Initialize lists to store padded tensors and lengths
        content_interp_pitch_padded = []
        content_tokens_padded = []
        acoustic_codes_padded = []
        content_seq_lens = []
        acoustic_codes_lens = []
        
        # Process each item in the batch
        for item in batch:
            # Get original shapes and store lengths
            content_len = item['content_tokens'].shape[1]
            acoustic_len = min(item['acoustic_codes'].shape[1], max_acoustic_len)
            content_seq_lens.append(content_len)
            acoustic_codes_lens.append(acoustic_len)
            
            # Pad content_interp_pitch
            content_interp_pitch = item['content_interp_pitch']
            pad_amount = max_content_len - content_len
            content_interp_pitch_padded.append(
                F.pad(content_interp_pitch, (0, pad_amount))
            )
            
            # Pad content_tokens
            content_tokens = item['content_tokens']
            content_tokens_padded.append(
                F.pad(content_tokens, (0, pad_amount))
            )
            
            # Pad acoustic_codes (padding on the second dimension)
            acoustic_codes = item['acoustic_codes']
            acoustic_codes_padded.append(
                F.pad(acoustic_codes,
                    (0, 0, 0, max_acoustic_len - acoustic_len), 
                    value=decoder_pad_token)[:, :max_acoustic_len, :]
            )
        
        # Stack all tensors
        result = {
            'content_interp_pitch': rearrange(torch.stack(content_interp_pitch_padded), "b d t -> b t d"),
            'content_tokens': rearrange(torch.stack(content_tokens_padded), "b d t -> b t d"),
            'acoustic_codes': torch.stack(acoustic_codes_padded).squeeze(1),
            'content_seq_lens': torch.tensor(content_seq_lens),
            'acoustic_codes_lens': torch.tensor(acoustic_codes_lens)
        }
        
        return result

if __name__ == "__main__":
    config = OmegaConf.load("configs/common.yaml")
    dataset = MyDataset(config, "twilight/filelist.txt")

    for k,v in dataset[0].items():
        print(k, v.shape)

    collator = Collator(config)
    collated = collator.collate([dataset[0], dataset[1]])
    for k,v in collated.items():
        print('collated ',k, v.shape)
        if 'lens' in k:
            print(v)

    from model import MyModel
    model = MyModel(config)
    model = model.cuda().eval()
    with torch.no_grad():
        outputs = model(**collated)
    print("logits shape", outputs.logits.shape)
    print("loss", outputs.loss)