import torch
from torch.utils.data import Dataset, Subset
import random

class MyDataset(Dataset):
    def __init__(self, filelist_path, config):
        self.filelist = []
        with open(filelist_path, "r", encoding="utf-8") as f:
            for line in f:
                self.filelist.append(line)
        self.config = config

        self.spk_set = set()
        for i in range(len(self.filelist)):
            line = self.filelist[i].strip().split("|")
            if self.config.train.override_sid is not None:
                pass
            elif self.config.model.n_speakers > 1:
                wav_path, spk, embed_path, feat_path = line
                self.spk_set.add(spk)

                self.spk_id_mapping = {}
                for i, spk in enumerate(sorted(list(self.spk_set))):
                    self.spk_id_mapping[spk] = i
        print(self.spk_id_mapping)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        line = self.filelist[idx].strip().split("|")
        if self.config.train.override_sid is not None:
            wav_path, embed_path, feat_path = line
            spk_id = self.config.train.override_sid
        elif self.config.model.n_speakers > 1:
            wav_path, spk, embed_path, feat_path = line
            spk_id = self.spk_id_mapping[spk]
        else:
            wav_path, embed_path, feat_path = line
        embed = torch.load(embed_path) # [1, T, 1024]
        feat = torch.load(feat_path) # [1, T, 768]
        if self.config.model.n_speakers > 1 or self.config.train.override_sid is not None:
            return wav_path, embed, feat, spk_id
        else:
            return wav_path, embed, feat

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

def collate_fn(batch):
    wav_paths, embeds, feats = zip(*batch)
    
    # Find the max length of embed and feat in the batch
    max_embed_len = max(embed.size(1) for embed in embeds)
    max_feat_len = max(feat.size(1) for feat in feats)
    
    # Pad embeds and feats to the max length
    padded_embeds = torch.zeros(len(embeds), max_embed_len, embeds[0].size(2))
    padded_feats = torch.zeros(len(feats), max_feat_len, feats[0].size(2))
    
    for i, (embed, feat) in enumerate(zip(embeds, feats)):
        padded_embeds[i, :embed.size(1)] = embed
        padded_feats[i, :feat.size(1)] = feat

    # Create masks for padded embeds and feats
    embed_masks = torch.zeros(len(embeds), max_embed_len).bool()
    feat_masks = torch.zeros(len(feats), max_feat_len).bool()
    for i, (embed, feat) in enumerate(zip(embeds, feats)):
        embed_masks[i, :embed.size(1)] = True
        feat_masks[i, :feat.size(1)] = True

    return wav_paths, padded_embeds, padded_feats, embed_masks, feat_masks

def collate_fn_multispk(batch):
    wav_paths, embeds, feats, spk_ids = zip(*batch)
    
    # Find the max length of embed and feat in the batch
    max_embed_len = max(embed.size(1) for embed in embeds)
    max_feat_len = max(feat.size(1) for feat in feats)
    
    # Pad embeds and feats to the max length
    padded_embeds = torch.zeros(len(embeds), max_embed_len, embeds[0].size(2))
    padded_feats = torch.zeros(len(feats), max_feat_len, feats[0].size(2))
    
    for i, (embed, feat) in enumerate(zip(embeds, feats)):
        padded_embeds[i, :embed.size(1)] = embed
        padded_feats[i, :feat.size(1)] = feat

    # Create masks for padded embeds and feats
    embed_masks = torch.zeros(len(embeds), max_embed_len).bool()
    feat_masks = torch.zeros(len(feats), max_feat_len).bool()
    for i, (embed, feat) in enumerate(zip(embeds, feats)):
        embed_masks[i, :embed.size(1)] = True
        feat_masks[i, :feat.size(1)] = True

    return wav_paths, padded_embeds, padded_feats, embed_masks, feat_masks, spk_ids
