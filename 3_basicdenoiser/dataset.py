from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from augment.noise import GaussianNoise, PerlinNoise
from augment.pca import PCAPerturbation
from augment.bias import BiasPerturbation
import torch
import numpy as np
from datasets import load_dataset

class SpeechFeatureDataset(Dataset):
    def __init__(self, split='train',
        data_path='dataset_2000.parquet', split_seed = 0):
        dataset = load_dataset('parquet', data_files=[data_path])['train']
        train_test_split = dataset.train_test_split(
            test_size=0.05, seed=split_seed)
        dataset = train_test_split[split]
        self.dataset = dataset
        self.gn = GaussianNoise()
        self.pca = PCAPerturbation()
        self.pn = PerlinNoise(inner_module=self.pca)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        clean = torch.Tensor(self.dataset[idx]['rvc_features'])
        # Apply perlin noise scaled PCA perturbation
        _, noisy, _ = self.pn.process_features(None, clean)
        # Apply gaussian noise
        _, noisy, _ = self.gn.process_features(None, noisy)
        return clean.squeeze(), noisy.squeeze()

def collate_fn(batch):
    with torch.no_grad():
        clean, noisy = zip(*batch)
        clean_padded = pad_sequence(clean)
        noisy_padded = pad_sequence(noisy)
        return clean_padded, noisy_padded