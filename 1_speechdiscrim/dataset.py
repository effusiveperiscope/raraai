import torch
import dataset
from datasets import load_dataset
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
import numpy as np

class SpeechClassDataset(Dataset):
    def __init__(self, split='train', seed=0,
        data_path='dataset_over5min.parquet'):
        dataset = load_dataset('parquet', data_files=[data_path])['train']
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(dataset.unique('speaker'))
        self.n_speakers = len(self.label_encoder.classes_)

        train_test_split = dataset.train_test_split(test_size=0.05, seed=seed)

        dataset = train_test_split[split]
        self.dataset = dataset
        self.length = len(dataset)

    def dump_to_file(self, file):
        with open(file, 'w') as f:
            for example in self.dataset:
                f.write(f'{example["speaker"]}: {example["transcription"]}\n\n')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        speech_features = torch.Tensor(
            self.dataset[idx]['rvc_features']).squeeze()
        speaker_label = torch.Tensor(
            self.label_encoder.transform([self.dataset[idx]['speaker']]))
        return speech_features, speaker_label

def collate_fn(batch):
    speech_features, speaker_label = zip(*batch)
    speech_features_padded = pad_sequence(
        speech_features)
    return speech_features_padded, torch.cat(speaker_label).long()