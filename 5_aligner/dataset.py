from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from datasets import load_dataset
from einops import rearrange
import torch

class SpeechFeatureDataset(Dataset):
    def __init__(self, split='train',
        data_path='dataset_unconditional_50.parquet', split_seed=0):
        dataset = load_dataset('parquet', data_files=[data_path])['train']
        train_test_split = dataset.train_test_split(
            test_size=0.05, seed=split_seed)
        dataset = train_test_split[split]
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        hubert = torch.Tensor(
            self.dataset[idx]['rvc_features']).squeeze()
        hubert_len = hubert.shape[0]
        whisper = torch.Tensor(
            self.dataset[idx]['whisper_decoder_features']).squeeze()
        whisper_len = whisper.shape[0]
        return hubert.squeeze(), hubert_len, whisper.squeeze(), whisper_len

def collate_fn(batch):
    with torch.no_grad():
        hubert, hubert_lens, whisper, whisper_lens = zip(*batch)

        hubert_padded = pad_sequence(hubert)
        whisper_padded = pad_sequence(whisper)
        hubert_lens = torch.tensor(hubert_lens)
        whisper_lens = torch.tensor(whisper_lens)

        hubert_padded = rearrange(hubert_padded, 'n b c -> b n c')
        whisper_padded = rearrange(whisper_padded, 'n b c -> b n c')

        return hubert_padded, hubert_lens, whisper_padded, whisper_lens
