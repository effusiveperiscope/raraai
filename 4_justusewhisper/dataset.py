from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from datasets import load_dataset
from einops import rearrange
import torch

class SpeechFeatureDataset(Dataset):
    def __init__(self, split='train',
        data_path='dataset_unconditional_2000.parquet', split_seed=0):
        dataset = load_dataset('parquet', data_files=[data_path])['train']
        train_test_split = dataset.train_test_split(
            test_size=0.05, seed=split_seed)
        dataset = train_test_split[split]
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        hubert = torch.Tensor(self.dataset[idx]['rvc_features']).squeeze()
        whisper = torch.Tensor(self.dataset[idx]['whisper_features']).squeeze()
        return hubert.squeeze(), whisper.squeeze()

def collate_fn(batch):
    with torch.no_grad():
        hubert, whisper = zip(*batch)
        hubert_padded = pad_sequence(hubert)
        whisper_padded = pad_sequence(whisper)
        max_len = max(hubert_padded.shape[0],
            whisper_padded.shape[0])

        # F.pad pads last dimension
        hubert_padded = rearrange(hubert_padded, 'n b c -> c b n')
        whisper_padded = rearrange(whisper_padded, 'n b c -> c b n')
        hubert_padded = F.pad(
            hubert_padded, (0, max_len - hubert_padded.shape[2],))
        whisper_padded = F.pad(
            whisper_padded, (0, max_len - whisper_padded.shape[2],))
        hubert_padded = rearrange(hubert_padded, 'c b n -> n b c')
        whisper_padded = rearrange(whisper_padded, 'c b n -> n b c')

        #import pdb
        #pdb.set_trace()

        return hubert_padded, whisper_padded
