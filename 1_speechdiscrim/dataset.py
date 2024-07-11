import torch
import dataset
from datasets import load_dataset, Audio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
import numpy as np
import soundfile as sf
from logging import warn
import io

#from svc_helper.augmentation.pedalboard import PedalboardRandomAugmentor
#from svc_helper.sfeatures.models import RVCHubertModel
device = 'cuda' if torch.cuda.is_available else 'cpu'

# Problem: On windows, there is basically no sane way to do this with multiple
# workers since hubert model will also duplicate across workers
# The only "reasonable" way to do an augmented dataset is to store it on disk
# which I don't have disk space for :(

# class SpeechClassDataset1(Dataset):
    # r"""Dataset that randomly augments and computes speech features on the fly"""
    # def __init__(self, split='train', seed=0,
        # dataset='synthbot/pony-speech'):
# 
        # train_data = load_dataset(dataset)['train']
        # character_durations = {}
        # character_counts = {}
        # threshold_duration = 300
# 
        # def aggregate_durations(example):
            # if not example['speaker'] in character_durations:
                # character_durations[example['speaker']] = 0.0
                # character_counts[example['speaker']] = 0
            # character_durations[example['speaker']] += (example['end'] - example['start'])
            # character_counts[example['speaker']] += 1
# 
        # train_data.map(aggregate_durations)
        # qualified_speakers = {
            # speaker for speaker, duration in character_durations.items()
                # if duration > threshold_duration}
# 
        # https://github.com/huggingface/datasets/issues/5947
        # def soundfile_validate_filter(example):
            # try:
                # b = io.BytesIO(example['audio']['bytes'])
                # array, sr = sf.read(b)
                # return True
            # except sf.LibsndfileError as e:
                # return False
        # dataset = train_data.cast_column('audio',
            # Audio(decode=False)).filter(
            # soundfile_validate_filter).filter(
            # lambda ex: ex['speaker'] in qualified_speakers
            # ).cast_column('audio', Audio(decode=True))
# 
        # self.label_encoder = LabelEncoder()
        # self.label_encoder.fit(dataset.unique('speaker'))
        # self.n_speakers = len(self.label_encoder.classes_)
# 
        # train_test_split = dataset.train_test_split(test_size=0.05, seed=seed)
# 
        # dataset = train_test_split[split]
        # self.dataset = dataset
        # self.length = len(dataset)
        # self.avoid_idxs = set()
# 
    # def dump_to_file(self, file):
        # with open(file, 'w') as f:
            # for example in self.dataset:
                # f.write(f'{example["speaker"]}: {example["transcription"]}\n\n')
    # 
    # def classes(self):
        # return self.dataset['speaker']
# 
    # def __len__(self):
        # return self.length
# 
    # https://github.com/pytorch/vision/issues/689
    # def open_process_locals(self):
        # self.augmentor = PedalboardRandomAugmentor()
        # self.sfeatures_model = RVCHubertModel(device = device) 
# 
    # def __getitem__(self, idx):
        # if not hasattr(self, 'augmentor'):
            # self.open_process_locals()
        # try:
            # audio = self.dataset[idx]['audio']
        # except sf.LibsndfileError as e:
            # if idx not in self.avoid_idxs:
                # self.avoid_idxs.add(idx)
                # warn(f"Missing audio data for line {self.dataset[idx]}")
            # audio = {'array': np.array([0.0]), 'sampling_rate': 0}
# 
        # sr = audio['sampling_rate']
        # audio = self.augmentor.process(audio['array'], sr)
# 
        # speech_features = self.sfeatures_model.extract_features(
            # audio=torch.tensor(audio), sr=sr).detach().squeeze()
        # speaker_label = torch.Tensor(
            # self.label_encoder.transform([self.dataset[idx]['speaker']]))
        # return speech_features, speaker_label

class SpeechClassDataset0(Dataset):
    r"""Dataset that uses precomputed speech features"""
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

    def classes(self):
        return self.dataset['speaker']

    def __getitem__(self, idx):
        speech_features = torch.Tensor(
            self.dataset[idx]['rvc_features']).squeeze()
        speaker_label = torch.Tensor(
            self.label_encoder.transform([self.dataset[idx]['speaker']]))
        return speech_features, speaker_label

def collate_fn(batch):
    with torch.no_grad():
        speech_features, speaker_label = zip(*batch)
        speech_features_padded = pad_sequence(
            speech_features)
        return speech_features_padded, torch.cat(speaker_label).long()

from torch.utils.data import Sampler
class AvoidSampler(Sampler):
    def __init__(self, classes, avoid_idxs = set()):
        super().__init__()
        self.avoid_idxs = avoid_idxs
        self.len = len(classes)

    def __len__(self):
        return self.len

    def __iter__(self):
        for i in range(0,self.len):
            if i not in self.avoid_idxs:
                yield i

class BalancedSampler(Sampler):
    r"""
    Random sampling balanceable according to an alpha parameter
    classes - should be an array of values representing classes in the same
    order as the dataset"""

    def __init__(self, classes, avoid_idxs=set(), alpha=1.0):
        """
        alpha is the 'balanced'-ness, being completely unbalanced at 0
        """
        super().__init__()
        class_totals = {}
        class_probs = []
        class_pools = {}
        total_sum = 0
        for i, class_label in enumerate(classes):
            if class_label not in class_totals:
                class_totals[class_label] = 0
                class_pools[class_label] = []

            class_totals[class_label] += 1
            class_pools[class_label].append(i)
            total_sum += 1

        class_count = len(class_totals)
        for class_label in class_totals.keys():
            class_total = class_totals[class_label]
            class_probs.append(np.interp(
                alpha, [0,1], [class_total / total_sum, 1 / class_count]))

        self.classes = list(class_totals.keys())
        self.len = len(classes)
        self.class_pools = class_pools
        self.class_probs = class_probs
        self.avoid_idxs = avoid_idxs

    def __len__(self):
        return self.len

    def __iter__(self):
        idx = 0
        for i in range(0,self.len):
            selected_class = np.random.choice(self.classes, p=self.class_probs).item()
            idx = np.random.choice(self.class_pools[selected_class]).item()
            if idx in self.avoid_idxs:
                while idx in self.avoid_idxs:
                    selected_class = np.random.choice(
                        self.classes, p=self.class_probs)
                    idx = np.random.choice(self.class_pools[selected_class]).item()
            yield idx
