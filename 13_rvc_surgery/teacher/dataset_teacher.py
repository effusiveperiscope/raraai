import os
import random
from omegaconf import OmegaConf
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class FeatureDataset(Dataset):
    def __init__(self, config: OmegaConf, is_train=True, override_filelist=None):
        filelist_to_use = config.train.filelist if is_train else config.train.val_filelist
        filelist_to_use = override_filelist if override_filelist is not None else filelist_to_use
        self.feature_dir = os.path.dirname(filelist_to_use)
        self.files = []

        log_ons = False

        with open(filelist_to_use, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if '|' in line:

                    if not log_ons:
                        print('=== Multispeaker filelist detected! ===')
                    log_ons=True

                    split = line.strip().split('|')
                    line = split[0]
                    sid = int(split[1])
                    self.files.append((line, sid))
                else:
                    self.files.append(line.strip())
        self.config = config

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if type(self.files[idx]) == tuple:
            path = self.files[idx][0]
            sid = self.files[idx][1]

        if type(self.files[idx]) == tuple:
            path = self.files[idx][0]
            sid = self.files[idx][1]
        else:
            path = self.files[idx]
            sid = 0
        basename = os.path.basename(path)
        load_dir = self.feature_dir if self.feature_dir else '.'

        whisp_feat = torch.load(os.path.join(load_dir, basename + '.whisp_feat')).squeeze(0)
        svc5_feat = torch.load(os.path.join(load_dir, basename + '.svc5_feat')).squeeze(0)
        pitch_fine = torch.load(os.path.join(load_dir, basename + '.pitch_fine')).squeeze(0)
        spk = torch.load(os.path.join(load_dir, basename + '.spk_feat')).squeeze(0)

        seq_len = min(whisp_feat.shape[0], svc5_feat.shape[0], pitch_fine.shape[0])
        whisp_feat = whisp_feat[:seq_len, :]
        svc5_feat = svc5_feat[:seq_len, :]
        pitch_fine = pitch_fine[:seq_len]
    
        item_dict = {
            'whisp_feat': whisp_feat, # [T, C]
            'svc5_feat': svc5_feat, # [T, C]
            'pitch_fine': pitch_fine, # [T]
            'spk_feat': spk, # [256]
            'sid': sid,
        }

        return item_dict

import torch
from torch.nn.utils.rnn import pad_sequence

class FeatureCollator:
    def __call__(self, batch):
        # Extract features and lengths
        whisp_feats = [item['whisp_feat'] for item in batch]
        svc5_feats = [item['svc5_feat'] for item in batch]
        pitch_fines = [item['pitch_fine'] for item in batch]
        sids = [item['sid'] for item in batch]
        spk_feats = [item['spk_feat'] for item in batch]

        lengths = torch.tensor([feat.size(0) for feat in whisp_feats], dtype=torch.long)

        # Pad sequences
        whisp_feats_padded = pad_sequence(whisp_feats, batch_first=True)  # [B, T_max, C]
        svc5_feats_padded = pad_sequence(svc5_feats, batch_first=True)    # [B, T_max, C]
        pitch_fines_padded = pad_sequence(pitch_fines, batch_first=True)  # [B, T_max]

        sids_tensor = torch.tensor(sids, dtype=torch.long)

        return {
            'whisp_feat': whisp_feats_padded,
            'svc5_feat': svc5_feats_padded,
            'pitch_fine': pitch_fines_padded,
            'length': lengths,
            'sid': sids_tensor,
            'spk': torch.stack(spk_feats)
        }


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    config = 'configs/teacher_test.yaml'
    config = OmegaConf.load(config)
    dataset = FeatureDataset(config, is_train=True)
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True,
        collate_fn=FeatureCollator()
    )

    # Usage
    for batch in dataloader:
        import pdb; pdb.set_trace()