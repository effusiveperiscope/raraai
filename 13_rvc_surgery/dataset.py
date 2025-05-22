import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torch.nn.functional as F

class FeatureDataset(Dataset):
    def __init__(self, config, is_train=True):
        self.feature_dir = os.path.dirname(config.train.filelist)
        if is_train:
            with open(config.train.filelist, 'r', encoding='utf-8') as f:
                self.files = [line.strip() for line in f.readlines()]
        else:
            with open(config.train.val_filelist, 'r', encoding='utf-8') as f:
                self.files = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        basename = os.path.basename(path)

        rvc_feat = torch.load(os.path.join(self.feature_dir, basename + '.rvc_feat'))  # [1, T, D]
        whisp_feat = torch.load(os.path.join(self.feature_dir, basename + '.whisp_feat'))  # [1, T, D]
        pitch = torch.load(os.path.join(self.feature_dir, basename + '.pitch'))  # [1, T]
        pitch_fine = torch.load(os.path.join(self.feature_dir, basename + '.pitch_fine'))  # [1, T]

        return {
            'rvc_feat': rvc_feat.squeeze(0),        # [T, D]
            'whisp_feat': whisp_feat.squeeze(0),    # [T, D]
            'pitch': pitch.squeeze(0),              # [T]
            'pitch_fine': pitch_fine.squeeze(0)     # [T]
        }

class FeatureCollator:
    def __call__(self, batch):
        # Handle empty batch
        if not batch:
            return {
                'rvc_feat': torch.empty((0, 0, 0)),    # B, T_max, D_rvc
                'whisp_feat': torch.empty((0, 0, 0)), # B, T_max, D_whisp
                'pitch': torch.empty((0, 0)),         # B, T_max
                'pitch_fine': torch.empty((0, 0)),    # B, T_max
                'lengths': torch.empty((0), dtype=torch.long)
            }

        # 1. Extract all individual feature sequences
        rvc_feats_list = [item['rvc_feat'] for item in batch]    # List of [T, D]
        whisp_feats_list = [item['whisp_feat'] for item in batch] # List of [T, D]
        pitch_list = [item['pitch'] for item in batch]           # List of [T]
        pitch_fine_list = [item['pitch_fine'] for item in batch] # List of [T]

        # 2. Determine original lengths (e.g., based on pitch, as in the original code)
        lengths = torch.tensor([p.size(0) for p in pitch_list], dtype=torch.long)

        # 3. Find the global maximum length across ALL sequences in the batch
        global_max_len = 0
        all_sequences_for_len_calc = rvc_feats_list + whisp_feats_list + pitch_list + pitch_fine_list
        if not all_sequences_for_len_calc: # Should only happen if batch was non-empty but items were strange
             global_max_len = 0
        else:
            for seq in all_sequences_for_len_calc:
                if seq.size(0) > global_max_len:
                    global_max_len = seq.size(0)
        
        # Handle case where all sequences might be empty (global_max_len could be 0)
        if global_max_len == 0:
            # If all sequences are empty, feature dimensions might be unknown for rvc/whisp
            # Try to infer D from the first item if available, otherwise use 0 or a default.
            D_rvc = 0
            if rvc_feats_list and rvc_feats_list[0].ndim > 1:
                D_rvc = rvc_feats_list[0].size(1)
            
            D_whisp = 0
            if whisp_feats_list and whisp_feats_list[0].ndim > 1:
                D_whisp = whisp_feats_list[0].size(1)

            num_items = len(batch)
            return {
                'rvc_feat': torch.zeros((num_items, 0, D_rvc)),
                'whisp_feat': torch.zeros((num_items, 0, D_whisp)),
                'pitch': torch.zeros((num_items, 0)),
                'pitch_fine': torch.zeros((num_items, 0)),
                'lengths': lengths # lengths would be all zeros here
            }


        # 4. Pad each sequence individually to global_max_len and then stack
        padded_rvc_feats = []
        for feat in rvc_feats_list:
            pad_len = global_max_len - feat.size(0)
            # F.pad expects (pad_left, pad_right, pad_top, pad_bottom, ...)
            # For [T, D], we want to pad T. So (0,0 for D, 0, pad_len for T)
            padded_feat = F.pad(feat, (0, 0, 0, pad_len), mode='constant', value=0)
            padded_rvc_feats.append(padded_feat)
        rvc_feats_batch = torch.stack(padded_rvc_feats, dim=0)

        padded_whisp_feats = []
        for feat in whisp_feats_list:
            pad_len = global_max_len - feat.size(0)
            padded_feat = F.pad(feat, (0, 0, 0, pad_len), mode='constant', value=0)
            padded_whisp_feats.append(padded_feat)
        whisp_feats_batch = torch.stack(padded_whisp_feats, dim=0)

        padded_pitch = []
        for p in pitch_list:
            pad_len = global_max_len - p.size(0)
            # For [T], we want to pad T. So (0, pad_len for T)
            padded_p = F.pad(p, (0, pad_len), mode='constant', value=0)
            padded_pitch.append(padded_p)
        pitch_batch = torch.stack(padded_pitch, dim=0)

        padded_pitch_fine = []
        for pf in pitch_fine_list:
            pad_len = global_max_len - pf.size(0)
            padded_pf = F.pad(pf, (0, pad_len), mode='constant', value=0)
            padded_pitch_fine.append(padded_pf)
        pitch_fine_batch = torch.stack(padded_pitch_fine, dim=0)

        return {
            'rvc_feat': rvc_feats_batch,
            'whisp_feat': whisp_feats_batch,
            'pitch': pitch_batch,
            'pitch_fine': pitch_fine_batch,
            'lengths': lengths  # Original lengths, useful for downstream tasks
        }