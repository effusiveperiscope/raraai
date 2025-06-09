import os
import random
from omegaconf import OmegaConf
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class FeatureDatasetPaired(Dataset):
    def __init__(self, config: OmegaConf, is_train=True, override_filelist=None):
        filelist_to_use = config.train.filelist if is_train else config.train.val_filelist
        filelist_to_use = override_filelist if override_filelist is not None else filelist_to_use
        self.feature_dir = os.path.dirname(filelist_to_use)
        self.files_by_speaker = {}

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
                    
                    if sid not in self.files_by_speaker.keys():
                        self.files_by_speaker[sid] = []
                    self.files_by_speaker[sid].append((line, sid))
                else:
                    print(' Warning - no speaker ID found: ')
                    print(line)

        self.sorted_speaker_ids = sorted(self.files_by_speaker.keys())
        for sid in self.sorted_speaker_ids:
            self.files_by_speaker[sid].sort()

        self.all_files = []
        for sid in self.sorted_speaker_ids:
            self.all_files += self.files_by_speaker[sid]

        self.config = config
        self.max_len = self.config.data.max_len
        self.hop_length = self.config.data.hop_length
        self.is_train = is_train

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        file = self.all_files[idx]
        itemdict_A = self.itemdict(file)

        sid = itemdict_A['sid']
        random_other_sid = random.choice(
            [s for s in self.sorted_speaker_ids if s != sid])

        file = random.choice(self.files_by_speaker[random_other_sid])
        itemdict_B = self.itemdict(file)

        itemdict = {
            'A': itemdict_A,
            'B': itemdict_B,
        }

        return itemdict

    def itemdict(self, file: tuple):
        path = file[0]
        sid = file[1]

        basename = os.path.basename(path)
        load_dir = self.feature_dir if self.feature_dir else '.'

        # Load full-length tensors
        wave = torch.load(os.path.join(load_dir, basename + '.wave'))

        if os.path.exists(os.path.join(load_dir, basename + '.whisp_feat')):
            rvc_feat = torch.load(os.path.join(load_dir, basename + '.whisp_feat')).squeeze(0)

        pitch = torch.load(os.path.join(load_dir, basename + '.pitch')).squeeze(0)
        pitch_fine = torch.load(os.path.join(load_dir, basename + '.pitch_fine')).squeeze(0)

        num_samples = wave.shape[0]
        max_len = min(self.max_len, num_samples)
        frame_len = max_len // self.hop_length

        max_start = num_samples - max_len
        if not self.is_train:
            max_start = 0 # Just start from the beginning
        start_sample = random.randint(0, max_start) if max_start > 0 else 0
        end_sample = start_sample + max_len
        start_frame = start_sample // self.hop_length
        end_frame = start_frame + frame_len

        item_dict = {
            'wave': wave[start_sample:end_sample],
            'rvc_feat': rvc_feat[start_frame:end_frame],
            'pitch': pitch[start_frame:end_frame],
            'pitch_fine': pitch_fine[start_frame:end_frame],
            'sid': sid
        }
        
        spec_path = os.path.join(load_dir, basename + '.spec')
        if os.path.exists(spec_path):
            spec = torch.load(spec_path).squeeze(0)
            item_dict['spec'] = spec[start_frame:end_frame]

        return item_dict

import torch
import torch.nn.functional as F

def paired_feature_collator(batch):
    """
    Collator function for FeatureDatasetPaired.
    
    Args:
        batch: List of items from dataset, where each item has structure:
               {'A': itemdict_A, 'B': itemdict_B}
    
    Returns:
        Dictionary with structure:
        {
            'A': {
                'rvc_feat': padded tensor [batch_size, max_frames, channels],
                'pitch': padded tensor [batch_size, max_frames],
                'pitch_fine': padded tensor [batch_size, max_frames], 
                'spec': padded tensor [batch_size, max_frames, spec_channels],
                'wave': padded tensor [batch_size, max_samples],
                'lengths': tensor [batch_size] - actual lengths in frames,
                'sids': tensor [batch_size] - speaker IDs
            },
            'B': { ... same structure ... }
        }
    """
    batch_size = len(batch)
    
    # Collect all items for A and B separately
    items_A = [item['A'] for item in batch]
    items_B = [item['B'] for item in batch]
    
    # Find maximum lengths across ALL items (both A and B)
    all_items = items_A + items_B
    
    # Get frame lengths (all frame-level features should have same length)
    frame_lengths = []
    sample_lengths = []
    
    for item in all_items:
        frame_len = item['rvc_feat'].shape[0]
        frame_lengths.append(frame_len)
        sample_lengths.append(item['wave'].shape[0])
    
    max_frames = max(frame_lengths)
    max_samples = max(sample_lengths)
    
    def collate_items(items):
        """Helper function to collate a list of items (either A or B)"""
        # Initialize lists to collect tensors
        rvc_feats = []
        pitches = []
        pitch_fines = []
        specs = []
        waves = []
        lengths = []
        sids = []
        
        for item in items:
            frame_len = item['rvc_feat'].shape[0]
            sample_len = item['wave'].shape[0]
            
            # Pad frame-level features
            rvc_feat = item['rvc_feat']  # [frames, channels]
            pitch = item['pitch']       # [frames]
            pitch_fine = item['pitch_fine']  # [frames]
            
            # Pad to max_frames
            if frame_len < max_frames:
                pad_frames = max_frames - frame_len
                rvc_feat = F.pad(rvc_feat, (0, 0, 0, pad_frames))  # pad last dim (frames)
                pitch = F.pad(pitch, (0, pad_frames))
                pitch_fine = F.pad(pitch_fine, (0, pad_frames))
            
            rvc_feats.append(rvc_feat)
            pitches.append(pitch)
            pitch_fines.append(pitch_fine)
            
            # Handle spec (might not exist for all items)
            if 'spec' in item:
                spec = item['spec']  # [frames, spec_channels]
                if frame_len < max_frames:
                    spec = F.pad(spec, (0, 0, 0, pad_frames))
                specs.append(spec)
            else:
                # Create dummy spec if not available
                if len(specs) == 0:
                    # We don't know the spec channels yet, so we'll handle this later
                    specs.append(None)
                else:
                    # Use the same shape as previous specs but filled with zeros
                    dummy_spec = torch.zeros(max_frames, specs[0].shape[1])
                    specs.append(dummy_spec)
            
            # Pad wave (sample-level)
            wave = item['wave']  # [samples]
            if sample_len < max_samples:
                pad_samples = max_samples - sample_len
                wave = F.pad(wave, (0, pad_samples))
            
            waves.append(wave)
            lengths.append(frame_len)  # Store original frame length
            sids.append(item['sid'])
        
        # Stack tensors
        result = {
            'rvc_feat': torch.stack(rvc_feats),      # [batch_size, max_frames, channels]
            'pitch': torch.stack(pitches),          # [batch_size, max_frames]
            'pitch_fine': torch.stack(pitch_fines), # [batch_size, max_frames]
            'wave': torch.stack(waves),             # [batch_size, max_samples]
            'lengths': torch.tensor(lengths, dtype=torch.long),  # [batch_size]
            'sids': torch.tensor(sids, dtype=torch.long)         # [batch_size]
        }
        
        # Handle spec - check if any items actually have spec
        valid_specs = [s for s in specs if s is not None]
        if valid_specs:
            # Fill in None specs with zeros of correct shape
            spec_channels = valid_specs[0].shape[1]
            final_specs = []
            for spec in specs:
                if spec is None:
                    final_specs.append(torch.zeros(max_frames, spec_channels))
                else:
                    final_specs.append(spec)
            result['spec'] = torch.stack(final_specs)  # [batch_size, max_frames, spec_channels]
        else:
            # No specs available, create dummy
            result['spec'] = torch.zeros(batch_size, max_frames, 1)
        
        return result
    
    # Collate A and B items separately
    collated_A = collate_items(items_A)
    collated_B = collate_items(items_B)
    
    return {
        'A': collated_A,
        'B': collated_B,
        'max_frames': max_frames
    }

if __name__ == '__main__':
    config = OmegaConf.load('configs/base10v1.yaml')
    dataset = FeatureDatasetPaired(config, is_train=True)
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True,
        collate_fn=paired_feature_collator
    )

    # Usage
    for batch in dataloader:
        import pdb; pdb.set_trace()