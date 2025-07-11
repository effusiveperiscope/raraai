import os
import random
from omegaconf import OmegaConf
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class FeatureDataset(Dataset):
    def __init__(self, config: OmegaConf, is_train=True, override_filelist=None,
        default_sid=0):
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
        self.max_len = self.config.data.max_len
        self.hop_length = self.config.data.hop_length
        self.default_sid = default_sid

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if type(self.files[idx]) == tuple:
            path = self.files[idx][0]
            sid = self.files[idx][1]
        else:
            path = self.files[idx]
            sid = self.default_sid
        basename = os.path.basename(path)
        load_dir = self.feature_dir if self.feature_dir else '.'

        # Load full-length tensors
        wave = torch.load(os.path.join(load_dir, basename + '.wave'))

        if os.path.exists(os.path.join(load_dir, basename + '.whisp_feat')):
            whisp_feat = torch.load(os.path.join(load_dir, basename + '.whisp_feat')).squeeze(0)
        else:
            whisp_feat = torch.zeros_like(wave) 

        if os.path.exists(os.path.join(load_dir, basename + '.rvc_feat')):
            rvc_feat = torch.load(os.path.join(load_dir, basename + '.rvc_feat')).squeeze(0)
        else:
            rvc_feat = torch.zeros_like(whisp_feat) # Hack to avoid having to process RVC feats

        pitch_fine = torch.load(os.path.join(load_dir, basename + '.pitch_fine')).squeeze(0)
        if os.path.exists(os.path.join(load_dir, basename + '.pitch')):
            pitch = torch.load(os.path.join(load_dir, basename + '.pitch')).squeeze(0)
        else:
            pitch = torch.zeros_like(pitch_fine)

        if os.path.exists(os.path.join(load_dir, basename + '.spk_feat')):
            spk_feat = torch.load(os.path.join(load_dir, basename + '.spk_feat')).squeeze(0)
        else:
            raise ValueError('spk_feat does not exist for {}'.format(basename))

        num_samples = wave.shape[0]
        max_len = min(self.max_len, num_samples)
        frame_len = max_len // self.hop_length

        max_start = num_samples - max_len
        start_sample = random.randint(0, max_start) if max_start > 0 else 0
        end_sample = start_sample + max_len
        start_frame = start_sample // self.hop_length
        end_frame = start_frame + frame_len

        item_dict = {
            'wave': wave[start_sample:end_sample],
            'rvc_feat': rvc_feat[start_frame:end_frame],
            'whisp_feat': whisp_feat[start_frame:end_frame],
            'pitch': pitch[start_frame:end_frame],
            'pitch_fine': pitch_fine[start_frame:end_frame],
            'sid': sid,
            'spk_feat': spk_feat,
        }
        
        spec_path = os.path.join(load_dir, basename + '.spec')
        if os.path.exists(spec_path):
            spec = torch.load(spec_path).squeeze(0)
            item_dict['spec'] = spec[start_frame:end_frame]

        return item_dict

class FeatureCollator:
    def _get_dim_and_dtype(self, tensor_list, default_dim_if_2d=0, default_dtype=torch.float):
        dim = default_dim_if_2d
        dtype = default_dtype
        # Find the first non-None tensor that has elements to infer properties
        first_valid_tensor = next((t for t in tensor_list if t is not None and hasattr(t, 'numel') and t.numel() > 0), None)
        
        if first_valid_tensor is not None:
            dtype = first_valid_tensor.dtype
            if first_valid_tensor.ndim > 1:
                dim = first_valid_tensor.size(1)
            # If ndim is 1 (or 0), dim remains default_dim_if_2d
        return dim, dtype

    def _pad_sequences(self, sequence_list, max_len, D, dtype_for_none, is_1d=False, feature_name="feature"):
        padded_sequences = []
        for i, seq in enumerate(sequence_list):
            if seq is None: # Handle items where this feature was missing
                if is_1d:
                    padded_seq = torch.zeros(max_len, dtype=dtype_for_none)
                else:
                    padded_seq = torch.zeros((max_len, D), dtype=dtype_for_none)
            else:
                if not is_1d and D > 0 and seq.ndim > 1 and seq.size(1) != D:
                    raise ValueError(
                        f"Inconsistent {feature_name} dimension for item {i}. "
                        f"Expected D={D}, got {seq.size(1)}. Sequence shape: {seq.shape}"
                    )
                
                current_len = seq.size(0) if seq.ndim > 0 else 0
                pad_len = max_len - current_len

                if pad_len < 0: # Should not happen if max_len is calculated correctly
                    raise ValueError(f"Negative padding length for {feature_name} item {i}. max_len: {max_len}, current_len: {current_len}")

                if seq.ndim == 2:  # Expected [T, D]
                    padded_seq = F.pad(seq, (0, 0, 0, pad_len), mode='constant', value=0)
                elif seq.ndim == 1:  # Expected [T]
                    if not is_1d and D > 0 : # Trying to pad a 1D tensor as a 2D tensor e.g. spec [T,1]
                        if D == 1: # This sequence is [T], needs to be [T,1]
                             padded_seq = F.pad(seq.unsqueeze(1), (0, 0, 0, pad_len), mode='constant', value=0)
                        else: # D > 1 but seq is 1D. This is an inconsistency unless D was determined from other samples.
                              # This specific case should ideally be caught by D consistency checks or data prep.
                              # For robustness, create zeros.
                            # print(f"Warning: {feature_name} item {i} is 1D, but D={D}. Padding as zeros.")
                            padded_seq = torch.zeros((max_len, D), dtype=seq.dtype)

                    else: # is_1d is True (like pitch, wave), or (not is_1d and D==0, e.g. all features were empty [T,0])
                        padded_seq = F.pad(seq, (0, pad_len), mode='constant', value=0)
                elif seq.ndim == 0: # scalar tensor
                     if is_1d:
                        padded_seq = torch.zeros(max_len, dtype=seq.dtype)
                        if max_len > 0: padded_seq[0] = seq # put scalar at the beginning
                     else:
                        padded_seq = torch.zeros((max_len, D), dtype=seq.dtype)
                        if max_len > 0 and D > 0 : padded_seq[0,0] = seq
                else: # ndim > 2, unexpected
                    raise ValueError(f"Unsupported tensor ndim={seq.ndim} for {feature_name} item {i}")

            padded_sequences.append(padded_seq)
        
        if not padded_sequences:
            if is_1d: return torch.empty((0, max_len), dtype=dtype_for_none)
            else: return torch.empty((0, max_len, D), dtype=dtype_for_none)

        try:
            return torch.stack(padded_sequences, dim=0)
        except RuntimeError as e:
            # Provide more context if stacking fails
            # for i, p_seq in enumerate(padded_sequences):
            #     print(f"Shape of padded sequence {i} ({feature_name}): {p_seq.shape}")
            raise RuntimeError(f"Failed to stack {feature_name}. Error: {e}")


    def __call__(self, batch):
        if not batch:
            return {
                'rvc_feat': torch.empty((0, 0, 0)),
                'whisp_feat': torch.empty((0, 0, 0)),
                'pitch': torch.empty((0, 0)),
                'pitch_fine': torch.empty((0, 0)),
                'spec': torch.empty((0,0,0)),
                'wave': torch.empty((0,0)), # Wave is now also batched
                'lengths': torch.empty((0), dtype=torch.long)
            }

        num_items = len(batch)

        # 1. Extract all individual feature sequences
        rvc_feats_list = [item['rvc_feat'] for item in batch]
        whisp_feats_list = [item['whisp_feat'] for item in batch]
        pitch_list = [item['pitch'] for item in batch]
        pitch_fine_list = [item['pitch_fine'] for item in batch]
        spec_list = [item.get('spec') for item in batch]
        wave_list = [item.get('wave') for item in batch]
        sid_list = [item.get('sid') for item in batch]
        spk_feat_list = [item.get('spk_feat') for item in batch]

        # 2. Determine original frame lengths (e.g., based on pitch)
        lengths = torch.tensor([p.size(0) for p in pitch_list], dtype=torch.long)

        # 3. Determine feature dimensions (D_x) and dtypes from the batch
        D_rvc, dtype_rvc = self._get_dim_and_dtype(rvc_feats_list)
        D_whisp, dtype_whisp = self._get_dim_and_dtype(whisp_feats_list)
        D_spec, dtype_spec = self._get_dim_and_dtype(spec_list)
        _, dtype_pitch = self._get_dim_and_dtype(pitch_list, default_dtype=torch.float)
        _, dtype_pf = self._get_dim_and_dtype(pitch_fine_list, default_dtype=torch.float)
        _, dtype_wave = self._get_dim_and_dtype(wave_list, default_dtype=torch.float)

        # 4. Calculate global_max_len for frame-level features (rvc, whisp, pitch, pitch_fine, spec)
        global_max_len = 0
        seqs_for_frame_len_calc = []
        for lst in [rvc_feats_list, whisp_feats_list, pitch_list, pitch_fine_list, spec_list]:
            seqs_for_frame_len_calc.extend(s for s in lst if s is not None and hasattr(s, 'size') and s.ndim > 0)
        
        if seqs_for_frame_len_calc:
            frame_lengths = [s.size(0) for s in seqs_for_frame_len_calc if s.numel() > 0] # Only consider non-empty tensors
            if frame_lengths:
                global_max_len = max(frame_lengths)
        # global_max_len remains 0 if all frame-level features are None or empty tensors

        # 5. Calculate max_wave_len for wave features (sample-level)
        max_wave_len = 0
        valid_waves_for_len = [w for w in wave_list if w is not None and hasattr(w, 'size') and w.ndim > 0]
        if valid_waves_for_len:
            wave_lengths = [w.size(0) for w in valid_waves_for_len if w.numel() > 0] # Only consider non-empty waves
            if wave_lengths:
                max_wave_len = max(wave_lengths)
        # max_wave_len remains 0 if all waves are None or empty tensors

        # 6. Pad and stack frame-level features
        if global_max_len == 0 and num_items > 0 : # All frame features are empty or not present
            rvc_feats_batch = torch.zeros((num_items, 0, D_rvc), dtype=dtype_rvc)
            whisp_feats_batch = torch.zeros((num_items, 0, D_whisp), dtype=dtype_whisp)
            pitch_batch = torch.zeros((num_items, 0), dtype=dtype_pitch)
            pitch_fine_batch = torch.zeros((num_items, 0), dtype=dtype_pf)
            spec_batch = torch.zeros((num_items, 0, D_spec), dtype=dtype_spec)
        elif num_items > 0:
            rvc_feats_batch = self._pad_sequences(rvc_feats_list, global_max_len, D_rvc, dtype_rvc, feature_name="rvc_feat")
            whisp_feats_batch = self._pad_sequences(whisp_feats_list, global_max_len, D_whisp, dtype_whisp, feature_name="whisp_feat")
            pitch_batch = self._pad_sequences(pitch_list, global_max_len, 0, dtype_pitch, is_1d=True, feature_name="pitch")
            pitch_fine_batch = self._pad_sequences(pitch_fine_list, global_max_len, 0, dtype_pf, is_1d=True, feature_name="pitch_fine")
            spec_batch = self._pad_sequences(spec_list, global_max_len, D_spec, dtype_spec, feature_name="spec")
        else: # num_items == 0, already handled by `if not batch`
            pass # Should not reach here due to the initial empty batch check

        # 7. Pad and stack wave features
        if max_wave_len == 0 and num_items > 0: # All waves are empty or not present
            wave_batch = torch.zeros((num_items, 0), dtype=dtype_wave)
        elif num_items > 0:
            wave_batch = self._pad_sequences(wave_list, max_wave_len, 0, dtype_wave, is_1d=True, feature_name="wave")
        else: # num_items == 0
            pass # Should not reach here

        return {
            'rvc_feat': rvc_feats_batch,
            'whisp_feat': whisp_feats_batch,
            'pitch': pitch_batch,
            'pitch_fine': pitch_fine_batch,
            'spec': spec_batch,
            'wave': wave_batch,   # Now a batched tensor
            'lengths': lengths,    # Original frame-level lengths
            'sids': torch.Tensor(sid_list).long(),
            'spk': torch.stack(spk_feat_list)
        }