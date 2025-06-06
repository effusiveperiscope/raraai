import torch
import torch.nn.functional as F
import random

def random_subsample_segments(x: torch.Tensor,
                              x_mask: torch.Tensor,
                              min_segment_len: int = 4,
                              max_segment_len: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly subsamples a contiguous segment from m_p for each item in the batch,
    respecting the x_mask for valid positions.

    Args:
        x (torch.Tensor): The input tensor, e.g., prior means.
                            Shape: (batch_size, time_steps, feature_dim)
        x_mask (torch.Tensor): Boolean mask indicating valid time steps.
                               True for valid, False for invalid/padding.
                               Shape: (batch_size, time_steps)
        min_segment_len (int): Minimum length of the segment to sample.
        max_segment_len (int): Maximum length of the segment to sample.
                               The output tensors will be padded to this length.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - subsampled_m_p (torch.Tensor): The subsampled segments, padded to max_segment_len.
                                             Shape: (batch_size, max_segment_len, feature_dim)
            - subsampled_mask (torch.Tensor): Boolean mask for the subsampled segments.
                                              Shape: (batch_size, max_segment_len)
    """
    batch_size, _, feature_dim = x.shape
    device = x.device

    # Ensure max_segment_len is at least min_segment_len
    # This also defines the output sequence length for the padded batch
    output_seq_len = max(min_segment_len, max_segment_len)

    # Initialize output tensors (will be padded)
    batched_subsampled_m_p = torch.zeros(batch_size, output_seq_len, feature_dim,
                                         device=device, dtype=x.dtype)
    batched_subsampled_mask = torch.zeros(batch_size, output_seq_len,
                                          device=device, dtype=torch.bool)

    for i in range(batch_size):
        # Get indices of valid (True) frames for the current batch item
        # x_mask[i] is 1D, e.g., [True, True, True, False, False]
        # valid_indices will be like tensor([0, 1, 2])
        valid_indices = torch.where(x_mask[i])[0]
        num_valid_frames = len(valid_indices)

        if num_valid_frames == 0:
            # No valid frames for this item, segment will be all padding (zeros)
            # The initialized zeros in batched_subsampled_m_p and batched_subsampled_mask are correct
            current_actual_segment_len = 0
        else:
            # Determine the length of the segment to sample for this item
            if num_valid_frames < min_segment_len:
                # If fewer valid frames than min_segment_len, take all of them
                current_actual_segment_len = num_valid_frames
            else:
                # Sample a random length between min_segment_len and
                # min(max_segment_len, num_valid_frames)
                # The upper bound for randint must be >= lower bound.
                upper_bound_for_len_choice = min(output_seq_len, num_valid_frames)
                current_actual_segment_len = random.randint(min_segment_len, upper_bound_for_len_choice)

            # Choose a random start index for the segment *within the list of valid_indices*
            # The segment must fit within the available valid frames.
            # Max start index in valid_indices list: num_valid_frames - current_actual_segment_len
            if num_valid_frames - current_actual_segment_len < 0:
                 # This should ideally not happen if logic for current_actual_segment_len is correct.
                 # This case implies current_actual_segment_len > num_valid_frames,
                 # which means num_valid_frames < min_segment_len was true, and
                 # current_actual_segment_len was set to num_valid_frames.
                 # So, num_valid_frames - current_actual_segment_len should be 0.
                start_offset_in_valid_indices = 0
            else:
                start_offset_in_valid_indices = random.randint(0, num_valid_frames - current_actual_segment_len)

            # Get the actual frame indices from m_p's time dimension
            selected_original_indices = valid_indices[
                start_offset_in_valid_indices : start_offset_in_valid_indices + current_actual_segment_len
            ]

            # Subsample m_p using these original indices
            segment_data = x[i, selected_original_indices, :]

            # Place into the batched output tensors (first part, rest is padding)
            batched_subsampled_m_p[i, :current_actual_segment_len, :] = segment_data
            batched_subsampled_mask[i, :current_actual_segment_len] = True

    return batched_subsampled_m_p, batched_subsampled_mask

def slice_segments_general(x, ids_str, segment_size=4, pad_value=0.0):
    """
    Extract fixed-length segments from a 2D or 3D tensor, padding if out-of-bounds.

    Args:
        x (Tensor): 2D (B, T) or 3D (B, C, T) tensor
        ids_str (list or Tensor): Start indices per sample
        segment_size (int): Desired segment length
        pad_value (float): Value to use for padding if segment exceeds bounds

    Returns:
        Tensor: Padded segments of shape:
                - (B, segment_size) if input is 2D
                - (B, C, segment_size) if input is 3D
    """
    is_3d = x.dim() == 3
    B = x.size(0)
    T = x.size(-1)

    if is_3d:
        C = x.size(1)
        ret = x.new_full((B, C, segment_size), pad_value)
        for i in range(B):
            idx = ids_str[i]
            end = min(idx + segment_size, T)
            length = end - idx
            if length > 0:
                ret[i, :, :length] = x[i, :, idx:end]
    else:
        ret = x.new_full((B, segment_size), pad_value)
        for i in range(B):
            idx = ids_str[i]
            end = min(idx + segment_size, T)
            length = end - idx
            if length > 0:
                ret[i, :length] = x[i, idx:end]

    return ret

def load_state_dict_mismatch(model, state_dict):
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    mismatched_keys = []

    for key in state_dict:
        if key in model_state_dict:
            if state_dict[key].shape == model_state_dict[key].shape:
                filtered_state_dict[key] = state_dict[key]
            else:
                mismatched_keys.append((key, state_dict[key].shape, model_state_dict[key].shape))
        else:
            mismatched_keys.append((key, state_dict[key].shape, None))  # Key not in model

    if mismatched_keys:
        print("Mismatched or missing keys (skipped):")
        for key, shape_ckpt, shape_model in mismatched_keys:
            print(f"{key}: checkpoint shape = {shape_ckpt}, model shape = {shape_model}")

    model.load_state_dict(filtered_state_dict, strict=False)

def smooth_random_amplitude_modulation(spectrogram: torch.Tensor, 
                                       min_gain: float = 0.7, 
                                       max_gain: float = 1.3, 
                                       points: int = 64) -> torch.Tensor:
    """
    Apply smooth, random amplitude modulation over time to a spectrogram.
    
    Args:
        spectrogram (torch.Tensor): Tensor of shape [B, Time, Channels].
        min_gain (float): Minimum multiplicative gain.
        max_gain (float): Maximum multiplicative gain.
        points (int): Number of points in the low-res gain curve.
    
    Returns:
        torch.Tensor: Modulated spectrogram.
    """
    B, T, C = spectrogram.shape

    # Create low-res gain curve and reshape to [B, 1, points]
    low_res_gains = torch.rand(B, 1, points, device=spectrogram.device) * (max_gain - min_gain) + min_gain

    # Interpolate to [B, 1, T] (Time dimension)
    gain_curve = F.interpolate(low_res_gains, size=T, mode='linear', align_corners=True)

    # Reshape to [B, T, 1] for broadcasting over channels
    gain_curve = gain_curve.permute(0, 2, 1)

    # Apply gain
    return spectrogram * gain_curve

def tensor_check(tensor: torch.Tensor):
    if tensor.abs().max() > 1e4:
        print("Large value detected in tensor")
        import pdb; pdb.set_trace()
    if tensor.abs().max() < 1e-4:
        print("Small value detected in tensor")
        import pdb; pdb.set_trace()
    if torch.isnan(tensor).any():
        print("NaN detected in tensor")
        import pdb; pdb.set_trace()
    if torch.isinf(tensor).any():
        print("Inf detected in tensor")
        import pdb; pdb.set_trace()