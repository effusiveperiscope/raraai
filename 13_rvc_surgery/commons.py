import torch
import torch.nn.functional as F

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