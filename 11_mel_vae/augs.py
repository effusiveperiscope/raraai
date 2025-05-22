import random
import torch

def time_mask_synced(log_mels, x_masks, max_width=20, num_masks=2):
    """
    Applies the same time masks to a list of log_mel spectrograms.
    log_mels: List of [B, T, F] tensors.
    x_masks: List of corresponding [B, T] boolean/int masks.
    """
    if not log_mels:
        return []

    # Clone all input tensors to avoid in-place modification of originals
    augmented_log_mels = [lm.clone() for lm in log_mels]
    
    B, T_max, F = augmented_log_mels[0].shape # Assume all have same F, B and T_max for padding

    for b in range(B):
        # Determine the minimum valid_t for this batch item across all spectrograms
        # This ensures the generated mask parameters are valid for all.
        min_valid_t_for_batch_item = T_max 
        current_valid_ts = []
        for x_mask_single in x_masks:
            # Ensure x_mask_single[b] is 1D, then sum
            valid_t_single = int(x_mask_single[b].sum().item())
            current_valid_ts.append(valid_t_single)
            min_valid_t_for_batch_item = min(min_valid_t_for_batch_item, valid_t_single)

        if min_valid_t_for_batch_item <= 1: # Not enough frames in at least one spectrogram
            continue

        for _ in range(num_masks):
            # Determine mask width 't', ensuring it's not wider than min_valid_t_for_batch_item
            current_max_maskable_width = min(max_width, min_valid_t_for_batch_item)
            if current_max_maskable_width == 0:
                continue
            
            t = random.randint(0, current_max_maskable_width) # t=0 means no mask this iter
            if t == 0:
                continue

            # Determine starting point 't0' based on min_valid_t_for_batch_item
            # t0 must be in [0, min_valid_t_for_batch_item - t]
            if min_valid_t_for_batch_item - t <= 0: # Should not happen if t <= min_valid_t_for_batch_item and t > 0
                                                # Handles t == min_valid_t_for_batch_item
                t0 = 0
            else:
                t0 = random.randint(0, min_valid_t_for_batch_item - t)
            
            # Apply this same (t, t0) to all spectrograms for this batch item
            for aug_lm in augmented_log_mels:
                # The mask [t0:t0+t] is guaranteed to be within the min_valid_t,
                # so it's safe for all spectrograms in the list for this batch item.
                aug_lm[b, t0:t0+t, :] = 0.0
                
    return augmented_log_mels

def freq_mask_synced(log_mels, max_width=10, num_masks=2):
    """
    Applies the same frequency masks to a list of log_mel spectrograms.
    log_mels: List of [B, T, F] tensors.
    """
    if not log_mels:
        return []

    augmented_log_mels = [lm.clone() for lm in log_mels]
    B, T, F = augmented_log_mels[0].shape # Assume all have same B, T, F

    if F == 0: # No frequency bins to mask
        return augmented_log_mels

    for b in range(B): # Loop through batch, but masks are applied across full time T for each item
        for _ in range(num_masks):
            # Determine mask width 'f'
            current_max_maskable_width = min(max_width, F)
            if current_max_maskable_width == 0:
                continue
            
            f = random.randint(0, current_max_maskable_width) # f=0 means no mask this iter
            if f == 0:
                continue

            # Determine starting point 'f0'
            if F - f <= 0: # Should not happen if f <= F and f > 0
                f0 = 0
            else:
                f0 = random.randint(0, F - f)
            
            # Apply this same (f, f0) to all spectrograms
            for aug_lm in augmented_log_mels:
                aug_lm[b, :, f0:f0+f] = 0.0
                
    return augmented_log_mels

def time_jitter_synced(log_mels, x_masks, max_shift=2):
    """
    Applies the same time jitter to a list of log_mel spectrograms.
    log_mels: List of [B, T, F] tensors.
    x_masks: List of corresponding [B, T] boolean/int masks.
    """
    if not log_mels:
        return []

    # Jittering requires a clone as source and destination are different within the valid part
    jittered_log_mels = [lm.clone() for lm in log_mels]
    
    B, T_max, F = jittered_log_mels[0].shape # Assume all have same F, B and T_max for padding

    for b in range(B):
        # Determine the minimum valid_t for this batch item across all spectrograms
        min_valid_t_for_batch_item = T_max
        current_valid_ts = [] # Store individual valid_t to use for each tensor
        for x_mask_single in x_masks:
            valid_t_single = int(x_mask_single[b].sum().item())
            current_valid_ts.append(valid_t_single)
            min_valid_t_for_batch_item = min(min_valid_t_for_batch_item, valid_t_single)

        if min_valid_t_for_batch_item <= 1: # Cannot jitter meaningfully
            continue

        # Determine shift amount. This shift must be applicable to the shortest valid sequence.
        # Cap actual_max_shift to be less than min_valid_t_for_batch_item
        effective_max_shift = min(max_shift, min_valid_t_for_batch_item - 1)
        if effective_max_shift < 0: effective_max_shift = 0 # In case min_valid_t_for_batch_item was 0 or 1

        shift = random.randint(-effective_max_shift, effective_max_shift)

        if shift == 0:
            continue
        
        # Apply this same shift to all spectrograms
        for idx, (original_lm, jittered_lm) in enumerate(zip(log_mels, jittered_log_mels)):
            valid_t = current_valid_ts[idx] # Use the specific valid_t for this tensor
            if valid_t <=1: # Should have been caught by min_valid_t check, but good for safety
                continue

            # Source is the original valid portion of the current log_mel
            src_valid_part = original_lm[b, :valid_t, :]
            
            # Destination buffer (within the valid_t window for this specific tensor), initialized to zeros
            # Note: Using original_lm[b, :valid_t, :] to get the shape for zeros_like,
            # as jittered_lm is what we are modifying.
            dest_valid_part = torch.zeros_like(original_lm[b, :valid_t, :])

            if shift > 0: # Shift right, pad beginning
                # Number of elements to copy must not exceed what's available in src after shifting
                num_elements_to_copy = max(0, valid_t - shift)
                if num_elements_to_copy > 0:
                    dest_valid_part[shift : shift + num_elements_to_copy, :] = src_valid_part[:num_elements_to_copy, :]
            else: # shift < 0 (shift left, pad end)
                abs_shift = -shift
                num_elements_to_copy = max(0, valid_t - abs_shift)
                if num_elements_to_copy > 0:
                    dest_valid_part[:num_elements_to_copy, :] = src_valid_part[abs_shift : abs_shift + num_elements_to_copy, :]
            
            jittered_lm[b, :valid_t, :] = dest_valid_part
            # Padded regions beyond valid_t in jittered_lm[b] remain as they were (from clone).
            # If the original log_mel was padded, ensure those pads are zero or desired value.
            # Here, we explicitly zero out parts of valid_t that are "shifted away".
            # If you want to preserve original padding beyond valid_t, no extra step needed.
            if valid_t < T_max: # if there was padding in original
                 jittered_lm[b, valid_t:, :] = original_lm[b, valid_t:, :] # restore original padding


    return jittered_log_mels