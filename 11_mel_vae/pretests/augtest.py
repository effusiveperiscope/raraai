import random
import torch
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# --- Your Augmentation Functions (Copied and pasted here for self-containment) ---
def time_mask_synced(log_mels, x_masks, max_width=20, num_masks=2):
    if not log_mels: return []
    augmented_log_mels = [lm.clone() for lm in log_mels]
    B, T_max, F = augmented_log_mels[0].shape
    for b in range(B):
        min_valid_t_for_batch_item = T_max
        for x_mask_single in x_masks:
            valid_t_single = int(x_mask_single[b].sum().item())
            min_valid_t_for_batch_item = min(min_valid_t_for_batch_item, valid_t_single)
        if min_valid_t_for_batch_item <= 1: continue
        for _ in range(num_masks):
            current_max_maskable_width = min(max_width, min_valid_t_for_batch_item)
            if current_max_maskable_width == 0: continue
            t = random.randint(0, current_max_maskable_width)
            if t == 0: continue
            if min_valid_t_for_batch_item - t <= 0: t0 = 0
            else: t0 = random.randint(0, min_valid_t_for_batch_item - t)
            for aug_lm in augmented_log_mels:
                aug_lm[b, t0:t0+t, :] = 0.0
    return augmented_log_mels

def freq_mask_synced(log_mels, max_width=10, num_masks=2):
    if not log_mels: return []
    augmented_log_mels = [lm.clone() for lm in log_mels]
    B, T, F_bins = augmented_log_mels[0].shape
    if F_bins == 0: return augmented_log_mels
    for b in range(B):
        for _ in range(num_masks):
            current_max_maskable_width = min(max_width, F_bins)
            if current_max_maskable_width == 0: continue
            f = random.randint(0, current_max_maskable_width)
            if f == 0: continue
            if F_bins - f <= 0: f0 = 0
            else: f0 = random.randint(0, F_bins - f)
            for aug_lm in augmented_log_mels:
                aug_lm[b, :, f0:f0+f] = 0.0
    return augmented_log_mels

def time_jitter_synced(log_mels, x_masks, max_shift=2):
    if not log_mels: return []
    jittered_log_mels = [lm.clone() for lm in log_mels]
    B, T_max, F_bins = jittered_log_mels[0].shape
    for b in range(B):
        min_valid_t_for_batch_item = T_max
        current_valid_ts = []
        for x_mask_single in x_masks:
            valid_t_single = int(x_mask_single[b].sum().item())
            current_valid_ts.append(valid_t_single)
            min_valid_t_for_batch_item = min(min_valid_t_for_batch_item, valid_t_single)
        if min_valid_t_for_batch_item <= 1: continue
        effective_max_shift = min(max_shift, min_valid_t_for_batch_item - 1)
        if effective_max_shift < 0: effective_max_shift = 0
        shift = random.randint(-effective_max_shift, effective_max_shift)
        if shift == 0: continue
        for idx, (original_lm_cloned, jittered_lm) in enumerate(zip(log_mels, jittered_log_mels)):
            # Note: log_mels here are the *clones* made at the start of the function.
            # We actually want the true original to source from if we're applying to one item.
            # For this test, log_mels[idx] is fine as it's the initial clone.
            valid_t = current_valid_ts[idx]
            if valid_t <=1: continue
            
            # Use the original_lm_cloned (which is a clone of the input before jittering) as the source
            src_valid_part = original_lm_cloned[b, :valid_t, :]
            dest_valid_part = torch.zeros_like(src_valid_part)

            if shift > 0:
                num_elements_to_copy = max(0, valid_t - shift)
                if num_elements_to_copy > 0:
                    dest_valid_part[shift : shift + num_elements_to_copy, :] = src_valid_part[:num_elements_to_copy, :]
            else:
                abs_shift = -shift
                num_elements_to_copy = max(0, valid_t - abs_shift)
                if num_elements_to_copy > 0:
                    dest_valid_part[:num_elements_to_copy, :] = src_valid_part[abs_shift : abs_shift + num_elements_to_copy, :]
            
            jittered_lm[b, :valid_t, :] = dest_valid_part
            if valid_t < T_max: # Restore original padding
                 jittered_lm[b, valid_t:, :] = original_lm_cloned[b, valid_t:, :]
    return jittered_log_mels

# --- Helper function for plotting ---
def plot_spectrograms(original_spec, augmented_spec, title, sr, hop_length):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle(title, fontsize=16)

    # Original Spectrogram
    img1 = librosa.display.specshow(original_spec, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', ax=axs[0])
    axs[0].set_title("Original")
    fig.colorbar(img1, ax=axs[0], format='%+2.0f dB')

    # Augmented Spectrogram
    img2 = librosa.display.specshow(augmented_spec, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', ax=axs[1])
    axs[1].set_title("Augmented")
    fig.colorbar(img2, ax=axs[1], format='%+2.0f dB')

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle


# --- Main Test Script ---
if __name__ == "__main__":
    # 1. Load an example audio file
    # You can replace this with your own audio file
    try:
        audio_path = librosa.ex('trumpet')
    except Exception as e:
        print(f"Could not load librosa example audio: {e}")
        print("Using a generated noise signal instead.")
        sr_target = 22050
        duration = 5
        y = np.random.randn(sr_target * duration)
        sr = sr_target
    else:
        y, sr = librosa.load(audio_path, sr=None) # sr=None to load with original sampling rate

    # 2. Compute Log-Mel Spectrogram
    n_fft = 2048
    hop_length = 512
    n_mels = 128

    mel_spec_power = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_mel_spec_np = librosa.power_to_db(mel_spec_power, ref=np.max) # (F, T)

    # 3. Prepare for augmentation functions
    # Reshape to (B, T, F) and convert to PyTorch tensor
    # Here B=1 since we have one audio file
    log_mel_torch = torch.from_numpy(log_mel_spec_np.T).unsqueeze(0).float() # (1, T, F)
    
    # Create a dummy x_mask (assuming the whole spectrogram is valid)
    B, T_frames, F_bins = log_mel_torch.shape
    x_mask_torch = torch.ones(B, T_frames, dtype=torch.bool)

    print(f"Original Log-Mel Spectrogram shape: {log_mel_torch.shape}") # Should be (1, num_frames, num_mels)

    # --- Test Time Masking ---
    print("\nTesting Time Masking...")
    # For visualization, let's use a single log_mel in the list
    augmented_tm_list = time_mask_synced(
        log_mels=[log_mel_torch],
        x_masks=[x_mask_torch],
        max_width=int(T_frames * 0.15), # Mask up to 15% of total time
        num_masks=3
    )
    augmented_tm_torch = augmented_tm_list[0]
    augmented_tm_np = augmented_tm_torch.squeeze(0).numpy().T # Back to (F, T) for plotting
    plot_spectrograms(log_mel_spec_np, augmented_tm_np, "Time Masking Effect", sr, hop_length)

    # --- Test Frequency Masking ---
    print("\nTesting Frequency Masking...")
    augmented_fm_list = freq_mask_synced(
        log_mels=[log_mel_torch],
        max_width=int(F_bins * 0.2), # Mask up to 20% of frequency bins
        num_masks=2
    )
    augmented_fm_torch = augmented_fm_list[0]
    augmented_fm_np = augmented_fm_torch.squeeze(0).numpy().T
    plot_spectrograms(log_mel_spec_np, augmented_fm_np, "Frequency Masking Effect", sr, hop_length)

    # --- Test Time Jitter ---
    print("\nTesting Time Jitter...")
    # Important: Pass the original log_mel_torch (or its clone) to time_jitter_synced
    # if you want to see jitter from the original.
    # If you pass an already augmented one, it will jitter that.
    augmented_tj_list = time_jitter_synced(
        log_mels=[log_mel_torch.clone()], # Pass a clone of the original for jitter
        x_masks=[x_mask_torch],
        max_shift=int(T_frames * 0.05) # Shift up to 5% of total time
    )
    augmented_tj_torch = augmented_tj_list[0]
    augmented_tj_np = augmented_tj_torch.squeeze(0).numpy().T
    plot_spectrograms(log_mel_spec_np, augmented_tj_np, "Time Jitter Effect", sr, hop_length)
    
    plt.show()