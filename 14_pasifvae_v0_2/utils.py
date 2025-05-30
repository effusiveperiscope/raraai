import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image # For converting plot to image array
import torch.nn as nn
import random

def random_subsample_segments(m_p: torch.Tensor,
                              x_mask: torch.Tensor,
                              min_segment_len: int = 4,
                              max_segment_len: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly subsamples a contiguous segment from m_p for each item in the batch,
    respecting the x_mask for valid positions.

    Args:
        m_p (torch.Tensor): The input tensor, e.g., prior means.
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
    batch_size, _, feature_dim = m_p.shape
    device = m_p.device

    # Ensure max_segment_len is at least min_segment_len
    # This also defines the output sequence length for the padded batch
    output_seq_len = max(min_segment_len, max_segment_len)

    # Initialize output tensors (will be padded)
    batched_subsampled_m_p = torch.zeros(batch_size, output_seq_len, feature_dim,
                                         device=device, dtype=m_p.dtype)
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
            segment_data = m_p[i, selected_original_indices, :]

            # Place into the batched output tensors (first part, rest is padding)
            batched_subsampled_m_p[i, :current_actual_segment_len, :] = segment_data
            batched_subsampled_mask[i, :current_actual_segment_len] = True

    return batched_subsampled_m_p, batched_subsampled_mask

def reset_corrupt_batchnorm_stats(model: nn.Module):
    """
    Resets running stats (mean/var) for BatchNorm layers in the model **only if**
    any values are NaN.
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            has_nan = (
                torch.isnan(module.running_mean).any() or
                torch.isnan(module.running_var).any()
            )
            if has_nan:
                print(f"Resetting corrupted stats in: {module.__class__.__name__}")
                module.reset_running_stats()

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

def load_submodule_prefix(model, prefix : str, state_dict: dict):
    state_dict = {
        k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)
    }
    load_state_dict_mismatch(model, state_dict)

def visualize_phoneme_probabilities(
    phoneme_logits: torch.Tensor,
    phoneme_class_names: list[str],
    start_step: int = 0,
    window_size: int = 50,
    figsize: tuple = (15, 10),
    title: str = "Phoneme Probabilities Over Time Window",
    max_xticks: int = 20 # Max number of x-axis ticks to display for clarity
) -> None:
    """
    Visualizes phoneme probabilities over a sequence window using a heatmap.

    Args:
        phoneme_logits (torch.Tensor): A PyTorch tensor of shape [1, seq, classes]
                                       containing raw logits for phonemes.
        phoneme_class_names (list[str]): A list of strings representing the names
                                         of the phoneme classes. It's assumed these
                                         correspond to the first N indices of the
                                         'classes' dimension in phoneme_logits.
        start_step (int, optional): The starting time step of the window to display.
                                    Defaults to 0.
        window_size (int, optional): The number of time steps to include in the window.
                                     If None, displays the full sequence from start_step.
                                     Defaults to 50.
        figsize (tuple, optional): Size of the matplotlib figure. Defaults to (15, 10).
        title (str, optional): Title of the plot. Defaults to "Phoneme Probabilities Over Time Window".
        max_xticks (int, optional): Maximum number of x-axis ticks to display. Helps prevent clutter.
                                    Defaults to 20.
    """
    if not isinstance(phoneme_logits, torch.Tensor):
        raise TypeError("phoneme_logits must be a PyTorch Tensor.")
    if phoneme_logits.ndim != 3 or phoneme_logits.shape[0] != 1:
        raise ValueError("phoneme_logits must have shape [1, seq, classes]. "
                         f"Got {phoneme_logits.shape}")
    if not phoneme_class_names:
        raise ValueError("phoneme_class_names list cannot be empty.")
    
    num_actual_phonemes = len(phoneme_class_names)
    if num_actual_phonemes > phoneme_logits.shape[2]:
        raise ValueError(f"Number of phoneme_class_names ({num_actual_phonemes}) "
                         f"cannot exceed total classes in logits ({phoneme_logits.shape[2]}).")

    phoneme_class_names = phoneme_class_names + ["<unk>"] * (
        phoneme_logits.shape[2] - num_actual_phonemes)

    # 1. Convert logits to probabilities
    #    Softmax over the classes dimension (dim=2)
    probabilities = torch.softmax(phoneme_logits, dim=2)

    # 2. Squeeze the batch dimension (shape: [seq, classes])
    probabilities_squeezed = probabilities.squeeze(0)

    # 3. Select only the probabilities for the actual phoneme classes
    #    Assumes phoneme classes are the first N entries.
    # phoneme_probabilities = probabilities_squeezed[:, :num_actual_phonemes]
    phoneme_probabilities = probabilities_squeezed

    # 4. Detach from graph, move to CPU (if on GPU), and convert to NumPy
    phoneme_probabilities_np = phoneme_probabilities.detach().cpu().numpy()

    # 5. Determine the window to display
    full_seq_len = phoneme_probabilities_np.shape[0]

    # Adjust start_step if out of bounds
    start_step = max(0, min(start_step, full_seq_len - 1 if full_seq_len > 0 else 0))

    if window_size is None:
        current_window_size = full_seq_len - start_step
    else:
        current_window_size = window_size
    
    end_step = min(start_step + current_window_size, full_seq_len)
    
    # Slice the data for the window
    windowed_data = phoneme_probabilities_np[start_step:end_step, :] # Shape: [window_len, num_phonemes]

    if windowed_data.shape[0] == 0:
        print("Warning: The specified window is empty or out of bounds. Nothing to display.")
        return

    # 6. Transpose data for heatmap (phonemes on y-axis, time on x-axis)
    #    Shape: [num_phonemes, window_len]
    heatmap_data = windowed_data.T

    # 7. Prepare plot
    plt.figure(figsize=figsize)
    
    # Generate x-tick labels (actual time steps)
    x_tick_labels_full = list(range(start_step, end_step))
    
    # Determine which x-ticks to display to avoid clutter
    num_timesteps_in_window = heatmap_data.shape[1]
    if num_timesteps_in_window <= max_xticks:
        # Show all ticks if they are few enough
        xticklabels_param = x_tick_labels_full
    else:
        # Show sparse ticks, ensuring the labels are the actual time steps
        tick_indices = np.linspace(0, num_timesteps_in_window - 1, max_xticks, dtype=int)
        xticklabels_param = [x_tick_labels_full[i] if i < len(x_tick_labels_full) else "" for i in tick_indices]


    ax = sns.heatmap(
        heatmap_data,
        yticklabels=phoneme_class_names,
        xticklabels=xticklabels_param,
        cmap="viridis",  # A perceptually uniform colormap
        cbar_kws={'label': 'Probability'}
    )
    
    plt.xlabel(f"Time Step (Window: {start_step} to {end_step-1})")
    plt.ylabel("Phoneme Class")
    plt.title(title)
    
    # Rotate x-axis tick labels if they are numeric and long
    if all(isinstance(label, (int, float)) or (isinstance(label, str) and label.isdigit()) for label in xticklabels_param):
         ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout() # Adjust plot to prevent labels from overlapping
    plt.show()