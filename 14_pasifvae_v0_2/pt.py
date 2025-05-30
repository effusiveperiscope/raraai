import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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

    # 1. Convert logits to probabilities
    #    Softmax over the classes dimension (dim=2)
    probabilities = torch.softmax(phoneme_logits, dim=2)

    # 2. Squeeze the batch dimension (shape: [seq, classes])
    probabilities_squeezed = probabilities.squeeze(0)

    # 3. Select only the probabilities for the actual phoneme classes
    #    Assumes phoneme classes are the first N entries.
    phoneme_probabilities = probabilities_squeezed[:, :num_actual_phonemes]

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

if __name__ == '__main__':
    # --- Example Usage ---

    # Define some parameters
    seq_length = 100
    num_phoneme_types = 20  # e.g., 'AA', 'AE', 'AH', ...
    num_control_tokens = 3  # e.g., BOS, EOS, PAD
    total_classes = num_phoneme_types + num_control_tokens

    # Generate dummy phoneme class names
    phoneme_names = [f"PH_{i:02d}" for i in range(num_phoneme_types)]
    # Add some more descriptive names for a few
    phoneme_names[0] = "SIL"
    phoneme_names[1] = "AA"
    phoneme_names[2] = "AE"
    phoneme_names[3] = "AH"


    # Generate dummy logit data
    # Shape: [1, seq_length, total_classes]
    # Let's make some phonemes more probable at certain times for a more interesting plot
    dummy_logits = torch.randn(1, seq_length, total_classes) * 0.5
    
    # Make phoneme 'AA' (index 1) peak around step 10-20
    dummy_logits[0, 10:20, 1] += 3.0 
    # Make phoneme 'AH' (index 3) peak around step 30-35 and 70-75
    dummy_logits[0, 30:35, 3] += 4.0
    dummy_logits[0, 70:75, 3] += 3.5
    # Make 'SIL' (index 0) generally probable but less so during peaks
    dummy_logits[0, :, 0] += 1.0
    dummy_logits[0, 50:60, num_phoneme_types + 0] += 5.0 # A control token (e.g. PAD) becomes probable

    print(f"Phoneme logits shape: {dummy_logits.shape}")
    print(f"Phoneme class names (first few): {phoneme_names[:5]} (Total: {len(phoneme_names)})")

    # --- Test Cases ---

    # 1. Visualize a small window
    print("\nVisualizing a small window (20-50):")
    visualize_phoneme_probabilities(
        dummy_logits,
        phoneme_names,
        start_step=20,
        window_size=30,
        title="Phoneme Probabilities (Window: 20-50)"
    )

    # 2. Visualize the full sequence (by setting window_size to cover it, or None)
    print("\nVisualizing the full sequence (0-100):")
    visualize_phoneme_probabilities(
        dummy_logits,
        phoneme_names,
        start_step=0,
        window_size=seq_length, # or None
        title="Phoneme Probabilities (Full Sequence)",
        figsize=(20, 8) # Wider for full sequence
    )

    # 3. Visualize a window near the end
    print("\nVisualizing a window near the end (80-100):")
    visualize_phoneme_probabilities(
        dummy_logits,
        phoneme_names,
        start_step=80,
        window_size=20,
        title="Phoneme Probabilities (Window: 80-100)"
    )
    
    # 4. Test with a very small window_size
    print("\nVisualizing a very small window (5-10):")
    visualize_phoneme_probabilities(
        dummy_logits,
        phoneme_names,
        start_step=5,
        window_size=5,
        title="Phoneme Probabilities (Window: 5-10)"
    )

    # 5. Test with window_size=None (full sequence from start_step)
    print("\nVisualizing with window_size=None (from step 0):")
    visualize_phoneme_probabilities(
        dummy_logits,
        phoneme_names,
        start_step=0,
        window_size=None, 
        title="Phoneme Probabilities (Full Sequence from step 0, window_size=None)",
        figsize=(20,8)
    )
    
    # 6. Test with start_step out of bounds (should adjust)
    print("\nVisualizing with start_step out of bounds (should start at end):")
    visualize_phoneme_probabilities(
        dummy_logits,
        phoneme_names,
        start_step=seq_length + 10, # e.g. 110
        window_size=10, 
        title="Phoneme Probabilities (Out of Bounds Start)"
    )
    
    # 7. Test with small number of phonemes
    short_phoneme_list = ["S_A", "S_B", "S_C"]
    short_logits = torch.randn(1, 50, len(short_phoneme_list) + 2) # 2 control tokens
    short_logits[0, 10:15, 0] += 3.0
    short_logits[0, 20:25, 1] += 3.0
    print("\nVisualizing with few phonemes:")
    visualize_phoneme_probabilities(
        short_logits,
        short_phoneme_list,
        start_step=0,
        window_size=50,
        title="Phoneme Probabilities (Few Phonemes)"
    )

    # Edge case: empty phoneme_class_names (should raise error)
    try:
        print("\nTesting with empty phoneme_class_names (expect ValueError):")
        visualize_phoneme_probabilities(dummy_logits, [])
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # Edge case: num_phoneme_class_names > total_classes in logits (should raise error)
    try:
        print("\nTesting with too many phoneme_class_names (expect ValueError):")
        extra_phoneme_names = phoneme_names + ["EXTRA_PHONEME"] * (total_classes + 5 - num_phoneme_types)
        visualize_phoneme_probabilities(dummy_logits, extra_phoneme_names)
    except ValueError as e:
        print(f"Caught expected error: {e}")