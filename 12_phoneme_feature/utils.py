import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image # For converting plot to image array


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

def plot_phoneme_probabilities(
    logits_tensor: torch.Tensor,
    phoneme_list: list[str],
    return_image: bool = False,
    title: str = "Phoneme Probabilities Over Time",
    xlabel: str = "Time Step (T)",
    ylabel: str = "Phoneme Class",
    cmap: str = "viridis",
    figsize: tuple[float, float] = (12, 6),
    xtick_interval: int = 10
) -> np.ndarray | None:
    """
    Visualizes phoneme probabilities as a heatmap.

    Args:
        logits_tensor (torch.Tensor): Tensor of phoneme logits.
            Expected shape: [1, T, C] (batch, time, classes) or [T, C].
        phoneme_list (list[str]): List of phoneme names, length C.
        return_image (bool, optional): If True, returns the plot as an HWC NumPy array
            (suitable for TensorBoard's add_image). If False, displays the plot using
            plt.show(). Defaults to False.
        title (str, optional): Title of the plot.
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis.
        cmap (str, optional): Colormap for the heatmap. Defaults to "viridis".
        figsize (tuple[float, float], optional): Figure size. Defaults to (12, 6).
        xtick_interval (int, optional): Interval for x-axis ticks.
            Shows a label every `xtick_interval` time steps. Defaults to 10.

    Returns:
        np.ndarray | None: If return_image is True, returns an HWC NumPy array of the plot.
                           Otherwise, returns None.
    """
    if not isinstance(logits_tensor, torch.Tensor):
        raise TypeError("logits_tensor must be a torch.Tensor")
    if not isinstance(phoneme_list, list):
        raise TypeError("phoneme_list must be a list of strings")

    # --- 1. Convert Logits to Probabilities ---
    probabilities_tensor = torch.softmax(logits_tensor, dim=-1)

    # --- 2. Prepare Data for Plotting ---
    if probabilities_tensor.ndim == 3:
        probabilities_tensor = probabilities_tensor.squeeze(0) # Remove batch dim if present
    elif probabilities_tensor.ndim != 2:
        raise ValueError(f"logits_tensor must have 2 or 3 dimensions, got {logits_tensor.ndim}")

    probabilities_np = probabilities_tensor.cpu().numpy() # Shape: [T, C]
    T, C = probabilities_np.shape

    if len(phoneme_list) != C:
        raise ValueError(f"Length of phoneme_list ({len(phoneme_list)}) "
                         f"must match number of classes in logits_tensor ({C})")

    # Transpose for heatmap: [C, T] so phonemes are on y-axis, time on x-axis
    probabilities_for_plot = probabilities_np.T

    # --- 3. Visualization ---
    fig, ax = plt.subplots(figsize=figsize) # Use fig, ax for more control

    # Determine x-tick labels
    if T <= xtick_interval * 2: # If sequence is short, show all or more ticks
        xticklabels = True # Seaborn will decide or show all
    else:
        xticklabels = xtick_interval

    sns.heatmap(
        probabilities_for_plot,
        xticklabels=xticklabels,
        yticklabels=phoneme_list,
        cmap=cmap,
        cbar_kws={'label': 'Probability'},
        ax=ax # Plot on the created Axes object
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()

    if return_image:
        # Save the plot to an in-memory buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        # Open as PIL image and convert to NumPy array
        img = Image.open(buf).convert('RGB') # Ensure 3 channels (RGB)
        img_np = np.array(img)
        buf.close()
        plt.close(fig) # Close the figure to free memory
        return img_np # Expected format HWC for TensorBoard
    else:
        plt.show()
        plt.close(fig) # Close the figure after showing
        return None

def greedy_decode_phonemes(
    logits_tensor: torch.Tensor,
    phoneme_list: list[str],
    separator: str = " "
) -> str:
    """
    Performs greedy decoding on phoneme logits to produce a sequence string.

    Args:
        logits_tensor (torch.Tensor): Tensor of phoneme logits.
            Expected shape: [1, T, C] (batch, time, classes) or [T, C].
        phoneme_list (list[str]): List of phoneme names, length C.
        separator (str, optional): The string used to join phoneme symbols.
            Defaults to " ".

    Returns:
        str: A string representing the decoded phoneme sequence.

    Raises:
        TypeError: If logits_tensor is not a torch.Tensor or phoneme_list is not a list.
        ValueError: If logits_tensor has an unsupported number of dimensions or
                    if the length of phoneme_list does not match the class dimension
                    of the logits_tensor.
    """
    if not isinstance(logits_tensor, torch.Tensor):
        raise TypeError("logits_tensor must be a torch.Tensor")
    if not isinstance(phoneme_list, list):
        raise TypeError("phoneme_list must be a list of strings")

    # Handle tensor dimensions
    if logits_tensor.ndim == 3:
        if logits_tensor.shape[0] == 1:
            # Squeeze the batch dimension if it's 1
            processed_logits = logits_tensor.squeeze(0) # Shape: [T, C]
        else:
            raise ValueError(
                "logits_tensor with 3 dimensions must have batch size 1 (shape [1, T, C]). "
                f"Got batch size {logits_tensor.shape[0]}."
            )
    elif logits_tensor.ndim == 2:
        processed_logits = logits_tensor # Shape: [T, C]
    else:
        raise ValueError(
            f"logits_tensor must have 2 or 3 dimensions ([T, C] or [1, T, C]), "
            f"got {logits_tensor.ndim} dimensions."
        )

    num_classes = processed_logits.shape[-1]
    if len(phoneme_list) != num_classes:
        raise ValueError(
            f"Length of phoneme_list ({len(phoneme_list)}) "
            f"must match the number of classes in logits_tensor ({num_classes})."
        )

    # Perform greedy decoding (argmax along the class dimension)
    # Softmax is not strictly needed for argmax, as argmax(logits) == argmax(softmax(logits))
    predicted_indices = torch.argmax(processed_logits, dim=-1) # Shape: [T]

    # Convert indices to phoneme symbols
    # .cpu() is important if the tensor is on GPU
    # .tolist() converts the tensor to a Python list of numbers
    decoded_phonemes = [phoneme_list[idx] for idx in predicted_indices.cpu().tolist()]

    # Join the phonemes into a single string
    return separator.join(decoded_phonemes)
