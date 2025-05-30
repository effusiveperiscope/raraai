import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image # For converting plot to image array

# Ensure matplotlib backend is suitable for non-interactive environments
# if you plan to run this in scripts without a display.
# import matplotlib
# matplotlib.use('Agg') # 'Agg' is a good backend for writing to files/buffers

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

# --- Example Usage ---
if __name__ == "__main__":
    # --- Sample Data ---
    T_seq = 60  # Sequence length
    C_classes = 12  # Number of phoneme classes
    phonemes = [f"P{i:02d}" for i in range(C_classes)]

    # Generate some random logits
    example_logits = torch.randn(1, T_seq, C_classes)
    # Make some phonemes more likely at certain times for a better visual.
    for i in range(C_classes):
        start_t = (T_seq // C_classes) * i
        end_t = min(T_seq, (T_seq // C_classes) * (i + 1))
        example_logits[0, start_t:end_t, i] += 5.0 # Boost logits

    # Option 1: Display the plot
    print("Displaying plot...")
    plot_phoneme_probabilities(
        example_logits,
        phonemes,
        return_image=False,
        figsize=(15, C_classes * 0.5), # Adjust figsize based on C
        xtick_interval=5
    )

    # Option 2: Get the plot as an image for TensorBoard
    print("\nGenerating image for TensorBoard...")
    image_array = plot_phoneme_probabilities(
        example_logits,
        phonemes,
        return_image=True,
        title="Phoneme Heatmap for TensorBoard",
        figsize=(15, C_classes * 0.5)
    )

    if image_array is not None:
        print(f"Image array shape: {image_array.shape}, dtype: {image_array.dtype}")
        # This image_array can now be used with TensorBoard SummaryWriter
        # e.g., writer.add_image("Phoneme_Probabilities", image_array, global_step=0, dataformats='HWC')

        # If you have TensorBoard and PyTorch installed, you can test it:
        try:
            from torch.utils.tensorboard import SummaryWriter
            import shutil
            import os

            log_dir = "runs/phoneme_plot_test"
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir) # Clean up previous run

            writer = SummaryWriter(log_dir)
            writer.add_image(
                "Phoneme_Probabilities/Heatmap",
                image_array,
                global_step=0,
                dataformats='HWC' # Height, Width, Channels
            )
            print(f"TensorBoard log written to: {log_dir}")
            print(f"Run: tensorboard --logdir={os.path.abspath(log_dir)}")
            writer.close()
        except ImportError:
            print("torch.utils.tensorboard not found. Skipping TensorBoard example.")
        except Exception as e:
            print(f"Error during TensorBoard logging: {e}")

    # Example with a 2D tensor (no batch dimension)
    print("\nDisplaying plot for 2D logits tensor...")
    example_logits_2d = torch.randn(T_seq, C_classes)
    for i in range(C_classes):
        start_t = (T_seq // C_classes) * i
        end_t = min(T_seq, (T_seq // C_classes) * (i + 1))
        example_logits_2d[start_t:end_t, i] += 5.0

    plot_phoneme_probabilities(
        example_logits_2d,
        phonemes,
        return_image=False,
        title="Phoneme Probabilities (2D input)",
        figsize=(15, C_classes * 0.5)
    )