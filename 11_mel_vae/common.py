import numpy as np
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from PIL import Image
from omegaconf import OmegaConf

# Normalization
def normalize(config: OmegaConf, log_mel: torch.Tensor) -> torch.Tensor:
    return (log_mel - config.stats.mean) / config.stats.std

def denormalize(config: OmegaConf, norm_log_mel: torch.Tensor) -> torch.Tensor:
    return norm_log_mel * config.stats.std + config.stats.mean

# Pseudo-Huber loss between tensors
# Uses parameters from https://openreview.net/pdf?id=FmqFfMTNnv
# Can apply weights and return per-sample or batch average Pseudo-Huber
# Parameters:  
#   x, y: input tensors
#   w: weights tensor (optional)
#   mean: whether to return batch average or per-sample Pseudo-Huber
#   c: parameter for the 'smoothness' of the loss function
# Returns:
#   loss: Pseudo-Huber loss
def huber(x,y, w=None, mean=True, c=0.00054):
    diff = torch.flatten((x-y)**2, start_dim=1)
    data_dim = diff.shape[-1]
    c = c*torch.sqrt(torch.ones((1,), device=x.device)*data_dim)
    diff = torch.sum(diff, -1)
    diff = torch.sqrt(diff+c**2)-c
    diff = torch.nan_to_num(diff)
    if w is not None:
        diff = diff*w.squeeze()
    if mean:
        return diff.mean()
    else:
        return diff

def create_half_mask(latent_dim: int, use_torch: bool = True) -> torch.Tensor | np.ndarray:
    """
    Creates a binary mask splitting the dimensions approximately in half.

    The first floor(latent_dim / 2) dimensions are marked 0 (identity),
    and the remaining dimensions are marked 1 (transformed).

    Args:
        latent_dim: The total number of latent dimensions.
        use_torch: If True, returns a torch.Tensor. Otherwise, returns a numpy.ndarray.

    Returns:
        A mask of shape (latent_dim,) with 0s and 1s.
    """
    if not isinstance(latent_dim, int) or latent_dim <= 0:
        raise ValueError("latent_dim must be a positive integer.")

    num_identity = latent_dim // 2  # Floor division
    num_transformed = latent_dim - num_identity

    mask = np.concatenate([
        np.zeros(num_identity, dtype=np.uint8),
        np.ones(num_transformed, dtype=np.uint8)
    ])

    if use_torch:
        # It's often convenient to have the mask directly as a tensor
        # Using uint8 is memory efficient; bool is also an option
        return torch.from_numpy(mask).to(torch.uint8)
    else:
        return mask

import matplotlib.pyplot as plt
import io
from torchvision.transforms import ToTensor
def mel_to_img(log_mel, eps=1e-5):
    fig, ax = plt.subplots()
    log_mel = log_mel.squeeze()
    
    # Convert to numpy
    mel_data = log_mel.T[:, :].detach().cpu().numpy()
    
    title = "Log Mel Spectrogram"
    
    # Use a better colormap
    im = ax.imshow(mel_data, aspect='auto', origin='lower',
                  cmap='viridis', interpolation='none')
    
    ax.set_title(title)
    
    # Add a colorbar for reference
    fig.colorbar(im, ax=ax)
    
    # Ensure the figure has no extra white space
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    image = ToTensor()(plt.imread(buf))
    plt.close()
    return image

def log_feature_maps_matplotlib(
    logger: TensorBoardLogger,
    feature_maps: list[torch.Tensor],
    tag_prefix: str, # e.g., "Real" or "Fake"
    global_step: int,
    cmap: str = 'viridis', # Choose a matplotlib colormap (e.g., 'viridis', 'plasma', 'magma', 'gray')
    max_batch_items: int = 4, # How many samples from the batch to visualize
    max_channels_per_map: int = 6, # Max channels per feature map to show
    figsize_scale: float = 1.5 # Adjust overall figure size
):
    """
    Logs intermediate feature maps from a discriminator to TensorBoard as image grids,
    rendered using Matplotlib with a specified colormap.

    Args:
        logger: An instance of pytorch_lightning.loggers.TensorBoardLogger.
        feature_maps: A list of tensors [feat_map_1, feat_map_2, ..., final_output],
                      where each tensor has shape (B, C, H, W).
        tag_prefix: String prefix for TensorBoard tags (e.g., "Real", "Fake").
        global_step: The current global training step.
        cmap: Matplotlib colormap name to use for visualizing feature activations.
        max_batch_items: Maximum number of batch items to include in the grid.
        max_channels_per_map: Maximum number of channels to visualize for each feature map.
        figsize_scale: Scaling factor for the generated matplotlib figure size.
    """
    if not isinstance(logger, TensorBoardLogger):
        print(f"Warning: Logger is not a TensorBoardLogger (got {type(logger)}). Skipping feature map visualization.")
        return
    if not feature_maps:
        print("Warning: No feature maps provided. Skipping visualization.")
        return

    try:
        writer = logger.experiment # Get the underlying SummaryWriter
    except AttributeError:
        print(f"Warning: Could not get SummaryWriter from logger. Skipping feature map visualization.")
        return

    # Ensure we don't try to visualize more items/channels than available
    if not feature_maps[0].shape[0] > 0:
         print("Warning: Feature maps have batch size 0. Skipping visualization.")
         return

    actual_batch_items = min(feature_maps[0].shape[0], max_batch_items)

    for i, feat_map in enumerate(feature_maps):
        # Detach, move to CPU (safer for visualization/numpy)
        feat_map_detached = feat_map.detach().cpu()

        if feat_map_detached.shape[1] == 0: # Skip if a layer somehow has 0 channels
             continue

        actual_channels = min(feat_map_detached.shape[1], max_channels_per_map)
        if actual_channels == 0:
            continue

        # Select subset of batch items and channels
        selected_maps_np = feat_map_detached[:actual_batch_items, :actual_channels, :, :].numpy()
        # Shape: (N, K, H, W) where N=actual_batch_items, K=actual_channels

        # Determine grid size for matplotlib
        n_rows = actual_batch_items
        n_cols = actual_channels
        fig_height = n_rows * figsize_scale
        fig_width = n_cols * figsize_scale * (selected_maps_np.shape[3] / selected_maps_np.shape[2] if selected_maps_np.shape[2] > 0 else 1) # Adjust width by aspect ratio


        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False) # Ensure axes is always 2D array
        fig.patch.set_facecolor('white') # Set background to white for better visibility if features are dark

        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                ax = axes[row_idx, col_idx]
                feature_slice = selected_maps_np[row_idx, col_idx, :, :]
                im = ax.imshow(feature_slice, cmap=cmap, aspect='auto') # Use selected colormap
                ax.set_xticks([])
                ax.set_yticks([])
                # Optional: Add titles to identify batch item and channel
                # ax.set_title(f"B{row_idx} C{col_idx}", fontsize=8)

        # Determine layer name
        layer_name = f"Layer_{i+1}"
        if i == len(feature_maps) - 1:
            layer_name = "Final_Output" # The last map is the raw prediction

        # Construct the tag for TensorBoard
        tag = f"{tag_prefix}_Features_MPL/{layer_name} (Top {actual_batch_items} items, First {actual_channels} ch.)"

        # Adjust layout and save figure to a buffer
        plt.subplots_adjust(wspace=0.05, hspace=0.05) # Reduce spacing between plots
        #plt.tight_layout(pad=0.1) # Alternative layout adjustment

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.02) # Save as PNG to buffer
        buf.seek(0) # Reset buffer position to the beginning

        # Read the PNG buffer using PIL and convert to tensor
        img = Image.open(buf)
        img_tensor = ToTensor(img) # Converts PIL image (H, W, C) or (H,W) to (C, H, W) tensor

        # Log the image tensor
        writer.add_image(tag, img_tensor, global_step=global_step)

        # Close the figure to free memory
        plt.close(fig)
        buf.close()