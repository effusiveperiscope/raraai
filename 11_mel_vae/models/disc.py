import math
from models.common import (DepthwiseSeparableConv2d)
import torch.nn as nn
import torch

def collapse_mask_to_lengths(mask: torch.Tensor) -> torch.Tensor:
  """
  Collapses a boolean sequence mask to a tensor of lengths.

  Args:
    mask: A boolean tensor of shape [B, T], where True indicates a non-padding element.

  Returns:
    A long tensor of shape [B,] containing the length of each sequence.
  """
  return torch.sum(mask.long(), dim=-1)

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid_channel = nn.Sigmoid()

        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_attn = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_attn

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = self.spatial(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_attn
        return x

# PatchGAN style discriminator
class DepthwisePatchGANDiscriminator(nn.Module):
    def __init__(self, config, use_spectral_norm=True):
        super().__init__()
        cfg = config.model.discriminator

        self.conv_params = []
        self.layers = nn.ModuleList()
        in_channels = 1

        # Construct layers from config
        for out_channels, ks, s, p in zip(cfg.channel_sizes + [1], cfg.kernel_sizes, cfg.strides, cfg.paddings):
            conv = DepthwiseSeparableConv2d(
                in_channels, out_channels,
                kernel_size=ks, stride=s, padding=p,
                use_spectral_norm=use_spectral_norm
            )
            self.layers.append(conv)

            if out_channels != 1:  # Don't apply activation to the final layer
                self.layers.append(nn.SiLU())

            if getattr(cfg, 'use_attention', False) and out_channels != 1 and in_channels != 1:
                self.layers.append(CBAM(out_channels))

            self.conv_params.append({'kernel_size': ks, 'stride': s, 'padding': p})
            in_channels = out_channels

        self.apply(self._initialize_weights)

    # Define the initialization function
    def _initialize_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            # Check if spectral norm was applied (it renames weight to weight_orig)
            weight = getattr(m, 'weight_orig', m.weight)
            # Use Kaiming Normal for SiLU (approximated by relu non-linearity)
            nn.init.kaiming_normal_(weight, mode='fan_in', nonlinearity='leaky_relu', a=0.01) # SiLU is closer to leaky_relu(0.01) or relu
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, channels) or (B, 1, T_max, C) - Input feature map.
            Assumes input is padded to T_max along seq_len dimension.

        Returns:
            list: Feature maps [feat_map_1, feat_map_2, ..., final_output_map]
        """
        if x.dim() == 3:  # (B, T, C)
            x = x.unsqueeze(1)  # (B, 1, T, C)
        elif x.dim() != 4:
            raise ValueError(f"Input must be 3D (B, T, C) or 4D (B, 1, T, C), got {x.dim()}D")

        feature_maps = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Save output after each activation (i.e., after every conv + activation pair)
            if isinstance(layer, nn.SiLU) or (i == len(self.layers) - 1):  # Save final output too
                feature_maps.append(x)

        return feature_maps

def calculate_feature_map_mask(
    conv_params_list,
    layer_index,
    input_lengths,
    input_feature_dim,
    max_input_length, 
    device
):
    """
    Calculates a mask for a specific feature map based on input sequence lengths,
    considering a maximum possible padded input length.

    Args:
        conv_params_list (list): List of dicts containing conv parameters
                                 (kernel_size, stride, padding) up to the desired layer.
        layer_index (int): The index of the feature map in the output list (0-based).
                           Mask will be calculated based on convolutions *up to and including*
                           the one producing this feature map.
        input_lengths (torch.Tensor): 1D tensor (batch_size,) of original sequence lengths.
        input_feature_dim (int): Size of the feature dimension (width) of the input.
        max_input_length (int): The maximum length the input sequences might be padded to
                                before entering the convolutional layers. This determines
                                the output mask's H dimension.
        device (torch.device): Device for tensor creation.

    Returns:
        torch.Tensor: Boolean mask (batch_size, 1, H_layer, W_layer). H_layer and W_layer
                      are determined by processing an input of size
                      (max_input_length, input_feature_dim).
    """
    batch_size = input_lengths.shape[0]

    # --- Helper function to apply conv formula ---
    def get_output_len(input_len, kernel, stride, padding):
        # Ensure input_len is float for calculation if it's a tensor
        if isinstance(input_len, torch.Tensor):
            input_len = input_len.float()
        else: # Assume scalar like max_input_length
            input_len = float(input_len)
        # Use torch.floor for tensors, math.floor for scalars
        floor_func = torch.floor if isinstance(input_len, torch.Tensor) else math.floor
        out = floor_func((input_len + 2 * padding - kernel) / stride) + 1
        return out

    # --- Calculate target output dimensions based on max_input_length ---
    target_h = float(max_input_length)
    target_w = float(input_feature_dim)

    num_convs_to_apply = layer_index + 1
    if num_convs_to_apply > len(conv_params_list):
         raise IndexError(f"layer_index {layer_index} is out of bounds for {len(conv_params_list)} conv layers")

    for i in range(num_convs_to_apply):
        params = conv_params_list[i]
        kH = params['kernel_size'] if isinstance(params['kernel_size'], int) else params['kernel_size'][0]
        kW = params['kernel_size'] if isinstance(params['kernel_size'], int) else params['kernel_size'][1]
        sH = params['stride'] if isinstance(params['stride'], int) else params['stride'][0]
        sW = params['stride'] if isinstance(params['stride'], int) else params['stride'][1]
        pH = params['padding'] if isinstance(params['padding'], int) else params['padding'][0]
        pW = params['padding'] if isinstance(params['padding'], int) else params['padding'][1]

        target_h = get_output_len(target_h, kH, sH, pH)
        target_w = get_output_len(target_w, kW, sW, pW)

    # Final target dimensions (integers)
    target_h = max(0, int(target_h))
    target_w = max(0, int(target_w))

    # --- Calculate actual output lengths for each item in the batch ---
    current_h = input_lengths.float().clone()
    current_w = torch.full_like(current_h, float(input_feature_dim)) # Width starts same for all

    for i in range(num_convs_to_apply):
        params = conv_params_list[i]
        kH = params['kernel_size'] if isinstance(params['kernel_size'], int) else params['kernel_size'][0]
        kW = params['kernel_size'] if isinstance(params['kernel_size'], int) else params['kernel_size'][1]
        sH = params['stride'] if isinstance(params['stride'], int) else params['stride'][0]
        sW = params['stride'] if isinstance(params['stride'], int) else params['stride'][1]
        pH = params['padding'] if isinstance(params['padding'], int) else params['padding'][0]
        pW = params['padding'] if isinstance(params['padding'], int) else params['padding'][1]

        # Use tensor-based calculation here
        current_h = get_output_len(current_h, kH, sH, pH)
        current_w = get_output_len(current_w, kW, sW, pW)

    # Valid lengths for each batch item
    output_lengths_h = torch.clamp(current_h, min=0).long()
    output_lengths_w = torch.clamp(current_w, min=0).long() # Should be same for all if input_feature_dim is fixed

    # --- Create the mask ---

    # Handle cases where target output size might become zero
    if target_h <= 0 or target_w <= 0:
        # Return an empty mask of the correct target shape but zero size in spatial dims
        return torch.zeros(batch_size, 1, target_h, target_w, dtype=torch.bool, device=device)

    # Create index tensors based on the target dimensions
    idx_h = torch.arange(target_h, device=device).unsqueeze(0).unsqueeze(-1) # (1, H_target, 1)
    idx_w = torch.arange(target_w, device=device).unsqueeze(0).unsqueeze(0)  # (1, 1, W_target)

    # Reshape valid lengths for broadcasting
    # (B,) -> (B, 1, 1)
    valid_h = output_lengths_h.unsqueeze(-1).unsqueeze(-1)
    valid_w = output_lengths_w.unsqueeze(-1).unsqueeze(-1)

    # Compare indices with valid lengths for each batch item
    # Broadcasting:
    # idx_h:      (1, H_target, 1)
    # valid_h:    (B, 1,       1)
    # -> mask_h:  (B, H_target, 1)
    mask_h = (idx_h < valid_h)

    # idx_w:      (1, 1,       W_target)
    # valid_w:    (B, 1,       1)
    # -> mask_w:  (B, 1,       W_target)
    mask_w = (idx_w < valid_w)

    # Combine H and W masks and add channel dimension
    # mask_h.unsqueeze(-1): (B, H_target, 1,        1)
    # mask_w.unsqueeze(1):  (B, 1,        1,        W_target) -> Need correction here
    # Let's recalculate combination:
    # mask_h: (B, H_target, 1) -> broadcast to (B, H_target, W_target)
    # mask_w: (B, 1, W_target) -> broadcast to (B, H_target, W_target)
    # Result: (B, H_target, W_target)
    mask = (mask_h & mask_w)

    # Add the channel dimension
    mask = mask.unsqueeze(1) # (B, 1, H_target, W_target)

    return mask 

def calculate_masked_feature_matching_loss(
    discriminator,
    real_features_list,
    fake_features_list,
    real_input_lengths,
    max_input_length,
    input_feature_dim,
    device,
    loss_type='l1' # 'l1' or 'l2'
):
    """
    Calculates feature matching loss, masking based on real input lengths.

    Args:
        discriminator (nn.Module): The discriminator instance (used to get conv params).
        real_features_list (list): List of feature maps from D(real_data).
        fake_features_list (list): List of feature maps from D(fake_data).
        real_input_lengths (torch.Tensor): Original lengths of real data sequences.
        max_input_length (int): Maximum length of real data sequences, including padding.
        input_feature_dim (int): Feature dimension (width) of the initial input.
        device (torch.device): Device for calculations.
        loss_type (str): 'l1' for Mean Absolute Error, 'l2' for Mean Squared Error.

    Returns:
        torch.Tensor: Scalar feature matching loss.
    """
    total_fm_loss = 0.0
    num_feature_maps = len(real_features_list) - 1 # Exclude final output map usually

    conv_params_list = discriminator.conv_params

    for i in range(num_feature_maps):
        # Get feature maps for this layer
        real_feat = real_features_list[i]
        fake_feat = fake_features_list[i]

        # Calculate mask based on *real* data lengths for this layer's dimensions
        # Assumption: Generator should match real features in valid real regions.
        mask = calculate_feature_map_mask(
            conv_params_list=conv_params_list,
            layer_index=i,
            input_lengths=real_input_lengths,
            max_input_length=max_input_length,
            input_feature_dim=input_feature_dim,
            device=device
        )


        # Ensure mask dimensions match feature map dimensions
        if mask.shape[2:] != real_feat.shape[2:]:
            # This can happen if input is extremely short, leading to zero-size output dims
             if mask.numel() == 0 and real_feat.numel() > 0 and mask.shape[0] == real_feat.shape[0]:
                 # If mask is empty (due to calculated output size <= 0) but feature map isn't,
                 # skip loss calculation for this layer as there are no valid elements.
                 continue # Skip to next layer
             elif mask.numel() > 0 and real_feat.numel() == 0:
                  # If feature map is empty but mask isn't (less likely), also skip.
                  continue
             else:
                # Otherwise, it's an unexpected mismatch
                raise ValueError(f"Mask shape {mask.shape} doesn't match feature map shape {real_feat.shape} at layer {i}")


        # Calculate difference
        if loss_type == 'l1':
            diff = torch.abs(real_feat - fake_feat)
        elif loss_type == 'l2':
            diff = (real_feat - fake_feat) ** 2
        else:
            raise ValueError("loss_type must be 'l1' or 'l2'")

        # Apply mask and calculate mean loss over valid elements
        masked_diff = diff * mask.float() # Mask has shape (B, 1, H, W), diff has (B, C, H, W) - broadcasts over C

        # Sum over all dimensions (B, C, H, W) and divide by number of valid elements
        # Number of valid elements = mask.sum() * C (channels)
        num_valid_elements = mask.sum() * real_feat.shape[1] + 1e-8 # Add epsilon for stability

        layer_loss = masked_diff.sum() / num_valid_elements
        total_fm_loss += layer_loss

    return total_fm_loss

# --- Example Usage ---
if __name__ == '__main__':
    # Example input setup
    batch_size = 4
    max_time_frames = 100
    n_mels = 80
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Simulate REAL data batch
    real_lengths = torch.tensor([100, 80, 95, 70], device=device)
    real_input_padded = torch.randn(batch_size, max_time_frames, n_mels, device=device)
    for i in range(batch_size):
        if real_lengths[i] < max_time_frames:
            real_input_padded[i, real_lengths[i]:] = 0

    # Simulate FAKE data batch (e.g., from Generator)
    # Fake data might have different effective lengths if generator also pads/truncates
    fake_lengths = torch.tensor([98, 85, 90, 75], device=device) # Example fake lengths
    fake_input_padded = torch.randn(batch_size, max_time_frames, n_mels, device=device)
    for i in range(batch_size):
         if fake_lengths[i] < max_time_frames: # Just for simulation
             fake_input_padded[i, fake_lengths[i]:] = 0

    # Instantiate the discriminator
    discriminator = DepthwisePatchGANDiscriminator(use_spectral_norm=True).to(device)

    # Get feature maps for both real and fake data
    real_features = discriminator(real_input_padded)
    fake_features = discriminator(fake_input_padded)

    # Calculate masked feature matching loss
    fm_loss = calculate_masked_feature_matching_loss(
        discriminator,
        real_features,
        fake_features,
        real_lengths,
        n_mels,
        device,
        loss_type='l1'
    )

    print(f"Masked Feature Matching Loss (L1): {fm_loss.item()}")

    # --- Optional: Calculate final discriminator loss masked ---
    # Assume simple MSE loss target (e.g., 1.0 for real, 0.0 for fake)
    final_real_output = real_features[-1]
    final_fake_output = fake_features[-1]
    target_real = torch.ones_like(final_real_output)
    target_fake = torch.zeros_like(final_fake_output)

    # Calculate masks for the *final* output layer
    final_layer_index = len(discriminator.conv_params) - 1
    mask_real_final = calculate_feature_map_mask(
        conv_params_list=discriminator.conv_params, 
        layer_index=final_layer_index, 
        input_lengths=real_lengths, 
        max_input_length=real_input_padded.shape[1],
        input_feature_dim=n_mels, 
        device=device)
    mask_fake_final = calculate_feature_map_mask(
        conv_params_list=discriminator.conv_params, 
        layer_index=final_layer_index, 
        input_lengths=fake_lengths, 
        max_input_length=fake_input_padded.shape[1],
        input_feature_dim=n_mels, 
        device=device
    )

    criterion = nn.MSELoss(reduction='none') # Get per-element loss

    # Real loss
    loss_real_raw = criterion(final_real_output, target_real)
    masked_loss_real = loss_real_raw * mask_real_final.float()
    num_valid_real = mask_real_final.sum() + 1e-8
    d_loss_real = masked_loss_real.sum() / num_valid_real

    # Fake loss
    loss_fake_raw = criterion(final_fake_output, target_fake)
    masked_loss_fake = loss_fake_raw * mask_fake_final.float()
    num_valid_fake = mask_fake_final.sum() + 1e-8
    d_loss_fake = masked_loss_fake.sum() / num_valid_fake

    d_loss_total = d_loss_real + d_loss_fake
    print(f"Masked Discriminator Loss (Real): {d_loss_real.item()}")
    print(f"Masked Discriminator Loss (Fake): {d_loss_fake.item()}")
    print(f"Masked Discriminator Loss (Total): {d_loss_total.item()}")