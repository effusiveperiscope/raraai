import accelerate
import torch

def vevo_load_checkpoint(build_model_func, cfg, ckpt_path, device):
    model = build_model_func(cfg, device)
    accelerate.load_checkpoint_and_dispatch(model, ckpt_path)
    return model

def create_sequence_mask(lengths, max_len=None):
    """
    Creates a binary mask from sequence lengths.
    
    Args:
        lengths (torch.Tensor): Tensor of shape (B, 1) or (B,) containing the lengths of sequences in the batch
        max_len (int, optional): Maximum sequence length. If None, uses the maximum value in lengths.
    
    Returns:
        torch.Tensor: Binary mask of shape (B, N) where N is max_len, with 1s for valid positions and 0s for padded positions
    """
    # Ensure lengths is a 1D tensor
    if lengths.dim() > 1:
        lengths = lengths.squeeze(-1)
    
    batch_size = lengths.size(0)
    
    # Use max length from the batch if not provided
    if max_len is None:
        max_len = lengths.max().item()
    
    # Create a tensor of indices representing positions: (B, N)
    positions = torch.arange(0, max_len, device=lengths.device).expand(batch_size, max_len)
    
    # Compare positions with lengths to create mask: (B, N)
    # 1 where position < length, 0 elsewhere
    mask = positions < lengths.unsqueeze(1)
    
    return mask

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total_params < 1e6:
        return f"{total_params} params"  # Parameters
    elif total_params < 1e9:
        return f"{total_params / 1e6:.2f} M"  # Millions
    else:
        return f"{total_params / 1e9:.2f} B"  # Billions
