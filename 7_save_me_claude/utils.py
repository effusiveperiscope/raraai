import torch
import torch.nn.functional as F 

def pad_to_multiple(x, multiple):
    """
    Pads the input tensor x along the sequence dimension (dim=1) 
    so that its length is a multiple of 'multiple'.
    
    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, seq_len, feature_dim].
        multiple (int): The factor to pad to.
    
    Returns:
        torch.Tensor: Padded tensor.
        int: Number of padding frames added.
    """
    seq_len = x.shape[1]
    pad_len = (multiple - (seq_len % multiple)) % multiple  # Compute padding needed

    # Apply padding at the end of the sequence (right padding)
    x_padded = F.pad(x, (0, 0, 0, pad_len), mode='constant', value=0)

    return x_padded, pad_len
