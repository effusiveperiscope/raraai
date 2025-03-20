import torch

def sequence_mask(length, max_length=None):
  if max_length is None:
    max_length = length.max()
  x = torch.arange(max_length, dtype=length.dtype, device=length.device)
  return x.unsqueeze(0) < length.unsqueeze(1)

def generate_path(duration, mask):
  """
  duration: [b, t_x]
  mask: [b, t_x, t_y]
  """
  device = duration.device
  
  b, t_x, t_y = mask.shape
  cum_duration = torch.cumsum(duration, 1)
  path = torch.zeros(b, t_x, t_y, dtype=mask.dtype).to(device=device)
  
  cum_duration_flat = cum_duration.view(b * t_x)
  path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
  path = path.view(b, t_x, t_y)
  path = path * ~F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:,:-1]
  path = path * mask
  return path

def convert_pad_shape(pad_shape):
  l = pad_shape[::-1]
  pad_shape = [item for sublist in l for item in sublist]
  return pad_shape

def squeeze(x, x_mask=None, n_sqz=2):
  b, c, t = x.size()

  t = (t // n_sqz) * n_sqz
  x = x[:,:,:t]
  x_sqz = x.view(b, c, t//n_sqz, n_sqz)
  x_sqz = x_sqz.permute(0, 3, 1, 2).contiguous().view(b, c*n_sqz, t//n_sqz)
  
  if x_mask is not None:
    x_mask = x_mask[:,:,n_sqz-1::n_sqz]
  else:
    x_mask = torch.ones(b, 1, t//n_sqz).to(device=x.device, dtype=x.dtype)
  return x_sqz * x_mask, x_mask


def unsqueeze(x, x_mask=None, n_sqz=2):
  b, c, t = x.size()

  x_unsqz = x.view(b, n_sqz, c//n_sqz, t)
  x_unsqz = x_unsqz.permute(0, 2, 3, 1).contiguous().view(b, c//n_sqz, t*n_sqz)

  if x_mask is not None:
    x_mask = x_mask.unsqueeze(-1).repeat(1,1,1,n_sqz).view(b, 1, t*n_sqz)
  else:
    x_mask = torch.ones(b, 1, t*n_sqz).to(device=x.device, dtype=x.dtype)
  return x_unsqz * x_mask, x_mask

@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
  n_channels_int = n_channels[0]
  in_act = input_a + input_b
  t_act = torch.tanh(in_act[:, :n_channels_int, :])
  s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
  acts = t_act * s_act
  return acts

def truncate_to_common_length(*tensors, dim_match=1):
    if not tensors:
        raise ValueError("At least one tensor must be provided")
    
    # Get the minimum sequence length from all tensors (third dimension)
    min_seq_len = min(tensor.size(dim_match) for tensor in tensors)
    
    # Truncate all tensors to the minimum sequence length
    truncated_tensors = [tensor[:, :, :min_seq_len] for tensor in tensors]
    
    return truncated_tensors

def clip_grad_value_(parameters, clip_value, norm_type=2):
  if isinstance(parameters, torch.Tensor):
    parameters = [parameters]
  parameters = list(filter(lambda p: p.grad is not None, parameters))
  norm_type = float(norm_type)
  clip_value = float(clip_value)

  total_norm = 0
  for p in parameters:
    param_norm = p.grad.data.norm(norm_type)
    total_norm += param_norm.item() ** norm_type

    p.grad.data.clamp_(min=-clip_value, max=clip_value)
  total_norm = total_norm ** (1. / norm_type)
  return total_norm