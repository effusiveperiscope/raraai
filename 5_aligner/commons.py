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