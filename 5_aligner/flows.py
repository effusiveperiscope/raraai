# Flow-based decoder from GlowTTS
import copy
import math
import numpy as np
import scipy
import torch
from torch import nn
from torch.nn import functional as F
import commons

class ActNorm(nn.Module):
  def __init__(self, channels, ddi=False, **kwargs):
    super().__init__()
    self.channels = channels
    self.initialized = not ddi

    self.logs = nn.Parameter(torch.zeros(1, channels, 1))
    self.bias = nn.Parameter(torch.zeros(1, channels, 1))

  def forward(self, x, x_mask=None, reverse=False, **kwargs):
    if x_mask is None:
      x_mask = torch.ones(x.size(0), 1, x.size(2)).to(
        device=x.device, dtype=x.dtype)
    x_len = torch.sum(x_mask, [1, 2])
    if not self.initialized:
      self.initialize(x, x_mask)
      self.initialized = True

    if reverse:
      z = (x - self.bias) * torch.exp(-self.logs) * x_mask
      logdet = None
    else:
      z = (self.bias + torch.exp(self.logs) * x) * x_mask
      logdet = torch.sum(self.logs) * x_len # [b]

    return z, logdet

  def store_inverse(self):
    pass

  def set_ddi(self, ddi):
    self.initialized = not ddi

  def initialize(self, x, x_mask):
    with torch.no_grad():
      denom = torch.sum(x_mask, [0, 2])
      m = torch.sum(x * x_mask, [0, 2]) / denom
      m_sq = torch.sum(x * x * x_mask, [0, 2]) / denom
      v = m_sq - (m ** 2)
      logs = 0.5 * torch.log(torch.clamp_min(v, 1e-6))

      bias_init = (-m * torch.exp(-logs)).view(*self.bias.shape).to(
        dtype=self.bias.dtype)
      logs_init = (-logs).view(*self.logs.shape).to(
        dtype=self.logs.dtype)

      self.bias.data.copy_(bias_init)
      self.logs.data.copy_(logs_init)

class InvConvNear(nn.Module):
  def __init__(self, channels, n_split=4, no_jacobian=False, **kwargs):
    super().__init__()
    assert(n_split % 2 == 0)
    self.channels = channels
    self.n_split = n_split
    self.no_jacobian = no_jacobian
    
    w_init = torch.qr(
        torch.FloatTensor(self.n_split, self.n_split).normal_())[0]
    if torch.det(w_init) < 0:
      w_init[:,0] = -1 * w_init[:,0]
    self.weight = nn.Parameter(w_init)

  def forward(self, x, x_mask=None, reverse=False, **kwargs):
    b, c, t = x.size()
    assert(c % self.n_split == 0)
    if x_mask is None:
      x_mask = 1
      x_len = torch.ones((b,), dtype=x.dtype, device=x.device) * t
    else:
      x_len = torch.sum(x_mask, [1, 2])

    x = x.view(
        b, 2, c // self.n_split, self.n_split // 2, t)
    x = x.permute(
        0, 1, 3, 2, 4).contiguous().view(b, self.n_split, c // self.n_split, t)

    if reverse:
      if hasattr(self, "weight_inv"):
        weight = self.weight_inv
      else:
        weight = torch.inverse(
            self.weight.float()).to(dtype=self.weight.dtype)
      logdet = None
    else:
      weight = self.weight
      if self.no_jacobian:
        logdet = 0
      else:
        logdet = torch.logdet(
            self.weight) * (c / self.n_split) * x_len # [b]

    weight = weight.view(self.n_split, self.n_split, 1, 1)
    z = F.conv2d(x, weight)

    z = z.view(b, 2, self.n_split // 2, c // self.n_split, t)
    z = z.permute(0, 1, 3, 2, 4).contiguous().view(b, c, t) * x_mask
    return z, logdet

  def store_inverse(self):
    self.weight_inv = torch.inverse(self.weight.float()).to(dtype=self.weight.dtype)

class WN(torch.nn.Module):
  def __init__(self, in_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
      super(WN, self).__init__()
      assert(kernel_size % 2 == 1)
      assert(hidden_channels % 2 == 0)
      self.in_channels = in_channels
      self.hidden_channels =hidden_channels
      self.kernel_size = kernel_size,
      self.dilation_rate = dilation_rate
      self.n_layers = n_layers
      self.gin_channels = gin_channels
      self.p_dropout = p_dropout

      self.in_layers = torch.nn.ModuleList()
      self.res_skip_layers = torch.nn.ModuleList()
      self.drop = nn.Dropout(p_dropout)

      if gin_channels != 0:
        cond_layer = torch.nn.Conv1d(gin_channels, 2*hidden_channels*n_layers, 1)
        self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

      for i in range(n_layers):
        dilation = dilation_rate ** i
        padding = int((kernel_size * dilation - dilation) / 2)
        in_layer = torch.nn.Conv1d(hidden_channels, 2*hidden_channels, kernel_size,
                                   dilation=dilation, padding=padding)
        in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
        self.in_layers.append(in_layer)

        # last one is not necessary
        if i < n_layers - 1:
            res_skip_channels = 2 * hidden_channels
        else:
            res_skip_channels = hidden_channels

        res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
        res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
        self.res_skip_layers.append(res_skip_layer)

  def forward(self, x, x_mask=None, g=None, **kwargs):
      output = torch.zeros_like(x)
      n_channels_tensor = torch.IntTensor([self.hidden_channels])

      if g is not None:
        g = self.cond_layer(g)

      for i in range(self.n_layers):
          x_in = self.in_layers[i](x)
          x_in = self.drop(x_in)
          if g is not None:
            cond_offset = i * 2 * self.hidden_channels
            g_l = g[:,cond_offset:cond_offset+2*self.hidden_channels,:]
          else:
            g_l = torch.zeros_like(x_in)

          acts = commons.fused_add_tanh_sigmoid_multiply(
              x_in,
              g_l,
              n_channels_tensor)

          res_skip_acts = self.res_skip_layers[i](acts)
          if i < self.n_layers - 1:
            x = (x + res_skip_acts[:,:self.hidden_channels,:]) * x_mask
            output = output + res_skip_acts[:,self.hidden_channels:,:]
          else:
            output = output + res_skip_acts
      return output * x_mask

  def remove_weight_norm(self):
    if self.gin_channels != 0:
      torch.nn.utils.remove_weight_norm(self.cond_layer)
    for l in self.in_layers:
      torch.nn.utils.remove_weight_norm(l)
    for l in self.res_skip_layers:
     torch.nn.utils.remove_weight_norm(l)

class CouplingBlock(nn.Module):
  def __init__(self, in_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0, sigmoid_scale=False):
    super().__init__()
    self.in_channels = in_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels
    self.p_dropout = p_dropout
    self.sigmoid_scale = sigmoid_scale

    start = torch.nn.Conv1d(in_channels//2, hidden_channels, 1)
    start = torch.nn.utils.weight_norm(start)
    self.start = start
    # Initializing last layer to 0 makes the affine coupling layers
    # do nothing at first.  It helps to stabilze training.
    end = torch.nn.Conv1d(hidden_channels, in_channels, 1)
    end.weight.data.zero_()
    end.bias.data.zero_()
    self.end = end

    self.wn = WN(in_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels, p_dropout)


  def forward(self, x, x_mask=None, reverse=False, g=None, **kwargs):
    b, c, t = x.size()
    if x_mask is None:
      x_mask = 1
    x_0, x_1 = x[:,:self.in_channels//2], x[:,self.in_channels//2:]

    x = self.start(x_0) * x_mask
    x = self.wn(x, x_mask, g)
    out = self.end(x)

    z_0 = x_0
    m = out[:, :self.in_channels//2, :]
    logs = out[:, self.in_channels//2:, :]
    if self.sigmoid_scale:
      logs = torch.log(1e-6 + torch.sigmoid(logs + 2))

    if reverse:
      z_1 = (x_1 - m) * torch.exp(-logs) * x_mask
      logdet = None
    else:
      z_1 = (m + torch.exp(logs) * x_1) * x_mask
      logdet = torch.sum(logs * x_mask, [1, 2])

    z = torch.cat([z_0, z_1], 1)
    return z, logdet

  def store_inverse(self):
    self.wn.remove_weight_norm()

class FlowDecoder(nn.Module):
  def __init__(self, 
      in_channels, 
      hidden_channels = 192, 
      kernel_size = 3, 
      dilation_rate = 1, 
      n_blocks=12, 
      n_layers=4, 
      p_dropout=0., 
      n_split=4,
      n_sqz=2,
      sigmoid_scale=False,
      gin_channels=0):
    super().__init__()

    self.in_channels = in_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_blocks = n_blocks
    self.n_layers = n_layers
    self.p_dropout = p_dropout
    self.n_split = n_split
    self.n_sqz = n_sqz
    self.sigmoid_scale = sigmoid_scale
    self.gin_channels = gin_channels

    self.flows = nn.ModuleList()
    for b in range(n_blocks):
      self.flows.append(
        ActNorm(channels=in_channels * n_sqz))
      self.flows.append(
        InvConvNear(channels=in_channels * n_sqz, n_split=n_split))
      self.flows.append(
        CouplingBlock(
          in_channels * n_sqz,
          hidden_channels,
          kernel_size=kernel_size, 
          dilation_rate=dilation_rate,
          n_layers=n_layers,
          gin_channels=gin_channels,
          p_dropout=p_dropout,
          sigmoid_scale=sigmoid_scale))

  def forward(self, x, x_mask, g=None, reverse=False):
    if not reverse:
      flows = self.flows
      logdet_tot = 0
    else:
      flows = reversed(self.flows)
      logdet_tot = None

    if self.n_sqz > 1:
      x, x_mask = commons.squeeze(x, x_mask, self.n_sqz)
    for f in flows:
      if not reverse:
        x, logdet = f(x, x_mask, g=g, reverse=reverse)
        logdet_tot += logdet
      else:
        x, logdet = f(x, x_mask, g=g, reverse=reverse)
    if self.n_sqz > 1:
      x, x_mask = commons.unsqueeze(x, x_mask, self.n_sqz)
    return x, logdet_tot

  def store_inverse(self):
    for f in self.flows:
      f.store_inverse()
