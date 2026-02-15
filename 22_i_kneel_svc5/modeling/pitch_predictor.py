import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

def get_masked_mean(x, mask, dim=1, keepdim=False, eps=1e-7):
    """Calculates the mean of masked elements along a dimension."""
    sum_val = x.sum(dim=dim, keepdim=keepdim)
    count = mask.sum(dim=dim, keepdim=keepdim)
    return sum_val / count.clamp(min=eps)

def get_masked_std(x, mask, dim=1, keepdim=False, eps=1e-7):
    """Calculates the standard deviation of masked elements along a dimension."""
    count = mask.sum(dim=dim, keepdim=keepdim)
    
    # Calculate mean using the function above
    # We use keepdim=True here to ensure broadcasting works for the subtraction
    mu = get_masked_mean(x, mask, dim=dim, keepdim=True, eps=eps)
    
    # Calculate variance
    squared_diff = (x - mu)**2
    # Only sum the squared differences for masked entries
    var = (squared_diff * mask).sum(dim=dim, keepdim=keepdim) / count.clamp(min=eps)
    
    std = torch.sqrt(var)
    
    if not keepdim:
        std = std.squeeze(dim)
        
    return std + eps


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
        spectral_norm=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if spectral_norm:
            self.depthwise = nn.utils.spectral_norm(nn.Conv1d(in_channels, in_channels, kernel_size,
                stride=stride, padding=padding, groups=in_channels, padding_mode='reflect'))
            self.pointwise = nn.utils.spectral_norm(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        else:
            self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size,
                stride=stride, padding=padding, groups=in_channels, padding_mode='reflect')
            self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class PitchPredictorV0(nn.Module):
    def __init__(self, hidden_dim: int = 192):
        super().__init__()
        self.in_proj = nn.Linear(1, hidden_dim)
        # Use a small MLP for time/stat conditioning to give the model more capacity
        self.cond_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim), # t, mean, std
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.net = nn.ModuleList([
            DepthwiseSeparableConv1d(hidden_dim, hidden_dim, 5, padding=2),
            DepthwiseSeparableConv1d(hidden_dim, hidden_dim, 7, padding=3),
            DepthwiseSeparableConv1d(hidden_dim, hidden_dim, 5, padding=2),
            DepthwiseSeparableConv1d(hidden_dim, hidden_dim, 3, padding=1),
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(4)])
        self.final_proj = nn.Linear(hidden_dim, 1)

    def forward(self, t, x_t, x_mask, x_u, x_std):
        # x_t: [B, T, 1], t: [B, 1, 1], stats: [B, 1, 2]
        x = self.in_proj(x_t)
        
        # Combine conditioning: time + target statistics
        cond = torch.cat([t, x_u, x_std], dim=-1) # [B, 1, 3]
        cond_emb = self.cond_mlp(cond) # [B, 1, hidden_dim]
        
        x = (x + cond_emb) * x_mask

        # Convolutional Backbone
        for conv, norm in zip(self.net, self.norms):
            res = x
            x = rearrange(x, 'b t c -> b c t')
            x = conv(x)
            x = rearrange(x, 'b c t -> b t c')
            x = norm(x)
            x = torch.nn.functional.silu(x) + res # Add residual connection
            x = x * x_mask

        return self.final_proj(x) * x_mask

    @staticmethod
    def compute_loss(model, f0):
        # 1. Handle Unvoiced (Avoid log(0))
        # Replace 0 with a very small value or the floor of the dataset pitch
        f0_mask = f0 != 0
        f0_safe = f0.clamp(min=1e-5) 
        log_f0 = torch.log(f0_safe)
        
        # 2. Extract Stats (Target Style)
        x_u = get_masked_mean(log_f0, f0_mask, keepdim=True)
        x_std = get_masked_std(log_f0, f0_mask, keepdim=True)
        
        # 3. Standardize Target
        x_1 = (log_f0 - x_u) / (x_std + 1e-5)
        x_0 = torch.randn_like(x_1)
        
        # 4. Sample Flow Time
        t = torch.rand(f0.shape[0], 1, device=f0.device)
        
        # 5. Flow Matching Path
        x_t = (1 - t) * x_0 + t * x_1
        v_target = x_1 - x_0
        
        v_pred = model(t.unsqueeze(-1), x_t.unsqueeze(-1), f0_mask.unsqueeze(-1), 
            x_u.unsqueeze(-1), x_std.unsqueeze(-1))
        
        # MSE only on voiced regions
        loss = torch.sum(((v_pred.squeeze(-1) - v_target) * f0_mask)**2) / f0_mask.sum()
        return loss

    def infer( # Euler integration
        self,
        f0: torch.Tensor, # [1, T]
        step_start: int,
        step_end: int,
        u_scale: float,
        std_scale: float   
    ):
        self.eval()
        assert step_end > step_start
        t_steps = np.linspace(0, 1, step_end + 1)

        # Log
        f0_mask = (f0 != 0)
        f0_safe = f0.clamp(min=1e-5) 
        log_f0 = torch.log(f0_safe)

        # Normalize
        x_u = get_masked_mean(log_f0, f0_mask, keepdim=True)
        x_std = get_masked_std(log_f0, f0_mask, keepdim=True)
        log_f0 = (log_f0 - x_u) / x_std

        x_mask = (f0 != 0)
        x_t = PitchPredictorV0.interpolate_linear(
            torch.randn_like(log_f0), log_f0, step_start / step_end).unsqueeze(-1)

        x1_u = x_u * u_scale
        x1_std = x_std * std_scale

        for i in range(step_start, step_end):
            t = t_steps[i]
            dt = t_steps[i + 1] - t_steps[i]
            t_batch = torch.Tensor([[t]]).to(f0.device)
            x_t = x_t + self.forward(
                t=t_batch.unsqueeze(-1), x_t = x_t, 
                x_mask = x_mask.unsqueeze(-1), 
                x_u = x1_u.unsqueeze(-1), 
                x_std = x1_std.unsqueeze(-1)) * dt

        log_f0_t = (x_t * x1_std.unsqueeze(-1)) + x1_u.unsqueeze(-1)

        return torch.exp(log_f0_t) * f0_mask

if __name__ == '__main__':
    T = 40
    t = torch.Tensor([1, 2, 3]).unsqueeze(-1) # [B, 1]
    f0 = torch.randn([t.shape[0], T]) # [B, T]

    net = PitchPredictorV0()
    print(PitchPredictorV0.compute_loss(net, f0))