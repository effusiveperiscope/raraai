import torch
import torch.nn as nn

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
        self.t_proj = nn.Linear(1, hidden_dim)
        self.net = nn.Sequential(
            DepthwiseSeparableConv1d(hidden_dim, hidden_dim, 5, padding=2),
            nn.SiLU(),
            DepthwiseSeparableConv1d(hidden_dim, hidden_dim, 5, padding=2),
            nn.SiLU(),
            DepthwiseSeparableConv1d(hidden_dim, hidden_dim, 5, padding=2),
            nn.SiLU(),
            DepthwiseSeparableConv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
            DepthwiseSeparableConv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
        )
        self.final_proj = nn.Linear(hidden_dim, 1)

    def forward(
        self, t: torch.Tensor, x_t: torch.Tensor, x_mask: torch.Tensor
    ):
        x = self.in_proj(x_t) + self.t_proj(t)
        x = self.net(x)
        x = self.final_proj(x) * x_mask
        return x

    def interpolate_linear(x_0, x_1, t):
        """Evaluates the linear interpolation path between x_0 and x_1 at step t."""
        x_t = ((1 - t) * x_0) + (t * x_1)
        return x_t

    def get_target_velocity(x_0, x_1):
        """
        Get the velocity for a given pair of noise and target points.
        This is the per-pair (conditional) velocity along the straight path.
        """
        return x_1 - x_0

    def compute_loss(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
        x_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the loss for a single batch of (X_0, X_1) couplings and flow steps T.
        """
        # Interpolate the data at the sampled time step
        x_t = PitchPredictorV0.interpolate_linear(x_0=x_0, x_1=x_1, t=t)
        # Get the target velocity
        v_target = PitchPredictorV0.get_target_velocity(x_0=x_0, x_1=x_1)
        # Predict the velocity
        v_pred = self.forward(t=t, x_t=x_t, x_mask=x_mask)
        # Compute the loss
        loss = ((v_pred - v_target) ** 2).mean()
        return loss
