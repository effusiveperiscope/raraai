from torch import nn
import torch

class FiLMGenerator(nn.Module):
    """ Generates FiLM scale and shift parameters """
    def __init__(self, condition_dim, target_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(condition_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, target_dim * 2),
        )

        nn.init.constant_(self.mlp[-1].bias[0:target_dim], 1.0)  # gamma bias
        nn.init.constant_(self.mlp[-1].bias[target_dim:], 0.0)   # beta bias

    def forward(self, condition):
        """
        Args:
            condition: (batch_size, seq_len, condition_dim)
        Returns:
            gamma: (batch_size, seq_len, target_dim) - Scale
            beta: (batch_size, seq_len, target_dim) - Shift
        """
        params = self.mlp(condition)
        gamma, beta = torch.chunk(params, 2, dim=-1)
        return gamma, beta