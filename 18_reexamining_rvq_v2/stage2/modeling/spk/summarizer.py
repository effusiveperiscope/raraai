import torch
import torch.nn as nn
from mambapy.mamba import Mamba, MambaConfig
from ..rvc.f0_predictor import ResBlock
from einops import rearrange

class SpeakerSummarizer(nn.Module):
    def __init__(self,
        spec_dim: int,
        hidden_dim: int,
        summary_dim: int):
        super().__init__()

        mamba_config = MambaConfig(d_model = hidden_dim, n_layers=2)
        self.in_proj = nn.Linear(spec_dim, hidden_dim)
        self.seq1 = ResBlock([
            nn.Conv1d(hidden_dim, hidden_dim, 7, padding=3),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            ])
        self.mamba = Mamba(mamba_config)
        self.seq2 = ResBlock([
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            ])
        self.final_proj = nn.Linear(hidden_dim, summary_dim * 2)

    def forward(self, x, x_mask):
        x = rearrange(x, "b c t -> b t c")
        x_mask = rearrange(x_mask, "b c t -> b t c")
        x = self.in_proj(x) * x_mask
        x = self.seq1(x) * x_mask
        x = self.mamba(x) * x_mask
        x = self.seq2(x) * x_mask
        x = self.final_proj(x)
        stats = torch.sum(x * x_mask, dim=1) / torch.sum(x_mask, dim=1).clamp(min=1)
        x_m, x_logs = torch.split(stats, stats.shape[1] // 2, dim=1)
        z = (x_m + torch.randn_like(x_m) * torch.exp(x_logs))
        return z, x_m, x_logs

if __name__ == '__main__':
    test_spec = torch.randn([2, 769, 680])
    test_mask = torch.ones([2, 680, 1])
    summarizer = SpeakerSummarizer(spec_dim=769, hidden_dim=256, summary_dim=128)
    summary = summarizer(test_spec, test_mask)
    print(summary.shape)
