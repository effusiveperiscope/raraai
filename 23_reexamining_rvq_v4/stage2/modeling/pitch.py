import torch
import torch.nn as nn
from torchaudio.models import Conformer
from modeling.rvc.commons import FiLMGenerator
import numpy as np

# Flow matching based pitch predictor
class F0FM(nn.Module):
    def __init__(self,
            hidden_features,
            spk_emb_dim,
            codec_dim,
            dropout=0.1):
        super().__init__()

        self.t_emb = nn.Linear(1, hidden_features)
        self.f0_proj = nn.Linear(1, hidden_features)
        self.ppg_proj = nn.Linear(codec_dim, hidden_features)

        self.spk_proj = nn.Linear(spk_emb_dim, hidden_features)
        self.mean_proj = nn.Linear(1, hidden_features)
        self.std_proj = nn.Linear(1, hidden_features)

        self.stats_film = FiLMGenerator(hidden_features, hidden_features)
        self.conformer = Conformer(
            input_dim=hidden_features,
            num_heads=4,
            ffn_dim=128,
            num_layers=4,
            depthwise_conv_kernel_size=15,
            dropout=dropout
        )
        self.out_proj = nn.Linear(hidden_features, 1)

    def forward(self, 
        t, # timestep
        f0_normalized, # f0_normalized = f0 / (f0.max(dim=1))
        ppg_q, # codec codes
        emb_spk, # speaker embedding
        f0_mean, # f0 mean
        f0_std, # f0 std
        lens # sequence lengths
        ):
        x = self.f0_proj(f0_normalized)
        t = self.t_emb(t)
        ppg_q = self.ppg_proj(ppg_q)

        stats_cond = self.spk_proj(emb_spk) + self.mean_proj(f0_mean) + \
            self.std_proj(f0_std)
        stats_cond = stats_cond.unsqueeze(1)
        film_cond = ppg_q + stats_cond
        gamma, beta = self.stats_film(film_cond)

        x = x + x * gamma + beta
        x = self.conformer(x + t, lens)
        f0_pred = self.out_proj(x)

        return f0_pred

    def lerp(x0, target, t):
        xt = ((1 - t) * x0) + (t * target)
        return xt

    def loss(self,
        f0_target,
        f0, ppg_q, emb_spk, lens):

        t = torch.rand(f0_target.shape[0]).to(f0_target.device).unsqueeze(-1)

        v_mask = (f0_target != 0).unsqueeze(-1).float()
        f0_target_scaled = torch.log(f0_target + 1)

        f0_0 = torch.randn_like(f0_target_scaled) * v_mask
        f0_t = self.lerp(
            x0=f0_0, target=f0_target_scaled, t=t)
        v_target = f0_t - f0_0

        f0_mean = f0.mean(dim=1).unsqueeze(-1)
        f0_std = f0.std(dim=1).unsqueeze(-1)
        f0_normalized = f0 / f0.max(dim=1)
        v_pred = self(t, f0_normalized, ppg_q, emb_spk, f0_mean, f0_std, lens)

        return ((v_pred - v_target) ** 2).mean()

    def sample_euler(self, n_steps,
        f0, ppg_q, emb_spk, f0_mean, f0_std, lens):
        
        v_mask = (f0 != 0).unsqueeze(-1).float()
        f0_0 = torch.randn_like(f0) * v_mask
        t_steps = np.linspace(0, 1, n_steps + 1)

        f0_normalized = f0 / f0.max(dim=1)

        f0_t = f0_0
        for i in range(n_steps):
            t = t_steps[i]
            dt = t_steps[i + 1] - t_steps[i]
            t_batch = torch.Tensor([[t for _ in range(n_steps)]]).to(f0.device)
            f0_t = f0_t + self(t_batch, f0_normalized, ppg_q, emb_spk, f0_mean, f0_std, lens) * dt

        return (torch.exp(f0_t) - 1) * v_mask

if __name__ == '__main__':
    f0_target = torch.randn([4, 100, 1])
    f0 = torch.randn([4, 100, 1])

    ppg_q = torch.randn([4, 100, 768])
    emb_spk = torch.randn([4, 1, 256])

    lens = torch.Tensor([100, 100, 100, 100])

    fm = F0FM(hidden_features=128, spk_emb_dim=256, codec_dim=768)

    loss = fm.loss(f0_target, f0, ppg_q, emb_spk, lens)
    print(loss.shape)

    fm.eval()
    with torch.no_grad():
        f0_pred = fm.sample_euler(n_steps=10,
            f0=f0, ppg_q=ppg_q, emb_spk=emb_spk, 
            f0_mean=f0.mean(dim=1).unsqueeze(-1),
            f0_std=f0.mean(dim=1).unsqueeze(-1),
            lens=lens)
        print(f0_pred.shape)

    pass