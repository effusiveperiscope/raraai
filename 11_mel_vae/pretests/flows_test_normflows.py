import normflows as nf
import torch

latent_dim = 128
base = nf.distributions.base.DiagGaussian(latent_dim)
flows = [
    nf.flows.GlowBlock(
        channels=latent_dim,
        hidden_channels=latent_dim*2
    )
]
nflow = nf.NormalizingFlow(base, flows)

seq_len = 10
x = torch.randn(2, latent_dim, seq_len)
z, log_det = nflow.forward_and_log_det(x)
log_prob = nflow.log_prob(x)
print(z.shape)
print(log_det)
print(log_prob)