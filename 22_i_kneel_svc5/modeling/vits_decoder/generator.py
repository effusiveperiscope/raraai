import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import Conv1d
from torch.nn import ConvTranspose1d
from torch.nn.utils import weight_norm
from torch.nn.utils import remove_weight_norm

from .nsf import SourceModuleHnNSF
from .bigv import init_weights, AMPBlock, SnakeAlias
from einops import rearrange


class SpeakerAdapter(nn.Module):

    def __init__(self,
                 speaker_dim,
                 adapter_dim,
                 epsilon=1e-5
                 ):
        super(SpeakerAdapter, self).__init__()
        self.speaker_dim = speaker_dim
        self.adapter_dim = adapter_dim
        self.epsilon = epsilon
        self.W_scale = nn.Linear(self.speaker_dim, self.adapter_dim)
        self.W_bias = nn.Linear(self.speaker_dim, self.adapter_dim)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.W_scale.weight, 0.0)
        torch.nn.init.constant_(self.W_scale.bias, 1.0)
        torch.nn.init.constant_(self.W_bias.weight, 0.0)
        torch.nn.init.constant_(self.W_bias.bias, 0.0)

    def forward(self, x, speaker_embedding):
        x = x.transpose(1, -1)
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        scale = self.W_scale(speaker_embedding)
        bias = self.W_bias(speaker_embedding)
        y *= scale.unsqueeze(1)
        y += bias.unsqueeze(1)
        y = y.transpose(1, -1)
        return y

class PitchConditioner(nn.Module):
    """Conditioning module that treats 0 as a special embedding, intended for
    use with f0 contours (i.e. where 0 = unvoiced)"""
    def __init__(self, inter_channels):
        super().__init__()
        self.pitch_uv_emb = nn.Parameter(torch.randn(inter_channels))
        self.pitch_proj = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, inter_channels),
            nn.SiLU(),
            nn.Linear(inter_channels, inter_channels),
        )

    def forward(self, pitch, convert_mel=True, use_dtype=torch.float32):
        if pitch.dtype == torch.long: # Handle quantized condition
            pitch = pitch.to(use_dtype)

        if convert_mel:
            mel_pitch = 1127 * torch.log1p(pitch / 700)
            mel_pitch = mel_pitch.unsqueeze(-1)
        else:
            mel_pitch = pitch.unsqueeze(-1)

        voiced_mask = (pitch > 0).to(pitch.dtype).unsqueeze(-1)

        pitch_feat = self.pitch_proj(mel_pitch) * voiced_mask
        pitch_feat += (1 - voiced_mask) * self.pitch_uv_emb

        pitch_feat = F.layer_norm(pitch_feat, pitch_feat.shape[-1:])

        return pitch_feat

class PitchConditioner2(nn.Module):
    """Same as PitchConditioner but handles special RMVPE features 
    (e.g. confidence). Operates both in conditioning and bypass mode."""
    def __init__(self, inter_channels):
        super().__init__()
        self.cond = PitchConditioner(inter_channels)
        self.confidence_proj = nn.Linear(1, inter_channels)
        self.subharmonic_proj = nn.Linear(1, inter_channels)
        self.inharmonic_proj = nn.Linear(1, inter_channels)
        # bypass embedding
        self.bypass_emb = nn.Parameter(torch.randn(inter_channels * 3))
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(inter_channels * 4, inter_channels, 5, padding=2),
            nn.SiLU(),
            nn.Conv1d(inter_channels, inter_channels, 3, padding=1),
        )
        self.final_proj = nn.Linear(inter_channels, inter_channels*2)

    def forward(self, pitch, confidence=None, 
        subharmonic=None, inharmonic=None, 
        convert_mel=True, use_dtype=torch.float32):
        cond = self.cond(pitch, convert_mel, use_dtype)
        #print('dtype of input:', pitch.dtype)
        if confidence is not None and subharmonic is not None and inharmonic is not None:
            #print('using conditioning mode')
            cond = torch.cat((
                cond,
                self.confidence_proj(confidence.unsqueeze(-1)),
                self.subharmonic_proj(subharmonic.unsqueeze(-1)),
                self.inharmonic_proj(inharmonic.unsqueeze(-1))), dim=2)
        else:
            #print('using bypass mode')
            cond = torch.cat(
                (cond, self.bypass_emb.unsqueeze(0).unsqueeze(0).repeat(cond.shape[0], cond.shape[1], 1)), dim=2)
        cond = rearrange(cond, "b t c -> b c t")
        cond = self.net(cond)
        cond = rearrange(cond, "b c t -> b t c")
        gamma, beta = self.final_proj(cond).chunk(2, dim=-1)
        return gamma, beta

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

class EnvGenerator(torch.nn.Module):
    def __init__(self,
        x_channels,
        hidden_channels,
        n_outputs):
        super(EnvGenerator, self).__init__()

        self.in_proj = nn.Linear(x_channels, hidden_channels)
        self.f0_proj = nn.Linear(1, hidden_channels)
        self.cnn = nn.Sequential(
            nn.SiLU(),
            DepthwiseSeparableConv1d(hidden_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
            DepthwiseSeparableConv1d(hidden_channels, hidden_channels, 5, padding=2),
            nn.SiLU(),
            DepthwiseSeparableConv1d(hidden_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
            DepthwiseSeparableConv1d(hidden_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
        )
        self.final_proj = nn.Linear(hidden_channels, n_outputs)

    def forward(self, x, f0):
        x = self.in_proj(x) + self.f0_proj(f0)

        x = rearrange(x, "b t c -> b c t")
        x = self.cnn(x)
        x = rearrange(x, "b c t -> b t c")

        x = self.final_proj(x)
        x = F.layer_norm(x, x.shape[-1:])
        x = (F.tanh(x) + 1) / 2 # [0, 1]
        return x


class Generator(torch.nn.Module):
    # this is our main BigVGAN model. Applies anti-aliased periodic activation for resblocks.
    def __init__(self, hp):
        super(Generator, self).__init__()
        self.hp = hp
        self.num_kernels = len(hp.gen.resblock_kernel_sizes)
        self.num_upsamples = len(hp.gen.upsample_rates)
        # speaker adaper, 256 should change by what speaker encoder you use
        self.adapter = SpeakerAdapter(hp.vits.spk_dim, hp.gen.upsample_input)
        # pre conv
        self.conv_pre = Conv1d(hp.gen.upsample_input,
                               hp.gen.upsample_initial_channel, 7, 1, padding=3)
        # nsf
        self.m_source = SourceModuleHnNSF(sampling_rate=hp.data.sampling_rate)
        self.noise_convs = nn.ModuleList()
        self.har_convs = nn.ModuleList()
        # transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(hp.gen.upsample_rates, hp.gen.upsample_kernel_sizes)):
            c_cur = hp.gen.upsample_initial_channel // (2 ** (i + 1))
            # print(f'ups: {i} {k}, {u}, {(k - u) // 2}')
            # base
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        hp.gen.upsample_initial_channel // (2 ** i),
                        hp.gen.upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2)
                )
            )
            # nsf
            if i + 1 < len(hp.gen.upsample_rates):
                stride_f0 = np.prod(hp.gen.upsample_rates[i + 1:])
                stride_f0 = int(stride_f0)
                self.noise_convs.append(
                    Conv1d(
                        1,
                        c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
                self.har_convs.append(
                    Conv1d(
                        1,
                        c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
            else:
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))
                self.har_convs.append(Conv1d(1, c_cur, kernel_size=1))

        self.pit = PitchConditioner2(hp.gen.upsample_input)
        self.env_gen = EnvGenerator(hp.gen.upsample_input, 192, len(self.noise_convs) * 2)

        # residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = hp.gen.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(hp.gen.resblock_kernel_sizes, hp.gen.resblock_dilation_sizes):
                self.resblocks.append(AMPBlock(ch, k, d))

        # post conv
        self.activation_post = SnakeAlias(ch)
        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        # weight initialization
        self.ups.apply(init_weights)

        self.upp = math.prod(hp.gen.upsample_rates)
    
    def freeze_layers(self, n):
        if n is None:
            n = len(self.ups)
        for i in range(n):
            for param in self.ups[i].parameters():
                param.requires_grad = False
            for param in self.noise_convs[i].parameters():
                param.requires_grad = False
            for param in self.har_convs[i].parameters():
                param.requires_grad = False

    def forward(self, spk, x, f0, pitch_extras=None):
        # Perturbation
        x = x + torch.randn_like(x)

        # nsf
        har_source = self.m_source(f0, upp=self.upp)
        har_source = har_source.transpose(1, 2)

        uv = f0 <= 0
        uv = uv.float()
        uv = F.interpolate(uv.unsqueeze(1), har_source.shape[-1], mode='nearest')

        if pitch_extras is not None:
            gamma, beta = self.pit(f0, use_dtype=x.dtype, **pitch_extras)
        else:
            gamma, beta = self.pit(f0, use_dtype=x.dtype)
        
        x = rearrange(x, "b c t -> b t c")
        x = x * gamma + beta
        x = rearrange(x, "b t c -> b c t")

        env = self.env_gen(rearrange(x, "b c t -> b t c"), f0.unsqueeze(-1))

        # adapter
        x = self.adapter(x, spk)
        x = self.conv_pre(x)
        x = x * torch.tanh(F.softplus(x))

        for i in range(self.num_upsamples):
            # upsampling
            x = self.ups[i](x)

            # Resample env at every upsample
            noi_source = torch.randn_like(har_source)

            noise_env = env[:, :, 2*i] # [B, T]
            har_env = env[:, :, 2*i+1] # [B, T]
            noise_env = rearrange(noise_env, "b t -> b 1 t")
            har_env = rearrange(har_env, "b t -> b 1 t")
            noise_env = F.interpolate(noise_env, har_source.shape[-1], mode='nearest')
            har_env = F.interpolate(har_env, har_source.shape[-1], mode='nearest')

            this_noise = (uv * noi_source + (1 - uv) * har_source * self.m_source.l_sin_gen.sine_amp / 3) * noise_env
            this_noise = this_noise.to(x.dtype)
            this_har = har_source * har_env
            this_har = this_har.to(x.dtype)

            x = x + self.noise_convs[i](this_noise) + self.har_convs[i](this_har)

            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # post conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()

    def eval(self, inference=False):
        super(Generator, self).eval()
        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()

    def pitch2source(self, f0):
        f0 = f0[:, None]
        f0 = self.f0_upsamp(f0).transpose(1, 2)  # [1,len,1]
        har_source = self.m_source(f0)
        har_source = har_source.transpose(1, 2)  # [1,1,len]
        return har_source

    def source2wav(self, audio):
        MAX_WAV_VALUE = 32768.0
        audio = audio.squeeze()
        audio = MAX_WAV_VALUE * audio
        audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE-1)
        audio = audio.short()
        return audio.cpu().detach().numpy()

    def inference(self, spk, x, har_source):
        # adapter
        x = self.adapter(x, spk)
        x = self.conv_pre(x)
        x = x * torch.tanh(F.softplus(x))

        for i in range(self.num_upsamples):
            # upsampling
            x = self.ups[i](x)
            # nsf
            x_source = self.noise_convs[i](har_source)
            x = x + x_source
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # post conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x
