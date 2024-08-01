from augment.base import DataAugmentation
import torch
import torch.nn.functional as F
import math
import random
import numpy as np
from noise import pnoise1

def default_noise_scale_delegate():
    return random.choice([
        0.0, 0.15, 0.3, 0.6])

class GaussianNoise(DataAugmentation):
    def __init__(self,
        noise_scale_delegate = default_noise_scale_delegate):
        self.noise_scale_delegate = noise_scale_delegate

    def process_features(self, audio, features : torch.Tensor, userdata = {}):
        features = features + torch.randn_like(
            features)*self.noise_scale_delegate()
        return audio, features.to(features.device), userdata

def default_perlin_noise_delegate():
    return 3.0*random.choice([ 1.0, 1.3, 1.6, 2.0 ])

def default_perlin_scale_delegate():
    return random.choice([0.02, 0.03, 0.04, 0.05])

class PerlinNoise(DataAugmentation):
    def __init__(self,
        noise_scale_delegate = default_perlin_noise_delegate,
        scale_delegate=default_perlin_scale_delegate,
        inner_module = None):
        self.noise_scale_delegate = noise_scale_delegate
        self.scale_delegate = scale_delegate
        self.inner_module = inner_module

    def process_features(self, audio, features : torch.Tensor, userdata = {}):
        batch, seq_len, feature_dim = features.shape
        scale = self.scale_delegate()
        noise = np.zeros(seq_len)
        phase = random.random() * 5000
        for i in range(seq_len):
            noise[i] = pnoise1(phase + # random phase offset
                i*scale, octaves=4)*self.noise_scale_delegate()
        noise = torch.tensor(noise, dtype=features.dtype,
            device=features.device).unsqueeze(0).unsqueeze(-1).expand(
                batch, seq_len, feature_dim)

        if self.inner_module is None:
            features = features + noise
        else:
            audio, mix_features, userdata = self.inner_module.process_features(
                audio, features, userdata)
            features = features + noise*(mix_features - features)

        return audio, features, userdata