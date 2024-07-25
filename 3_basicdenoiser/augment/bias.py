import random
import numpy as np
from augment.base import DataAugmentation
import torch

def default_bias_scale_delegate():
    return 0.25 - random.random()*0.5

# This does not work - typically features are centered around 0
class BiasPerturbation(DataAugmentation):
    def __init__(self, bias_scale_delegate=default_bias_scale_delegate):
        self.bias_scale_delegate = bias_scale_delegate
    
    def process_features(self, audio, features : torch.Tensor, userdata = {}):
        features += torch.ones_like(features)*default_bias_scale_delegate()
        return audio, features, userdata