from abc import ABC
import torch

class DataAugmentation(ABC):
    """ Intakes audio (needed for certain operations) and a set of features
    (torch.Tensor) and userdata. 
    Outputs the unchanged audio and features
        (torch.Tensor) and userdata."""
    def process_features(self, audio, features : torch.Tensor, userdata : dict):
        return audio, features