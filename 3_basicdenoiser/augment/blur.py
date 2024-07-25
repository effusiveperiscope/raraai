import torch
import torch.nn.functional as F
import math
import random
from augment.base import DataAugmentation

class GaussianBlur(DataAugmentation):
    def __init__(
        self, 
        sigma_delegate = lambda: random.random()*5.0,
        kernel_size_delegate = lambda: 7):
        self.sigma_delegate = sigma_delegate
        self.kernel_size_delegate = kernel_size_delegate
        pass

    def process_features(self, audio, features : torch.Tensor, userdata = {}):
        sigma = self.sigma_delegate()
        kernel_size = self.kernel_size_delegate()

        if sigma == 0.0:
            return features

        # Ensure odd kernel size
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        
        # Create a 1D Gaussian kernel
        # Create a range of values centered around zero
        x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1,
            dtype=torch.float32)
        # Compute the 1D Gaussian kernel
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        # Normalize the kernel
        kernel = kernel / kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        # Pad the input tensor
        padding = (kernel_size - 1) // 2
        padded_tensor = F.pad(features, (0, 0, padding, padding), mode='reflect')
        # Reshape the tensor to [768, 1, 456] to apply convolution separately on each channel
        reshaped = padded_tensor.permute(2, 0, 1)
        # Ensure same dtype
        kernel = kernel.to(features.dtype).to(features.device)
        # Apply the filter
        smoothed = F.conv1d(reshaped, kernel)
        # Reshape back to original dimensions
        smoothed = smoothed.permute(1, 2, 0)
        return audio, smoothed.to(features.device), userdata