import torch
import numpy as np
import random
from sklearn.decomposition import PCA
from augment.base import DataAugmentation

def default_apply_perturbations_delegate():
    return [
        0,
        6.0 - random.random()*12.0,
        6.0 - random.random()*12.0,
        6.0 - random.random()*12.0,
        ]

# Randomly perturbs PCA dimensions of speech features
class PCAPerturbation(DataAugmentation):
    def __init__(self,
        n_components=0.9,
        apply_perturbations_delegate=default_apply_perturbations_delegate):
        self.n_components = n_components
        self.apply_perturbations_delegate = apply_perturbations_delegate

    def process_features(self, audio, features : torch.Tensor, userdata = {}):
        out_features = []
        assert(features.dim() == 3)
        for feature_set in features:
            # Reshape the tensor to 2D for PCA
            original_shape = feature_set.shape
            flattened = feature_set.reshape(-1, original_shape[-1])

            # Perform PCA
            pca = PCA(n_components=self.n_components)
            pca_components = pca.fit_transform(flattened.cpu().numpy())

            # Convert back to PyTorch tensor
            pca_components = torch.from_numpy(pca_components).float()

            # Generate random perturbations
            perturbations = torch.zeros_like(pca_components)
            apply_perturbations = torch.tensor(self.apply_perturbations_delegate())
            #print(len(apply_perturbations))
            #print(perturbations.shape)
            perturbations[:, :len(apply_perturbations)] = apply_perturbations

            # Apply perturbations in PCA space
            perturbed_pca = pca_components + perturbations

            # Transform back to original space
            perturbed_features = torch.from_numpy(
                pca.inverse_transform(perturbed_pca.numpy())
            ).float()

            # Reshape back to original tensor shape
            perturbed_tensor = perturbed_features.reshape(original_shape)
            out_features.append(perturbed_tensor)

        out_tensor = torch.stack(out_features)

        return audio, out_tensor.to(features.device), userdata