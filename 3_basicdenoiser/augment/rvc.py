from svc_helper.svc.rvc import RVCModel
from svc_helper.sfeatures.models import RVCHubertModel
from augment.base import DataAugmentation
import random
import torch

#"""Augmentation module that randomly selects from a set of models/index paths"""
# Problem: This can probably only be done in preprocessing, prior to the audio
class RVCAugmentation(DataAugmentation):
    expected_sample_rate = RVCModel.expected_sample_rate
    """ratio_generator is a function that outputs what index ratio to use"""
    def __init__(self,
        device,
        model_path,
        index_path,
        f0_mean,
        ratio_generator = random.random):
        self.model = RVCModel()
        self.hubert_model = RVCHubertModel(device)
        self.model.load_model(model_path, index_path)
        self.f0_mean = f0_mean
        self.ratio_generator = ratio_generator

    def process_features(self, audio, features : torch.Tensor, userdata = {}):
        old_device = features.device
        features = features.to(self.device)
        proc_audio = self.model.infer_audio(
            audio, target_pitch=self.f0_mean,
            index_rate=self.ratio_generator(),
            feature_transform=lambda f: features)
        feats = self.hubert_model.extract_features(
            torch.from_numpy(proc_audio))
        return audio, feats.to(old_device), userdata