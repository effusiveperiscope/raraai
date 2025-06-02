from dataset import FeatureDataset, FeatureCollator
from omegaconf import OmegaConf
from modeling.model import PASIFVAE
from features import MyFeatures
from utils import (load_submodule_prefix, visualize_phoneme_probabilities, compare_tensors_heatmap,
    generate_one_hot_logits, plot_logits_1)
from torch.utils.data import DataLoader
from torch.nn import functional as F
from einops import rearrange
import librosa
import torch

config = OmegaConf.load('configs/test12.yaml')
model = PASIFVAE(config)
my_feats = MyFeatures()
state_dict = torch.load('checkpoints/test12_v0_2/best-checkpoint.ckpt')['state_dict']
load_submodule_prefix(model, 'model.', state_dict)
model.eval()
model.to('cuda')

spk_labels = [
        'p225', 'p226', 'p234', 'p299', 'p300', 'p376', 'p363', 'p340', 'p270', 'p271', 
        'p302', 'p311', 'p316', 'p318',
        'Pinkie_Sing', 'Rarity_Sing', 'Applejack_Sing', 'Flash Sentry_Sing', 
        'Sunset Shimmer_Sing', 'Rainbow_Sing', 'Twilight_Sing', 'Fluttershy_Sing',
        'ex01', 'ex02', 'ex03', 'ex04',
        'Twilight', 'Rarity', 'Rainbow', 'Pinkie', 'Applejack', 'Fluttershy',
        'Spike', 'Discord', 'Apple Bloom', 'Sweetie Belle', 'Granny Smith', 'Sunset Shimmer']

test_audio = 'tests/test2.wav'
test_audio, _ = librosa.load(test_audio, sr=MyFeatures.expected_sample_rate)
whisper = my_feats.get_whisper_features(test_audio)

# RVC expects speech feature seq dim to be interpolated up 2x
whisper = rearrange(whisper, "1 T C -> 1 C T")
whisper = F.interpolate(whisper, scale_factor=2)
whisper = rearrange(whisper, "1 C T -> 1 T C")
whisper_mask = torch.full(
    (whisper.shape[0], whisper.shape[1]), True, dtype=torch.bool).to(whisper.device)

phone_logits, _, _ = model.encoder.generate(whisper, whisper_mask)

# Get the predicted phoneme IDs using argmax
predicted_ids = torch.argmax(phone_logits, dim=-1)  # Shape: [1, T]

# Remove the batch dimension to get just the sequence
predicted_ids = predicted_ids.squeeze(0)  # Shape: [T]

# Convert to list if needed
predicted_ids_list = predicted_ids.tolist()

predicted_phonemes = my_feats.ids_to_phonemes(predicted_ids_list)
print(f"Predicted phonemes: {predicted_phonemes}")