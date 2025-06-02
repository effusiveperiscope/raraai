from dataset import FeatureDataset, FeatureCollator
from omegaconf import OmegaConf
from modeling.model import PASIFVAE
from features import MyFeatures
from utils import (load_submodule_prefix, visualize_phoneme_probabilities, compare_tensors_heatmap,
    generate_one_hot_logits, plot_logits_1)
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch

config = OmegaConf.load('configs/test12.yaml')
model = PASIFVAE(config)
my_feats = MyFeatures(no_whisper=True)
state_dict = torch.load('checkpoints/test12_v0_2/best-checkpoint.ckpt')['state_dict']
load_submodule_prefix(model, 'model.', state_dict)
model.eval()
model.to('cuda')

dataset = FeatureDataset(config, is_train=False)
collator = FeatureCollator(config)
loader = DataLoader(dataset, batch_size=1, collate_fn=collator)
spk_labels = [
        'p225', 'p226', 'p234', 'p299', 'p300', 'p376', 'p363', 'p340', 'p270', 'p271', 
        'p302', 'p311', 'p316', 'p318',
        'Pinkie_Sing', 'Rarity_Sing', 'Applejack_Sing', 'Flash Sentry_Sing', 
        'Sunset Shimmer_Sing', 'Rainbow_Sing', 'Twilight_Sing', 'Fluttershy_Sing',
        'ex01', 'ex02', 'ex03', 'ex04',
        'Twilight', 'Rarity', 'Rainbow', 'Pinkie', 'Applejack', 'Fluttershy',
        'Spike', 'Discord', 'Apple Bloom', 'Sweetie Belle', 'Granny Smith', 'Sunset Shimmer']

for batch in loader:
    whisper = batch['whisper']
    phones = batch['phones']
    spk_id = batch['spk_ids']
    whisper_mask = batch['whisper_mask']
    phones_mask = batch['phones_mask']

    print(f"Correct phonemes: { my_feats.ids_to_phonemes(phones.squeeze().tolist()) }")
    with torch.no_grad():
        y, phoneme_logits = model.generate(
            whisper, whisper_mask, spk_id)

    recon_loss = F.l1_loss(y, whisper)
    print(f"Recon loss: {recon_loss.item()}")

    visualize_phoneme_probabilities(phoneme_logits, my_feats.vocab)
    break