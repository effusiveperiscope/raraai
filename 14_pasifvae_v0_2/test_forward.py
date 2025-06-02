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

    # Step 1: Add BOS token
    phones = torch.cat(
        [
            torch.full((phones.shape[0], 1), config.model.bos_token_id, dtype=torch.long, device=phones.device),
            phones
        ],
        dim=1
    )
    # Step 2: Update phones_mask for BOS token
    phones_mask = torch.cat(
        [
            torch.ones((phones_mask.shape[0], 1), dtype=torch.bool, device=phones_mask.device),  # True for BOS
            phones_mask
        ],
        dim=1
    )

    # Step 3: Add EOS token
    # Create a new tensor for phones with +1 length for EOS
    new_phones = torch.full(
        (phones.shape[0], phones.shape[1] + 1),
        config.model.pad_token_id,
        dtype=torch.long,
        device=phones.device
    )
    new_phones_mask = torch.zeros(
        (phones_mask.shape[0], phones_mask.shape[1] + 1),
        dtype=torch.bool,
        device=phones_mask.device
    )

    # Copy original phones and mask
    new_phones[:, :-1] = phones
    new_phones_mask[:, :-1] = phones_mask

    new_phones = new_phones.to(phones.device)
    new_phones_mask = new_phones_mask.to(phones_mask.device)

    # Find the position to insert EOS (first False in mask or end of sequence)
    for i in range(phones.shape[0]):
        # Find the index where valid tokens end (first False or end of sequence)
        valid_length = phones_mask[i].sum().item()  # Number of True values
        eos_pos = valid_length  # Position after the last valid token
        new_phones[i, eos_pos] = config.model.eos_token_id
        new_phones_mask[i, eos_pos] = True  # Mark EOS as valid

    print(f"Correct phonemes: { my_feats.ids_to_phonemes(phones.squeeze().tolist()) }")

    y, phone_logits, spk_logits, m_p, log_var_p = model(
        whisper, whisper_mask, new_phones, new_phones_mask, spk_id)
    recon_loss = F.l1_loss(y, whisper)
    print(f"Recon loss with correct speaker: {recon_loss.item()}")
    print(f"Correct speaker is: {spk_labels[spk_id.item()]}")

    # plot_logits_1(spk_logits, spk_labels, title="Speaker Logits")

    print(" -- incorrect speaker test -- ")
    y, phone_logits, spk_logits, m_p, log_var_p = model(
        whisper, whisper_mask, phones, phones_mask, spk_id - 1 if spk_id != 0 else torch.tensor([1], dtype=torch.long, device=phones.device))
    recon_loss = F.l1_loss(y, whisper)
    print(f"Recon loss with incorrect speaker: {recon_loss.item()}")

    print(" -- incorrect phonemes test --")
    random_phones, ids = generate_one_hot_logits(B=1, T=phones.shape[1], num_classes=config.model.n_phonemes + 3)
    print(f"Random phonemes: { my_feats.ids_to_phonemes(ids.squeeze().tolist()) }")
    y = model.force_phonemes(whisper, whisper_mask, 
        random_phones,
        phones_mask, spk_id)
    recon_loss = F.l1_loss(y, whisper)
    print(f"Recon loss with random phonemes: {recon_loss.item()}")

    print("-- combined test --")
    y = model.force_phonemes(whisper, whisper_mask, 
        random_phones,
        phones_mask, spk_id - 1 if spk_id != 0 else torch.tensor([1], dtype=torch.long, device=phones.device))
    recon_loss = F.l1_loss(y, whisper)
    print(f"Recon loss with random phonemes and incorrect speaker: {recon_loss.item()}")

    # compare_tensors_heatmap(whisper, y, title1='Whisper', title2='Reconstructed')
