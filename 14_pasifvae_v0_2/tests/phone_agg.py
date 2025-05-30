import torch
from modeling.model import PASIFVAE
from utils import load_submodule_prefix, visualize_phoneme_probabilities
from omegaconf import OmegaConf
from features import MyFeatures
from dataset import FeatureCollator, FeatureDataset
from torch.utils.data import DataLoader
from torch import nn
from einops import rearrange
from collections import defaultdict # For potential dataset-wide aggregation later
from tqdm import tqdm

CONFIG = 'configs/test5.yaml'
CHECKPOINT = 'checkpoints/test5_v0_2/best-checkpoint.ckpt'
SID_MAPPING = ['p225', 'p226', 'p234', 'p299', 'p300', 'p376', 'p363', 'p340', 'p270', 'p271', 'ex01', 'ex02', 'ex03', 'ex04', 'Pinkie', 'Rarity', 'Applejack', 'Flash Sentry', 'Sunset Shimmer', 'Rainbow', 'Twilight', 'Fluttershy']

config = OmegaConf.load(CONFIG)
model = PASIFVAE(config)
model.to('cuda')

my_feats = MyFeatures(no_whisper=True)

dataset = FeatureDataset(config, is_train=False)
collator = FeatureCollator(config)
loader = DataLoader(dataset, batch_size=4, collate_fn=collator)

state_dict = torch.load(CHECKPOINT)['state_dict']
load_submodule_prefix(model, 'model.', state_dict)

model.eval()

# For dataset-wide aggregation (optional, if needed later)
total_per_speaker_phoneme_loss_sum = defaultdict(float)
total_per_speaker_phoneme_loss_count = defaultdict(int)

for batch_idx, batch in tqdm(enumerate(loader), total=len(loader)):
    whisper = batch['whisper'].to('cuda')
    phones = batch['phones'].to('cuda')
    spk_id = batch['spk_ids'].to('cuda')
    whisper_mask = batch['whisper_mask'].to('cuda')
    phones_mask = batch['phones_mask'].to('cuda')
    basename = batch['basenames'][0]

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

    # Find the position to insert EOS (first False in mask or end of sequence)
    for i in range(phones.shape[0]):
        valid_length = phones_mask[i].sum().item()  # Number of True values
        eos_pos = valid_length  # Position after the last valid token
        new_phones[i, eos_pos] = config.model.eos_token_id
        new_phones_mask[i, eos_pos] = True  # Mark EOS as valid

    with torch.no_grad():
        y, phone_logits, spk_logits, m_p, log_var_p = model(
            whisper,
            whisper_mask,
            new_phones.to('cuda'), # ensure new_phones is on the same device as model
            new_phones_mask.to('cuda'), # ensure new_phones_mask is on the same device as model
            spk_id)

    recon_loss = nn.L1Loss()(y, whisper)

    # Define the phoneme CrossEntropyLoss function (reused for overall and per-speaker)
    phoneme_ce_criterion = nn.CrossEntropyLoss(ignore_index=config.model.pad_token_id)

    # Overall phoneme CE loss for the batch
    # Targets are new_phones shifted by 1 (excluding BOS), predictions exclude last token
    overall_phone_ce_loss = phoneme_ce_criterion(
        rearrange(phone_logits[:, :-1, :], 'b s c -> b c s'), # (Batch, Classes, Seq_len-1)
        new_phones[:, 1:]                                   # (Batch, Seq_len-1)
    )

    spk_ce_loss = nn.CrossEntropyLoss()(spk_logits, spk_id)

    # --- Aggregate phoneme cross entropy loss by speaker class ---
    per_speaker_phone_ce_loss = {}
    unique_speaker_ids_in_batch = torch.unique(spk_id)

    for speaker_id_tensor in unique_speaker_ids_in_batch:
        current_speaker_id = speaker_id_tensor.item()
        
        # Create a boolean mask for samples belonging to the current speaker
        speaker_indices = (spk_id == speaker_id_tensor)

        # Filter phone_logits and new_phones for the current speaker
        speaker_phone_logits = phone_logits[speaker_indices]
        speaker_target_phones = new_phones[speaker_indices]

        if speaker_phone_logits.shape[0] > 0: # Should always be true due to torch.unique
            # Calculate phoneme CE loss for this specific speaker's data in the batch
            # Logits: (Num_speaker_samples, Seq_len, Num_classes_phoneme)
            # Targets: (Num_speaker_samples, Seq_len_target)
            # We align them: logits[:, :-1, :] and targets[:, 1:]
            loss_for_speaker = phoneme_ce_criterion(
                rearrange(speaker_phone_logits[:, :-1, :], 'b s c -> b c s'),
                speaker_target_phones[:, 1:]
            )
            per_speaker_phone_ce_loss[current_speaker_id] = loss_for_speaker.item()
            
            # Optional: For dataset-wide aggregation
            total_per_speaker_phoneme_loss_sum[current_speaker_id] += loss_for_speaker.item() * speaker_phone_logits.shape[0]
            total_per_speaker_phoneme_loss_count[current_speaker_id] += speaker_phone_logits.shape[0]
# 
    # print(f"--- Batch {batch_idx} (Basename: {basename}) ---")
    # print(f"Overall Recon Loss: {recon_loss.item():.4f}")
    # print(f"Overall Phoneme CE Loss: {overall_phone_ce_loss.item():.4f}")
    # print(f"Overall Speaker CE Loss: {spk_ce_loss.item():.4f}")
    # print(f"Per-speaker Phoneme CE Loss for this batch: {per_speaker_phone_ce_loss}")
    # print("-" * 30)

    # Example: Visualize for the first item in the batch if needed
    # if batch_idx == 0:
    #     visualize_phoneme_probabilities(
    #         phone_logits[0, :-1, :].cpu().float().softmax(dim=-1), # Probs for actual phonemes (excluding EOS prediction)
    #         new_phones[0, 1:].cpu(), # Target phonemes (excluding BOS, including EOS)
    #         config.model.pad_token_id,
    #         "visualization_output.png" # Make sure you have a phoneme map or list of phonemes
    #     )
    # break # Process only one batch for testing

# After the loop, if you want dataset-wide averages:
average_per_speaker_phoneme_loss_dataset = {
    spk_id: total_per_speaker_phoneme_loss_sum[spk_id] / total_per_speaker_phoneme_loss_count[spk_id]
    for spk_id in total_per_speaker_phoneme_loss_count if total_per_speaker_phoneme_loss_count[spk_id] > 0
}
average_per_speaker_phoneme_loss_dataset = {
    SID_MAPPING[spk_id]: loss for spk_id, loss in average_per_speaker_phoneme_loss_dataset.items()
}
if average_per_speaker_phoneme_loss_dataset:
    print("\n--- Dataset-wide Averages ---")
    print(f"Average Per-speaker Phoneme CE Loss (Dataset): {average_per_speaker_phoneme_loss_dataset}")

import matplotlib.pyplot as plt

# Sort the dictionary by loss
sorted_items = sorted(average_per_speaker_phoneme_loss_dataset.items(), key=lambda item: item[1])

# Unpack keys and values
speakers, losses = zip(*sorted_items)

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(speakers, losses, color='skyblue')
plt.xlabel('Speaker ID')
plt.ylabel('Phoneme Loss')
plt.title('Average Phoneme Loss per Speaker (Sorted)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Save the plot to a file
plt.savefig("average_phoneme_loss_per_speaker.png")