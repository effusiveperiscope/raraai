import torch
from modeling.model import PASIFVAE
from utils import load_submodule_prefix, visualize_phoneme_probabilities
from omegaconf import OmegaConf
from features import MyFeatures
from dataset import FeatureCollator, FeatureDataset
from torch.utils.data import DataLoader
from torch import nn
from einops import rearrange

CONFIG = 'configs/test5.yaml'
CHECKPOINT = 'checkpoints/test5_v0_2/best-checkpoint.ckpt'
CE_THRESHOLD = 0.4

config = OmegaConf.load(CONFIG)
model = PASIFVAE(config)

my_feats = MyFeatures(no_whisper=True)

dataset = FeatureDataset(config, is_train=False)
collator = FeatureCollator(config)
loader = DataLoader(dataset, batch_size=1, collate_fn=collator)

state_dict = torch.load(CHECKPOINT)['state_dict']
load_submodule_prefix(model, 'model.', state_dict)

model.eval()
for batch in loader:
    whisper = batch['whisper']
    phones = batch['phones']
    spk_id = batch['spk_ids']
    whisper_mask = batch['whisper_mask']
    phones_mask = batch['phones_mask']
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

    new_phones = new_phones.to(phones.device)
    new_phones_mask = new_phones_mask.to(phones_mask.device)

    # Find the position to insert EOS (first False in mask or end of sequence)
    for i in range(phones.shape[0]):
        # Find the index where valid tokens end (first False or end of sequence)
        valid_length = phones_mask[i].sum().item()  # Number of True values
        eos_pos = valid_length  # Position after the last valid token
        new_phones[i, eos_pos] = config.model.eos_token_id
        new_phones_mask[i, eos_pos] = True  # Mark EOS as valid

    with torch.no_grad():
        y, phone_logits, spk_logits, m_p, log_var_p = model(
            whisper,
            whisper_mask,
            new_phones,
            new_phones_mask,
            spk_id)
            
    recon_loss = nn.L1Loss()(y, whisper)
    phone_ce_loss = nn.CrossEntropyLoss(ignore_index=config.model.pad_token_id)(
        rearrange(phone_logits[:, :-1, :], 'b s c -> b c s'), new_phones[:, 1:])
    spk_ce_loss = nn.CrossEntropyLoss()(spk_logits, spk_id)

    true_phonemes = my_feats.ids_to_phonemes(new_phones.squeeze().tolist())

    if phone_ce_loss < CE_THRESHOLD:
        continue

    print(basename)
    print(true_phonemes)
    print(f"Recon loss: {recon_loss.item()}")
    print(f"Phone CE loss: {phone_ce_loss.item()}")
    print(f"Speaker CE loss: {spk_ce_loss.item()}")
    print('\n')

    visualize_phoneme_probabilities(phone_logits, my_feats.all_phonemes(), window_size=None)
    #import pdb; pdb.set_trace()