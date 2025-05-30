from einops import rearrange
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from dataset import FeatureDataset, FeatureCollator
from torch.utils.data import DataLoader
from utils import plot_phoneme_probabilities, greedy_decode_phonemes
from torch import nn
from model import PASIFVAE
from utils import load_submodule_prefix
from features import MyFeatures

cfg = OmegaConf.load("configs/test1.yaml")
dataset = FeatureDataset(cfg, is_train=False)
my_feat = MyFeatures(no_whisper=True)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=FeatureCollator(cfg))

model = PASIFVAE(cfg)
state_dict = torch.load("checkpoints/test1/last.ckpt")
load_submodule_prefix(
    model, "model.", state_dict)
model.eval()

loss_ctc = nn.CTCLoss(
    blank=cfg.model.space_id, zero_infinity=True)

stats = {
    'without_hint': [],
    'with_hint': [],
}

def step(batch, use_phoneme_hint):
    whisper = batch["whisper"]
    phones = batch["phones"]
    spk_id = batch["spk_ids"]
    whisper_mask = batch["whisper_mask"]
    phones_mask = batch["phones_mask"]
    basename = batch["basenames"]

    with torch.no_grad():
        phone_logits, m_p, log_var_p, y, speaker_logits = model(
            whisper, whisper_mask, spk_id, 
            phones if use_phoneme_hint else None, 
            phones_mask if use_phoneme_hint else torch.zeros_like(phones_mask))

    #import pdb; pdb.set_trace()
    plot_phoneme_probabilities(phone_logits, my_feat.all_phonemes() + [' '] * 4,
        title=f"Phoneme Probabilities {'Hint' if use_phoneme_hint else 'No Hint'} for {basename[0]}")
    # print(greedy_decode_phonemes(phone_logits, my_feat.all_phonemes() + [' '] * 4))

    phone_logits = rearrange(phone_logits, "B T C -> T B C")
    phone_log_probs = F.log_softmax(phone_logits.float(), dim=-1)

    ctc_loss = loss_ctc(
        phone_log_probs.float(),
        phones, 
        whisper_mask.sum(-1), 
        phones_mask.sum(-1))
    recon_loss = F.l1_loss(y, whisper)
    speaker_loss = nn.CrossEntropyLoss()(speaker_logits, spk_id)
    return ctc_loss, recon_loss, speaker_loss

print('Performance without phoneme hint')
for batch in dataloader:
    results = step(batch, False)
    stats['without_hint'].append({
        'basename': batch["basenames"][0],
        'ctc_loss': results[0].item(),
        'recon_loss': results[1].item(),
        'speaker_loss': results[2].item(),
    })

print('Performance with phoneme hint')
for batch in dataloader:
    results = step(batch, True)
    stats['with_hint'].append({
        'basename': batch["basenames"][0],
        'ctc_loss': results[0].item(),
        'recon_loss': results[1].item(),
        'speaker_loss': results[2].item(),
    })