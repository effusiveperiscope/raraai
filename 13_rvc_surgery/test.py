import librosa
from svc_helper.svc.rvc.lib.infer_pack.models import SynthesizerTrnMs768NSFsid
from features import MyFeatures
import torch
import soundfile as sf

import sys
import pdb
import os
sys.excepthook = lambda exc_type, exc_value, exc_traceback: pdb.post_mortem(exc_traceback)

RVC_CKPT = 'tests/RarityTitan.pth'
TEST_AUDIO = 'tests/test_speech2.flac'
NEW_CKPT = 'checkpoints/specialized/best-checkpoint-v1.ckpt'

state = torch.load(RVC_CKPT)
model = SynthesizerTrnMs768NSFsid(*state['config'], is_half=True)
del model.enc_q # posterior is not stored in this style of checkpoint
model.load_state_dict(state['weight'])
model = model.to('cuda')
model = model.half()
enc_p = model.enc_p # inputs [b, t, h]

# why are they feeding the pitch into the prior encoder?

# our goal: retool the encoder to use whisper features (?)
data_16k, _ = librosa.load(TEST_AUDIO, sr=16000)
my_feats = MyFeatures()
feats = my_feats.get_features(data_16k)
lens = torch.tensor([feats['rvc_feat'].shape[1]]).to('cuda')

# Baseline
with torch.no_grad():
    o, x_mask, z_stats = model.infer(
        feats['rvc_feat'].half(), 
        lens, 
        feats['pitch'].to('cuda'), 
        feats['pitch_fine'].to('cuda'),
        torch.tensor([0]).to('cuda'))
    o_np = o.squeeze().cpu().float().numpy()
    sf.write('tests/out.wav', o_np, 48000)

# New (using whisper)
basename = NEW_CKPT.removeprefix('checkpoints').replace('/', '_')[1:]
basename = basename.replace('.ckpt', '')
state_dict = torch.load(NEW_CKPT, map_location='cpu')['state_dict']
submodule_prefix = 'student_enc.'
submodule_state_dict = {
    k[len(submodule_prefix):]: v 
    for k, v in state_dict.items() 
    if k.startswith(submodule_prefix)
}
model.enc_p.load_state_dict(submodule_state_dict)
with torch.no_grad():
    o, x_mask, z_stats = model.infer(
        feats['whisp_feat'].half(), 
        lens, 
        feats['pitch'].to('cuda'), 
        feats['pitch_fine'].to('cuda'),
        torch.tensor([0]).to('cuda'))
    o_np = o.squeeze().cpu().float().numpy()
    sf.write(f'tests/out_{basename}.wav', o_np, 48000)