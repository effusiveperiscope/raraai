import librosa
from svc_helper.svc.rvc.lib.infer_pack.models import SynthesizerTrnMs768NSFsid
from features import MyFeatures
import torch
import soundfile as sf
from omegaconf import OmegaConf

import pdb
import sys
sys.excepthook = lambda exc_type, exc_value, exc_traceback: print(exc_type, exc_value, exc_traceback) or pdb.post_mortem(exc_traceback)

RVC_CKPT = 'tests/f0G48k.pth'
TEST_AUDIO = 'tests/test_me5.wav'
SID = 0
SUFFIX = f'titan_init2_test5'
CONFIG = 'configs/specialized.yaml'
TRANSPOSE = 0

config = OmegaConf.load(CONFIG)

data_16k, _ = librosa.load(TEST_AUDIO, sr=16000)
my_feats = MyFeatures()
feats = my_feats.get_features(data_16k)
lens = torch.tensor([feats['rvc_feat'].shape[1]]).to('cuda')

state = torch.load(RVC_CKPT)
model = SynthesizerTrnMs768NSFsid(**config.model, is_half=True)
state_dict = torch.load(RVC_CKPT, map_location='cpu')['model']
model.load_state_dict(state_dict)
model.to('cuda')
model = model.half()

basename = RVC_CKPT.removeprefix('checkpoints').replace('/', '_')[1:]

# Transpose
feats['pitch_fine'] = feats['pitch_fine'] * (2 ** (TRANSPOSE / 12))
feats['pitch'] = my_feats.f0_to_coarse(feats['pitch_fine']).squeeze(0)

feats['pitch'] = feats['pitch'].to('cuda')[:, :feats['rvc_feat'].shape[1]]
feats['pitch_fine'] = feats['pitch_fine'].to('cuda')[:, :feats['rvc_feat'].shape[1]]

with torch.no_grad():
    o, x_mask, z_stats = model.infer(
        feats['rvc_feat'].half(), 
        lens, 
        feats['pitch'].to('cuda'), 
        feats['pitch_fine'].to('cuda'),
        torch.tensor([SID]).to('cuda'))
    o_np = o.squeeze().cpu().float().numpy()
    sf.write(f'tests/out_{basename+SUFFIX}.wav', o_np, 48000)