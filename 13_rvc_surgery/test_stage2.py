import librosa
from svc_helper.svc.rvc.lib.infer_pack.models import SynthesizerTrnMs768NSFsid
from features import MyFeatures
import torch
import soundfile as sf
from omegaconf import OmegaConf

LT_CKPT = 'checkpoints/titan_stage2/last.ckpt'
TEST_AUDIO = 'tests/test_me4.wav'
SUFFIX = 'stage2_test4_new'
CONFIG = 'configs/titan.yaml'
TRANSPOSE = 0

config = OmegaConf.load(CONFIG)

data_16k, _ = librosa.load(TEST_AUDIO, sr=16000)
my_feats = MyFeatures()
feats = my_feats.get_features(data_16k)
lens = torch.tensor([feats['rvc_feat'].shape[1]]).to('cuda')

state = torch.load(LT_CKPT)
model = SynthesizerTrnMs768NSFsid(**config.model, is_half=True)
state_dict = torch.load(LT_CKPT, map_location='cpu')['state_dict']
submodule_prefix = 'net_g.'
submodule_state_dict = {
    k[len(submodule_prefix):]: v 
    for k, v in state_dict.items() 
    if k.startswith(submodule_prefix)
}
model.load_state_dict(submodule_state_dict)
model.to('cuda')
model = model.half()

basename = LT_CKPT.removeprefix('checkpoints').replace('/', '_')[1:]

# Transpose
feats['pitch_fine'] = feats['pitch_fine'] * (2 ** (TRANSPOSE / 12))
feats['pitch'] = my_feats.f0_to_coarse(feats['pitch_fine'].squeeze(0))

feats['pitch'] = feats['pitch'].to('cuda')[:, :feats['whisp_feat'].shape[1]]
feats['pitch_fine'] = feats['pitch_fine'].to('cuda')[:, :feats['whisp_feat'].shape[1]]


with torch.no_grad():
    o, x_mask, z_stats = model.infer(
        feats['whisp_feat'].half(), 
        lens, 
        feats['pitch'].to('cuda'), 
        feats['pitch_fine'].to('cuda'),
        torch.tensor([0]).to('cuda'))
    o_np = o.squeeze().cpu().float().numpy()
    sf.write(f'tests/out_{basename+SUFFIX}.wav', o_np, 48000)