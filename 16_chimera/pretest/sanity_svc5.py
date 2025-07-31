import torch
from modeling.vits.models import SynthesizerTrnOrig
from features import MyFeatures
from dataset import interp2
from omegaconf import OmegaConf
import ultimate_xc
import soundfile as sf

def sanity_check():
    extractor = MyFeatures(do_normalize=False)
    feats = extractor.extract_features("test/test.wav")
    hp = OmegaConf.load("config/svc5_orig.yaml")

    model = SynthesizerTrnOrig(
        spec_channels=hp.data.filter_length // 2 + 1,
        segment_size=hp.data.segment_size // hp.data.hop_length,
        hp=hp
    )
    state_dict = torch.load('pretrain/svc5.pth', map_location='cpu')['model_g']
    model.load_state_dict(state_dict)

    device = 'cuda'
    model.to(device)
    model.eval()
    ppg = interp2(feats['whisper']).unsqueeze(0).float().to(device)
    vec = interp2(feats['hubert']).unsqueeze(0).float().to(device)
    pit = feats['f0'].unsqueeze(0).float().to(device)
    pit = pit[:, :ppg.shape[1]]
    spk = feats['spk'].unsqueeze(0).float().to(device)

    from commons import plot_spectrogram
    plot_spectrogram(feats['spec'], save_path='test/new_spec.png')
    plot_spectrogram(extractor.orig_spectrogram('test/test.wav'), save_path='test/orig_spec.png')

    ppg_l = torch.tensor([ppg.shape[1]]).to(device)

    with torch.no_grad():
        audio = model.infer(ppg, vec, pit, spk, ppg_l).squeeze().cpu().numpy()
    sf.write("test/test_out.wav", audio, hp.data.sampling_rate)

if __name__ == "__main__":
    sanity_check()