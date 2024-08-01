import torch
import numpy as np
from einops import rearrange
from train import load_checkpoint
from omegaconf import OmegaConf

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument('--ckpt', dest='ckpt', action='store')
parser.add_argument('--conf', dest='conf', default='config.yaml', action='store')

if __name__ == '__main__':

    args = parser.parse_args()
    from model import SpeechClassifier1
    from svc_helper.sfeatures.models import RVCHubertModel
    import librosa

    sfeatures_model = RVCHubertModel()
    audio, sr = librosa.load(args.filename,
        sr=RVCHubertModel.expected_sample_rate)

    print('Extracting features')
    feats = sfeatures_model.extract_features(audio=torch.tensor(audio))
    print('Done extracting features')

    #torch.manual_seed(0)
    #np.random.seed(0)

    conf = OmegaConf.load(args.conf)
    _, _, label_encoder, model = load_checkpoint(
        conf, args.ckpt)
    # eval mode b/c of dropout!
    model.eval()
    with torch.no_grad():
        feats = rearrange(feats, 'b n c -> n b c')
        logits = model(feats)
        print(logits)
        print(logits.argmax().item())
        probs = logits.softmax(dim=1).squeeze()
        pred_idx = logits.argmax().item()
        print(f'Predicted character: {label_encoder[pred_idx]} | '
            f'Prob {probs[pred_idx].item()}')

        topn = 5
        pred_idx = np.argpartition(probs, -topn)[-topn:]
        pred_idx = pred_idx[np.argsort(probs[pred_idx])]
        for j,idx in enumerate(pred_idx):
            print(f'top {len(pred_idx)-j}: {label_encoder[idx]}')