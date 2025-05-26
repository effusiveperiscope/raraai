import os
from PyQt5.QtCore import pyqtRemoveInputHook
from PyQt5.QtWidgets import QApplication, QMainWindow
import librosa
import numpy as np
from modeling.my_rvc import VevoSynthesizer
from logging import getLogger
from svc_helper.gui import *
from omegaconf import OmegaConf
from features import MyFeatures
import torch
import soundfile as sf

logger = getLogger(__name__)
CHECKPOINTS_ROOT = 'checkpoints/titan_spk_vevo_stage2'

class MainWindow(QMainWindow):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.setWindowTitle("13 RVC TEST GUI")
        self.setGeometry(100, 100, 800, 600)

        gui = VoiceGUI()
        gui.addCheckpoint(Checkpoint(
            get_checkpoints=self.getCheckpoints, load_checkpoint=self.loadCheckpoint))
        gui.addFileInput(AudioFileInput())
        gui.addParam(IntParam(label="Transpose", id='transpose', min=-24, max=24, default=0))
        gui.addParam(IntParam(label="Speaker", id='sid', min=0, max=110, default=0))
        # TODO - pitch smooth
        gui.addInference(Inference(
            info=InferenceInfo(sr=48000, extension='flac'),
            infer_action=self.inferAction
        ))
        self.setCentralWidget(gui.build())

        my_feats = MyFeatures(
            extract_hubert=False, extract_whisper=False, extract_vevo=True)
        self.my_feats = my_feats

        self.hubert_stats_path = config.data.get('hubert_stats_path', 
            'hubert_large_stat.npz')
        stat = np.load(self.hubert_stats_path)
        self.hubert_feat_norm_mean = torch.tensor(stat["mean"])
        self.hubert_feat_norm_std = torch.tensor(stat["std"])

        self.config = config

    def getCheckpoints(self):
        return os.listdir(CHECKPOINTS_ROOT)

    def loadCheckpoint(self, checkpoint_name):
        logger.info(f'Loading checkpoint {checkpoint_name}')
        checkpoint_path = os.path.join(CHECKPOINTS_ROOT, checkpoint_name)
        # Points to a lightning checkpoint
        self.net_g = VevoSynthesizer(**self.config.model, is_half=True)

        state = torch.load(checkpoint_path, map_location='cpu')['state_dict']
        submodule_prefix = 'net_g.'
        state_dict = {
            k[len(submodule_prefix):]: v 
            for k, v in state.items() 
            if k.startswith(submodule_prefix)
        }
        self.net_g.load_state_dict(state_dict)
        self.net_g.to('cuda')
        self.net_g.eval()
        self.net_g.half()
        logger.info(f'Checkpoint {checkpoint_name} loaded')

    def inferAction(self, data: dict):
        transpose = data['transpose']
        files = data['audio_files']['files']

        if not hasattr(self, 'net_g'):
            logger.error('No checkpoint loaded')
            return InferenceResult(audios=[])

        logger.info(f'Infering {len(files)} files')

        out = []
        for file in files:
            data_16k, _ = librosa.load(file, sr=16000)
            feats = self.my_feats.get_features(data_16k)
            lens = torch.tensor([feats['vevo_quantized'].shape[1]]).to('cuda')

            # Normalize features
            feats['vevo_quantized'] = (
                (feats['vevo_quantized'] - self.hubert_feat_norm_mean.to(feats['vevo_quantized'].device)) /
                self.hubert_feat_norm_std.to(feats['vevo_quantized'].device))
            
            # Tranpsose
            feats['pitch_fine'] = feats['pitch_fine'] * (2 ** (transpose / 12))
            feats['pitch'] = self.my_feats.f0_to_coarse(feats['pitch_fine'].squeeze(0))

            # Truncate
            feats['pitch'] = feats['pitch'].to('cuda')[:, :feats['vevo_quantized'].shape[1]]
            feats['pitch_fine'] = feats['pitch_fine'].to('cuda')[:, :feats['vevo_quantized'].shape[1]]

            with torch.no_grad():
                o, x_mask, z_stats = self.net_g.infer(
                    feats['vevo_quantized'].half(), 
                    lens, 
                    feats['pitch'].to('cuda'), 
                    feats['pitch_fine'].to('cuda'),
                    torch.tensor([data['sid']]).to('cuda'))
                o_np = o.squeeze().cpu().float().numpy()
                out.append(AudioResult(
                    label=os.path.basename(file)+data['model_labels'][0],
                    audio=o_np))
        logger.info(f'Finished infering {len(files)} files')
        return InferenceResult(audios=out)

if __name__ == '__main__':
    app = QApplication([])
    config = OmegaConf.load('configs/titan_spk.yaml')
    window = MainWindow(config)
    window.show()
    app.exec_()