import ultimate_xc
import os
from PyQt5.QtCore import pyqtRemoveInputHook
from PyQt5.QtWidgets import QApplication, QMainWindow
import librosa
import numpy as np
from modeling.vits.models import SynthesizerTrn
from logging import getLogger
from svc_helper.gui import *
from omegaconf import OmegaConf
from features import MyFeatures
from commons import load_submodule_prefix
from dataset import interp_row
import torch
import soundfile as sf

logger = getLogger(__name__)
CHECKPOINTS_ROOT = 'checkpoints/test05_stage3'

class MainWindow(QMainWindow):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.config = config
        self.setWindowTitle("16 CHIMERA TEST GUI")
        self.setGeometry(100, 100, 800, 600)

        gui = VoiceGUI()
        gui.addCheckpoint(Checkpoint(
            get_checkpoints=self.getCheckpoints, load_checkpoint=self.loadCheckpoint))
        gui.addFileInput(AudioFileInput())
        gui.addParam(IntParam(label="Transpose", id='transpose', min=-24, max=24, default=0))
        gui.addParam(IntParam(label="Prior Transpose", id='coarse', min=-24, max=24, default=0))
        gui.addParam(DoubleParam(label="Noise Scale", id='noise', min=0, max=3, default=0.5))
        gui.addParam(DoubleParam(label="Noise Aug Scale", id='noise_aug', min=0, max=3, default=0.0))
        self.spk_index = torch.load(self.config.train.spk_index)
        gui.addParam(IntParam(label="Speaker", id='sid', min=0, max=len(self.spk_index) - 1, default=0))
        # TODO - pitch smooth
        gui.addInference(Inference(
            info=InferenceInfo(sr=48000, extension='flac'),
            infer_action=self.inferAction
        ))
        self.setCentralWidget(gui.build())

        my_feats = MyFeatures(
            feats_to_extract={'whisper', 'hubert', 'f0'})
        self.my_feats = my_feats
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # self.hubert_stats_path = config.data.get('hubert_stats_path', 
        #     'hubert_large_stat.npz')
        # stat = np.load(self.hubert_stats_path)
        # self.hubert_feat_norm_mean = torch.tensor(stat["mean"])
        # self.hubert_feat_norm_std = torch.tensor(stat["std"])

        self.config = config

    def getCheckpoints(self):
        return os.listdir(CHECKPOINTS_ROOT)

    def loadCheckpoint(self, checkpoint_name):
        logger.info(f'Loading checkpoint {checkpoint_name}')
        checkpoint_path = os.path.join(CHECKPOINTS_ROOT, checkpoint_name)
        # Points to a lightning checkpoint
        self.net_g = SynthesizerTrn(
            spec_channels=self.config.data.filter_length // 2 + 1,
            segment_size=self.config.data.segment_size // self.config.data.hop_length,
            hp=self.config
        )
        # These components are not necessary for inference
        del self.net_g.enc_q
        del self.net_g.speaker_classifier

        state = torch.load(
            checkpoint_path, map_location='cpu', weights_only=False)['state_dict']
        submodule_prefix = 'net_g.'
        load_submodule_prefix(self.net_g, submodule_prefix, state, quiet=True)
        self.net_g.to('cuda')
        self.net_g.eval()
        self.net_g.half()

        # def debug_forward(name):
            # def hook(module, input, output):
                # if type(input) == tuple:
                    # input = input[0]
                # if torch.isnan(input).any():
                    # print(f'NaNs detected before {name}')
                    # raise ValueError(f'NaNs in {name}')
            # return hook
        # for name, module in self.net_g.named_modules():
            # module.register_forward_hook(debug_forward(name))

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
            feats = self.my_feats.extract_features(file)
            feats = interp_row(feats)
            lens = torch.tensor([feats['whisper'].shape[1]]).to('cuda')

            # Tranpsose
            feats['f0'] = feats['f0'] * (2 ** (transpose / 12))

            # Truncate
            noise_aug = data['noise_aug']
            phone_aug = feats['whisper'] + (torch.randn_like(feats['whisper']) * noise_aug)
            feats['f0'] = feats['f0'][:phone_aug.shape[0]]

            with torch.no_grad():
                o = self.net_g.infer(
                    ppg=phone_aug.half().to(self.device).unsqueeze(0),
                    vec=feats['hubert'].half().to(self.device).unsqueeze(0),
                    pit=feats['f0'].half().to(self.device).unsqueeze(0),
                    spk=self.spk_index[str(data['sid'])].half().to(self.device).unsqueeze(0),
                    ppg_l=lens,
                    noise_scale=data['noise'])
                o_np = o.squeeze().cpu().float().numpy()
                out.append(AudioResult(
                    label=os.path.basename(file)+data['model_labels'][0],
                    audio=o_np))
        logger.info(f'Finished infering {len(files)} files')
        return InferenceResult(audios=out)

if __name__ == '__main__':
    app = QApplication([])
    config = OmegaConf.load('config/svc5_base.yaml')
    window = MainWindow(config)
    window.show()
    app.exec_()
