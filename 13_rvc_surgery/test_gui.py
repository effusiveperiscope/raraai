import os
from PyQt5.QtCore import pyqtRemoveInputHook
from PyQt5.QtWidgets import QApplication, QMainWindow
import librosa
import numpy as np
from modeling.v08.rvc import V08Synthesizer
from logging import getLogger
from svc_helper.gui import *
from omegaconf import OmegaConf
from features import MyFeatures
from commons import count_parameters
import torch
import soundfile as sf

logger = getLogger(__name__)
#CHECKPOINTS_ROOT = 'checkpoints/teacher/v08_test02'
CHECKPOINTS_ROOT = 'checkpoints/teacher/base10'
#CHECKPOINTS_ROOT = 'checkpoints/teacher/finetune_fs_04'

import pdb
import sys
import traceback
def custom_excepthook(exc_type, exc_value, exc_traceback):
    """
    Custom exception hook that prints the exception information
    and then drops into a pdb debugger session.
    """
    from PyQt5.QtCore import pyqtRemoveInputHook
    pyqtRemoveInputHook()
    # First, print the exception information as Python normally would.
    # We use traceback.print_exception to ensure consistent formatting.
    print("An unhandled exception occurred:")
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("\nDropping into debugger...")

    # Then, drop into the pdb debugger.
    # The post_mortem function starts the debugger at the point of the exception.
    pdb.post_mortem(exc_traceback)

# Set the custom exception hook
sys.excepthook = custom_excepthook



class MainWindow(QMainWindow):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.setWindowTitle("13 RVC TEST GUI")
        self.setGeometry(100, 100, 800, 600)

        gui = VoiceGUI()
        gui.addCheckpoint(Checkpoint(
            get_checkpoints=self.getCheckpoints, load_checkpoint=self.loadCheckpoint))
        gui.addFileInput(AudioFileInput())
        gui.addParam(IntParam(label="Feature Transpose", id='feat_transpose', min=-24, max=24, default=0))
        gui.addParam(IntParam(label="Transpose", id='transpose', min=-24, max=24, default=0))
        gui.addParam(IntParam(label="Prior Transpose", id='coarse', min=-24, max=24, default=0))
        gui.addParam(DoubleParam(label="Noise Scale", id='noise', min=0, max=3, default=0.5))
        gui.addParam(DoubleParam(label="Noise Aug Scale", id='noise_aug', min=0, max=3, default=0.0))
        gui.addParam(IntParam(label="Speaker", id='sid', min=0, max=config.model.spk_embed_dim - 1, default=0))
        # TODO - pitch smooth
        gui.addInference(Inference(
            info=InferenceInfo(sr=48000, extension='flac'),
            infer_action=self.inferAction
        ))
        self.setCentralWidget(gui.build())

        my_feats = MyFeatures(
            extract_hubert=False, extract_whisper=True, extract_vevo=False)
        self.my_feats = my_feats

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
        self.net_g = V08Synthesizer(**self.config.model, is_half=True)
        print(count_parameters(self.net_g.enc_p))

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
        feat_transpose = data['feat_transpose']
        transpose = data['transpose']
        coarse = data['coarse']
        files = data['audio_files']['files']

        if not hasattr(self, 'net_g'):
            logger.error('No checkpoint loaded')
            return InferenceResult(audios=[])

        logger.info(f'Infering {len(files)} files')

        out = []
        for file in files:
            data_16k, _ = librosa.load(file, sr=16000)
            feats = self.my_feats.get_features(data_16k, pitch_shift=feat_transpose)
            lens = torch.tensor([feats['whisp_feat'].shape[1]]).to('cuda')

            # Tranpsose
            feats['pitch_fine'] = feats['pitch_fine'] * (2 ** (transpose / 12))
            # Experiment with different coarse transpose relative to fine
            # feats['pitch'] = MyFeatures.f0_to_coarse(
                # (feats['pitch_fine'] * (2 ** (coarse / 12))).squeeze(0))

            # Truncate
            feats['pitch'] = feats['pitch'].to('cuda')[:, :feats['whisp_feat'].shape[1]]
            # feats['pitch_fine'] = feats['pitch_fine'].to('cuda')[:, :feats['whisp_feat'].shape[1]]

            noise_aug = data['noise_aug']
            #print(feats['whisp_feat'].std())
            phone_aug = feats['whisp_feat'] + (torch.randn_like(feats['whisp_feat']) * noise_aug)

            with torch.no_grad():
                o, x_mask, z_stats = self.net_g.infer(
                    phone=phone_aug.half(),
                    phone_lengths=lens, 
                    nsff0=feats['pitch_fine'].to('cuda').half(),
                    sid=torch.tensor([data['sid']]).to('cuda'),
                    noise_scale = data['noise'],
                    prior_pitch=(feats['pitch_fine'].to('cuda') * (2 ** (coarse / 12))).half(),
                    )
                o_np = o.squeeze().cpu().float().numpy()
                out.append(AudioResult(
                    label=os.path.basename(file)+data['model_labels'][0],
                    audio=o_np))
        logger.info(f'Finished infering {len(files)} files')
        return InferenceResult(audios=out)

if __name__ == '__main__':
    app = QApplication([])
    config = OmegaConf.load('configs/finetune.yaml')
    window = MainWindow(config)
    window.show()
    app.exec_()
