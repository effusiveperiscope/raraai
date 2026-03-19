from einops import rearrange
from modeling.intensity import IntensityModel
import ultimate_xc
import os
from PyQt5.QtCore import pyqtRemoveInputHook
from PyQt5.QtWidgets import QApplication, QMainWindow
import librosa
import numpy as np
from modeling.vits.models import SynthesizerTrn
from modeling.vits import commons
from logging import getLogger
import logging
import warnings
warnings.filterwarnings('error', category=RuntimeWarning)
warnings.simplefilter('ignore', category=UserWarning) # pyworld spams the log with messages
logging.getLogger('numba').setLevel(logging.WARNING)
from svc_helper.gui import *
from omegaconf import OmegaConf
from features import MyFeatures
from commons import load_submodule_prefix
from dataset import process_row
from svc_helper.pitch.utils import nonzero_mean, discretize_f0_log, smooth_pitch
import sys
sys.path.append('..')
from rvq.vevo_repcodec import VevoRepCodec
import torch
import soundfile as sf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
import soundfile as sf
import io

logger = getLogger(__name__)
CHECKPOINTS_ROOT = 'models'
CONFIG = 'configs/base_linux.yaml' # ~! remember to update this

class MainWindow(QMainWindow):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.config = config
        self.setWindowTitle("21 Inference GUI")
        self.setGeometry(100, 100, 800, 600)

        gui = VoiceGUI()
        gui.addCheckpoint(Checkpoint(
            get_checkpoints=self.getCheckpoints, load_checkpoint=self.loadCheckpoint))
        gui.addFileInput(AudioFileInput())
        gui.addFileInput(AudioFileInput(id='spk_files', label="Speaker Embedding Source"))
        self.prefill_input = AudioFileInput(id='prefill', label="Prefill")
        gui.addFileInput(self.prefill_input)
        gui.addParam(IntParam(label="Transpose", id='transpose', min=-24, max=24, default=0))
        gui.addParam(IntParam(label="Prior Transpose", id='coarse', min=-24, max=24, default=0))
        gui.addParam(DoubleParam(label="Noise Scale", id='noise', min=0, max=3, default=0.34))
        gui.addParam(DoubleParam(label="Perturb Scale", id='perturb_scale', min=0, max=9, default=0.0))
        gui.addParam(IntParam(label="Speaker", id='sid', min=0, max=20000, default=0))
        gui.addParam(BoolParam(label="Use pitch smoothing", id='use_smooth_pitch', default=False))
        gui.addInference(Inference(
            info=InferenceInfo(sr=48000, extension='flac'),
            infer_action=self.inferAction
        ))
        self.setCentralWidget(gui.build())

        my_feats = MyFeatures(
            feats_to_extract={'whisper', 'f0'})
        self.my_feats = my_feats
        self.dtype = torch.bfloat16
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # self.hubert_stats_path = config.data.get('hubert_stats_path', 
        #     'hubert_large_stat.npz')
        # stat = np.load(self.hubert_stats_path)
        # self.hubert_feat_norm_mean = torch.tensor(stat["mean"])
        # self.hubert_feat_norm_std = torch.tensor(stat["std"])

        self.emb_file = None
        self.emb = None
        self.config = config

    def spkEmbMemoized(self, file : str):
        if file == self.emb_file:
            return self.emb
        self.emb_file = file
        self.emb = self.my_feats.extract_speaker_features(file)
        return self.emb

    def getCheckpoints(self):
        return os.listdir(CHECKPOINTS_ROOT)

    def loadCheckpoint(self, checkpoint_name):
        logger.info(f'Loading checkpoint {checkpoint_name}')

        checkpoint_path = os.path.join(CHECKPOINTS_ROOT, checkpoint_name)
        true_checkpoint_path = next(
            (
                os.path.join(root, f)
                for root, _, files in os.walk(checkpoint_path)
                for f in files
                if f.endswith('.ckpt')
            ),
            None
        )
        spk_index = next(
            (
                os.path.join(root, f)
                for root, _, files in os.walk(checkpoint_path)
                for f in files
                if f.endswith('.pt')
            ),
            None
        )

        # Points to a lightning checkpoint
        self.net_g = SynthesizerTrn(
            spec_channels=self.config.data.filter_length // 2 + 1,
            segment_size=self.config.data.segment_size // self.config.data.hop_length,
            hp=self.config
        )
        hp = self.config
        self.codec = VevoRepCodec(
            input_channels=hp.codec.whisper_dim,
            output_channels=hp.codec.whisper_dim,
            encode_channels=hp.codec.whisper_dim,
            decode_channels=hp.codec.whisper_dim,
            code_dim=hp.codec.whisper_dim,
            codebook_num=1,
            codebook_size=hp.codec.codebook_size
        )
        del self.net_g.enc_q

        state = torch.load(
            true_checkpoint_path, map_location='cpu', weights_only=False)['state_dict']
        self.spk_index = torch.load(spk_index, map_location='cpu')
        if len(self.spk_index) > 0:
            if type(list(self.spk_index.keys())[0]) == str:
                self.spk_index = {int(k): v for k, v in self.spk_index.items()}
        load_submodule_prefix(self.net_g, 'net_g.', state, quiet=True)
        load_submodule_prefix(self.codec, 'codec.', state, quiet=True)
        self.net_g.to('cuda')
        self.net_g.eval()
        self.net_g.to(self.dtype)
        self.codec.to('cuda')
        self.codec.eval()
        self.codec.to(self.dtype)

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

        prefill_files = data['audio_files']['prefill']
        if len(prefill_files) > 0:
            if len(prefill_files) > 1:
                logger.warning('Only using first prefill file')
            prefill_data, _ = librosa.load(prefill_files[0], sr=self.my_feats.expected_sample_rate)
            prefill_len_16k = prefill_data.shape[0]
        else:
            prefill_len_16k = 0

        if len(data['audio_files']['spk_files']) > 0:
            if len(data['audio_files']['spk_files']) > 1:
                logger.warning('Only using first speaker embedding file')
            logger.info(f'Used speaker embedding from {data["audio_files"]["spk_files"][0]}')
            spk_files = data['audio_files']['spk_files']
            spk_feats = self.spkEmbMemoized(spk_files[0])
            spk_feats = spk_feats.to(self.dtype).to(self.device).unsqueeze(0)
        else:
            spk_feats = None
            self.emb_file = None

        out = []
        for file in files:
            filepath = file
            if len(prefill_files) > 0:
                # Read file
                wav_data, _ = librosa.load(file, sr=self.my_feats.expected_sample_rate)
                # Concatenate
                wav_data = np.concatenate((prefill_data, wav_data))
                # Write to file in memory
                file = io.BytesIO()
                sf.write(file, wav_data, samplerate=self.my_feats.expected_sample_rate,
                         format='WAV', subtype='PCM_16')
                file.seek(0)

            feats = self.my_feats.extract_features(file)

            ppg = feats['whisper'].to(self.dtype).to(self.device).unsqueeze(0)
            ppg_interp = F.interpolate(rearrange(ppg, 'b t d -> b d t'), scale_factor=2)
            ppg_interp = rearrange(ppg_interp, 'b d t -> b t d')
            ppg_len = torch.tensor([feats['whisper'].shape[0]]).to(self.device) * 2

            with torch.no_grad():
                _, ppg_q, _, _, _ = self.codec(ppg)
            ppg_q = F.interpolate(ppg_q, scale_factor=2)
            ppg_q = rearrange(ppg_q, 'b c t -> b t c')

            # Transpose
            f0 = feats['f0'] * (2 ** (transpose / 12))

            if data['use_smooth_pitch']:
                f0 = torch.from_numpy(smooth_pitch(f0.cpu().numpy()))

            # Truncate
            f0 = f0[:ppg_q.shape[1]]

            f0_confidence = feats['f0_confidence'].to(ppg.dtype).to(self.device)
            f0_subharmonic = feats['f0_subharmonic'].to(ppg.dtype).to(self.device)
            f0_inharmonic = feats['f0_inharmonic'].to(ppg.dtype).to(self.device)

            f0_confidence = f0_confidence[:ppg_q.shape[1]].unsqueeze(0)
            f0_subharmonic = f0_subharmonic[:ppg_q.shape[1]].unsqueeze(0)
            f0_inharmonic = f0_inharmonic[:ppg_q.shape[1]].unsqueeze(0)

            pitch_extras = {
                'confidence': f0_confidence,
                'subharmonic': f0_subharmonic,
                'inharmonic': f0_inharmonic
            }

            ppg_mask = commons.sequence_mask(ppg_len).to(self.device)

            if spk_feats is None: # Fall back to index if none is provided
                sid_key = data.get('sid', 0)
                if not sid_key in self.spk_index:
                    sid_key = str(sid_key)
                spk_feats = self.spk_index[sid_key].to(self.dtype).to(self.device).unsqueeze(0)

            with torch.no_grad():
                o = self.net_g.infer(
                    ppg_zq=ppg_q.to(self.dtype).to(self.device),
                    ppg_z=ppg_interp.to(self.dtype).to(self.device),
                    pit=f0.to(self.dtype).to(self.device).unsqueeze(0),
                    spk=spk_feats,
                    ppg_l=ppg_len,
                    sid=torch.Tensor([data.get('sid', 0)]).to(self.device).long(),
                    noise_scale=data['noise'],
                    pitch_extras=pitch_extras,
                    perturb_scale=data['perturb_scale'])
                o_np = o.squeeze().cpu().float().numpy()
                # Remove prefill
                o_np = o_np[int(prefill_len_16k * (48000/16000)):]
                out.append(AudioResult(
                    label=os.path.basename(filepath)+data['model_labels'][0],
                    audio=o_np))
        logger.info(f'Finished infering {len(files)} files')
        return InferenceResult(audios=out)

if __name__ == '__main__':
    app = QApplication([])
    config = OmegaConf.load(CONFIG)
    window = MainWindow(config)
    window.show()
    app.exec_()