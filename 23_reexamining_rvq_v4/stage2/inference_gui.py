import gc
from einops import rearrange
from modeling.vits.losses import kl_trend
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
from contextlib import nullcontext
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
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QVBoxLayout
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

logger = getLogger(__name__)
CHECKPOINTS_ROOT = 'models'
CONFIG = 'configs/mlp_base.yaml' # ~! remember to update this

TEST_MODE = False
if TEST_MODE:
    from pytorch_memlab import LineProfiler
PROFILE_MEM = False

SAMPLES_PER_WHISPER_FRAME = 480  # see per_file_ctx: this is the wave/whisper-frame ratio

class LossPlotWindow(QDialog):
    """Stacked, x-aligned plot of per-frame KL/recon losses against the waveform
    they were computed from. we convert frame index ->
    wave-sample index so every subplot lines up on the same x-axis."""

    def __init__(self, wave, kl_trend_f, kl_trend_r, q_recon_loss, recon_loss, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Loss / Waveform Diagnostics")
        self.resize(1000, 800)

        wave = np.asarray(wave).squeeze()
        curves = {
            'KL (fwd)': np.asarray(kl_trend_f).squeeze(),
            'KL (rev)': np.asarray(kl_trend_r).squeeze(),
            'q_recon_loss': np.asarray(q_recon_loss).squeeze(),
            'recon_loss': np.asarray(recon_loss).squeeze(),
        }
        samples_per_loss_frame = SAMPLES_PER_WHISPER_FRAME

        fig = Figure(figsize=(10, 8))
        axes = fig.subplots(len(curves) + 1, 1, sharex=True)

        axes[0].plot(np.arange(len(wave)), wave, linewidth=0.6, color='tab:gray')
        axes[0].set_ylabel('wave')

        for ax, (name, curve) in zip(axes[1:], curves.items()):
            x = np.arange(len(curve)) * samples_per_loss_frame
            ax.plot(x, curve, linewidth=0.8, color='tab:blue')
            ax.set_ylabel(name)

        axes[-1].set_xlabel('sample (aligned to wave)')
        fig.tight_layout()

        canvas = FigureCanvas(fig)
        layout = QVBoxLayout(self)
        layout.addWidget(canvas)
        self.canvas = canvas
        self.figure = fig

class MainWindow(QMainWindow):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.config = config
        self.setWindowTitle("23 Inference GUI")
        self.setGeometry(100, 100, 800, 600)

        gui = VoiceGUI()
        gui.addCheckpoint(Checkpoint(
            get_checkpoints=self.getCheckpoints, load_checkpoint=self.loadCheckpoint))
        gui.addFileInput(AudioFileInput())
        x = AudioFileInput(id='spk_files', label="Speaker Embedding Source")
        x.file_button.setText('Speaker Embedding File')
        gui.addFileInput(x)
        self.prefill_input = AudioFileInput(id='prefill', label="Prefill")
        self.prefill_input.file_button.setText('Prefill File')
        gui.addFileInput(self.prefill_input)
        gui.addParam(IntParam(label="Transpose", id='transpose', min=-24, max=24, default=0))
        gui.addParam(DoubleParam(label="Noise Scale", id='noise', min=0, max=3, default=0.34))
        gui.addParam(DoubleParam(label="Alpha Scale", id='alpha_scale', min=0, max=1, default=0.0))
        gui.addParam(IntParam(label="Speaker", id='sid', min=0, max=20000, default=0))
        gui.addParam(BoolParam(label="Use pitch smoothing", id='use_smooth_pitch', default=True))
        gui.addParam(BoolParam(label="Send pitch extras", id='use_pitch_extras', default=True))
        gui.addParam(IntParam(label="Chunk Length (s, 0=off)", id='chunk_length', min=0, max=600, default=8))
        gui.addParam(DoubleParam(label="Chunk Overlap (s)", id='chunk_overlap', min=0, max=60, default=0.2))
        gui.addInference(Inference(
            info=InferenceInfo(sr=48000, extension='flac'),
            infer_action=self.inferAction
        ))
        if TEST_MODE:
            gui.addInference(Inference(
                info=InferenceInfo(sr=48000, extension='flac'),
                infer_action=self.testAction,
                label="Test"
            ))
        self.setCentralWidget(gui.build())

        my_feats = MyFeatures(
            # whisper = 'openai/whisper-base', # XXX
            feats_to_extract={'whisper', 'f0', 'spk', 'spec', 'wave'}, do_normalize=True)
        self.my_feats = my_feats
        self.dtype = torch.bfloat16
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.emb_file = None
        self.emb = None
        self.config = config
        self.loss_windows = []

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
        config = next(
            (
                os.path.join(root, f)
                for root, _, files in os.walk(checkpoint_path)
                for f in files
                if f.endswith('.yaml')
            ),
            None
        )
        if config is not None:
            self.config = OmegaConf.load(config)

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
            encode_channels=hp.codec.get('hidden_dim', hp.codec.code_dim),
            decode_channels=hp.codec.get('hidden_dim', hp.codec.code_dim),
            code_dim=hp.codec.get('code_dim', hp.codec.whisper_dim),
            codebook_num=1,
            codebook_size=hp.codec.codebook_size
        )

        state = torch.load(
            true_checkpoint_path, map_location='cpu', weights_only=False)['state_dict']
        self.spk_index = torch.load(spk_index, map_location='cpu')
        if len(self.spk_index) > 0:
            if type(list(self.spk_index.keys())[0]) == str:
                self.spk_index = {int(k): v for k, v in self.spk_index.items()}

        if not TEST_MODE:
            del self.codec.decoder
            del self.net_g.enc_q 
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

    def setup_infer_ctx(self, data: dict):
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
            prefill_data=None
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

        if spk_feats is None: # Fall back to index if none is provided
            sid_key = data.get('sid', 0)
            if not sid_key in self.spk_index:
                sid_key = str(sid_key)
            spk_feats = self.spk_index[sid_key].to(self.dtype).to(self.device).unsqueeze(0)

        return transpose, files, spk_feats, prefill_files, prefill_data, prefill_len_16k

    def _compute_chunk_bounds(self, num_samples, chunk_samples, overlap_samples):
        """Returns a list of (start, end) sample index tuples covering
        [0, num_samples) using overlapping windows of length chunk_samples,
        hopping by (chunk_samples - overlap_samples) each time. If
        chunk_samples is <= 0 or covers the whole signal, returns a single
        (0, num_samples) bound (i.e. chunking is effectively disabled)."""
        if num_samples <= 0:
            return [(0, 0)]
        if chunk_samples <= 0 or chunk_samples >= num_samples:
            return [(0, num_samples)]
        hop = max(chunk_samples - overlap_samples, 1)
        bounds = []
        start = 0
        while start < num_samples:
            end = min(start + chunk_samples, num_samples)
            bounds.append((start, end))
            if end >= num_samples:
                break
            start += hop
        return bounds

    def _crossfade_audio(self, acc, new_audio, overlap_samples):
        """Overlap-add stitch of `new_audio` onto the end of `acc`, linearly
        crossfading over the last `overlap_samples` of acc / first
        `overlap_samples` of new_audio. If acc is None, new_audio is
        returned as-is (first chunk)."""
        if acc is None:
            return new_audio
        if overlap_samples <= 0:
            return np.concatenate([acc, new_audio])

        overlap_samples = min(overlap_samples, len(acc), len(new_audio))
        if overlap_samples <= 0:
            return np.concatenate([acc, new_audio])

        fade_out = np.linspace(1, 0, overlap_samples, dtype=acc.dtype)
        fade_in = 1 - fade_out

        acc_head, acc_tail = acc[:-overlap_samples], acc[-overlap_samples:]
        new_head, new_rest = new_audio[:overlap_samples], new_audio[overlap_samples:]
        blended = acc_tail * fade_out + new_head * fade_in
        return np.concatenate([acc_head, blended, new_rest])

    def _extract_single_chunk(self, data, clip_source, transpose, do_post=False):
        """Runs the whisper -> codec -> f0 feature extraction pipeline on a
        single, already-prepared audio clip (a file path or an in-memory WAV
        buffer). This is the per-chunk unit of work; per_file_ctx calls this
        once per (prefill + windowed slice) chunk."""
        feats = self.my_feats.extract_features(clip_source, no_spk=True,
            whisper_chunk_len=8) # No need to extract spk here

        ppg = feats['whisper'].to(self.dtype).to(self.device).unsqueeze(0)
        ppg_interp = F.interpolate(rearrange(ppg, 'b t d -> b d t'), scale_factor=2)
        ppg_interp = rearrange(ppg_interp, 'b d t -> b t d')
        ppg_len = torch.tensor([feats['whisper'].shape[0]]).to(self.device) * 2

        with torch.no_grad():
            ppg_zq, ppg_z = self.codec.forward_encode(ppg_interp)
            ppg_zq = rearrange(ppg_zq, "b c t -> b t c")
            ppg_z = rearrange(ppg_z, "b c t -> b t c")

        # Transpose
        f0 = feats['f0'] * (2 ** (transpose / 12))

        if data.get('use_smooth_pitch'):
            f0 = torch.from_numpy(smooth_pitch(f0.cpu().numpy()))

        # Truncate
        ppg_dim = min(ppg_zq.shape[1], f0.shape[0])

        if do_post:
            spec = feats['spec'].to(self.dtype).to(self.device)
            wave = feats['wave'].to(self.dtype).to(self.device)
            ppg_dim = min(ppg_dim, spec.shape[0])
            spec = spec[:ppg_dim, :]

        ppg_zq = ppg_zq[:, :ppg_dim, :]
        ppg_z = ppg_z[:, :ppg_dim, :]
        f0 = f0[:ppg_dim]

        f0_confidence = feats['f0_confidence'].to(ppg.dtype).to(self.device)
        f0_subharmonic = feats['f0_subharmonic'].to(ppg.dtype).to(self.device)
        f0_inharmonic = feats['f0_inharmonic'].to(ppg.dtype).to(self.device)

        f0_confidence = f0_confidence[:ppg_dim].unsqueeze(0)
        f0_subharmonic = f0_subharmonic[:ppg_dim].unsqueeze(0)
        f0_inharmonic = f0_inharmonic[:ppg_dim].unsqueeze(0)

        pitch_extras = {
            'confidence': f0_confidence,
            'subharmonic': f0_subharmonic,
            'inharmonic': f0_inharmonic
        }

        ppg_mask = commons.sequence_mask(ppg_len).to(self.device)

        if TEST_MODE:
            codec_metrics = {}
            yq, y, zq, z, vqloss, perplexity = self.codec(ppg_interp)

            # yq.shape = [B, T, C]
            # ppg_interp.shape = [B, T, C]
            # yq and ppg_interp MSE
            q_recon_loss = torch.mean((yq - ppg_interp) ** 2, dim=2)
            # y and ppg_interp MSE
            recon_loss = torch.mean((y - ppg_interp) ** 2, dim=2)
            codec_metrics['q_recon_loss'] = q_recon_loss
            codec_metrics['recon_loss'] = recon_loss
        else:
            codec_metrics = {}

        if not do_post:
            return ppg_zq, ppg_z, f0, ppg_len, pitch_extras
        else:
            return ppg_zq, ppg_z, f0, ppg_len, pitch_extras, spec, wave, codec_metrics

    def per_file_ctx(self, data, file, prefill_files, prefill_data, transpose,
            do_post=False):
        """Splits `file` into (optionally overlapping) chunks along the time
        axis and runs feature extraction on each one independently. The
        (fixed) prefill audio, if any, is prepended to *every* chunk, so the
        audio actually pushed through the whole whisper -> codec -> net_g
        pipeline for any single chunk never exceeds
        chunk_length + prefill_length seconds -- e.g. a 2s prefill with a 10s
        chunk length means every pass through the pipeline sees at most 12s
        of audio, regardless of how long the input file is.

        Returns (chunks, overlap_out_samples):
          - chunks: a list of per-chunk context tuples, each in the same
            shape the old, non-chunked per_file_ctx used to return a single
            one of (ppg_zq, ppg_z, f0, ppg_len, pitch_extras[, spec, wave,
            codec_metrics] if do_post).
          - overlap_out_samples: how many samples of overlap (at the model's
            *output* sample rate) consecutive chunks share, for the caller to
            crossfade the resulting audio segments with. 0 when chunking is
            effectively disabled (a single chunk covers the whole file).
        """
        has_prefill = len(prefill_files) > 0 and prefill_data is not None
        sr = self.my_feats.expected_sample_rate

        chunk_length_s = data.get('chunk_length', 0) or 0
        chunk_overlap_s = data.get('chunk_overlap', 0) or 0

        if chunk_length_s <= 0 and not has_prefill:
            # Fast path: nothing to slice or prepend, hand the file straight
            # to feature extraction exactly like the old implementation did.
            bounds = [None]
            wav_data = None
            overlap_samples = 0
        else:
            wav_data, _ = librosa.load(file, sr=sr)
            chunk_samples = int(chunk_length_s * sr) if chunk_length_s > 0 else len(wav_data)
            overlap_samples = int(chunk_overlap_s * sr) if chunk_overlap_s > 0 else 0
            if overlap_samples >= chunk_samples:
                logger.warning('Chunk overlap >= chunk length; clamping overlap')
                overlap_samples = max(chunk_samples - 1, 0)
            bounds = self._compute_chunk_bounds(len(wav_data), chunk_samples, overlap_samples)

        overlap_out_samples = 0
        if wav_data is not None and len(bounds) > 1:
            overlap_out_samples = int(round(overlap_samples * (48000 / sr)))

        chunks = []
        for bound in bounds:
            if bound is None:
                clip_source = file
            else:
                start, end = bound
                clip = wav_data[start:end]
                if has_prefill:
                    clip = np.concatenate((prefill_data, clip))

                # Normalize
                wav_max = np.abs(clip).max() + 1e-5
                clip = clip / wav_max * 0.99

                # Write to file in memory
                clip_source = io.BytesIO()
                sf.write(clip_source, clip, samplerate=sr, format='WAV', subtype='PCM_16')
                clip_source.seek(0)

            chunks.append(self._extract_single_chunk(
                data, clip_source, transpose, do_post=do_post))

        return chunks, overlap_out_samples

    def inferAction(self, data: dict):
        transpose, files, spk_feats, prefill_files, prefill_data, \
            prefill_len_16k = self.setup_infer_ctx(data)

        out = []
        for file in files:
            filepath = file
            chunks, overlap_out_samples = self.per_file_ctx(
                data, file, prefill_files, prefill_data, transpose,
            )

            stitched = None
            for ppg_zq, ppg_z, f0, ppg_len, pitch_extras in chunks:
                with torch.no_grad():
                    if PROFILE_MEM:
                        ctx = LineProfiler(self.net_g.infer)
                    else:
                        ctx = nullcontext()

                    with ctx as prof:
                        o = self.net_g.infer(
                            ppg_zq=ppg_zq.to(self.dtype).to(self.device),
                            ppg_z=ppg_z.to(self.dtype).to(self.device),
                            pit=f0.to(self.dtype).to(self.device).unsqueeze(0),
                            spk=spk_feats,
                            ppg_l=ppg_len,
                            sid=torch.Tensor([data.get('sid', 0)]).to(self.device).long(),
                            noise_scale=data['noise'],
                            pitch_extras=pitch_extras if data['use_pitch_extras'] else None,
                            ppg_alpha=data['alpha_scale'])

                    if PROFILE_MEM:
                        prof.print_stats()
                o_np = o.squeeze().cpu().float().numpy()
                # Remove prefill (prepended to every chunk)
                o_np = o_np[int(prefill_len_16k * (48000/16000)):]
                stitched = self._crossfade_audio(stitched, o_np, overlap_out_samples)

            out.append(AudioResult(
                label=os.path.basename(filepath)+data['model_labels'][0],
                audio=stitched))
        logger.info(f'Finished infering {len(files)} files')

        return InferenceResult(audios=out)

    def testAction(self, data: dict):
        transpose, files, spk_feats, prefill_files, prefill_data, \
            prefill_len_16k = self.setup_infer_ctx(data)

        out = []
        for file in files:
            filepath = file
            chunks, overlap_out_samples = self.per_file_ctx(
                data, file, prefill_files, prefill_data, transpose, do_post=True
            )

            stitched = None
            for ppg_zq, ppg_z, f0, ppg_len, pitch_extras, spec, wave, codec_metrics in chunks:
                with torch.no_grad():
                    sid=torch.Tensor([data.get('sid', 0)]).to(self.device).long()
                    fake_audio, z_mask, \
                        (z_f, z_r, z_p, m_p, logs_p, z_q, m_q,
                        logs_q, logdet_f, logdet_r, spk_preds) = self.net_g.test(
                            ppg_zq.to(self.dtype).to(self.device),
                            ppg_z.to(self.dtype).to(self.device),
                            f0.to(self.dtype).to(self.device).unsqueeze(0),
                            rearrange(spec.to(self.dtype).to(self.device).unsqueeze(0),
                                'b t c -> b c t'),
                            spk_feats, ppg_len, spec_l=ppg_len, sid=sid,
                            pitch_extras=pitch_extras, ppg_alpha=data['alpha_scale'])

                # In this case fake_audio comes from posterior encoder
                o_np = fake_audio.squeeze().cpu().float().numpy()
                # Remove prefill (prepended to every chunk)
                o_np = o_np[int(prefill_len_16k * (48000/16000)):]
                stitched = self._crossfade_audio(stitched, o_np, overlap_out_samples)

                # KL loss
                kl_trend_f = kl_trend(z_f, logs_q, m_p, logs_p, logdet_f, z_mask) # [1, C, T]
                kl_trend_r = kl_trend(z_r, logs_p, m_q, logs_q, logdet_r, z_mask) # [1, C, T]

                kl_trend_f = torch.mean(kl_trend_f, dim=1)
                kl_trend_r = torch.mean(kl_trend_r, dim=1)
                q_recon_loss = codec_metrics['q_recon_loss']
                recon_loss = codec_metrics['recon_loss']
                loss_window = LossPlotWindow(
                    wave=wave.detach().cpu().float().numpy(),
                    kl_trend_f=kl_trend_f.detach().cpu().float().numpy(),
                    kl_trend_r=kl_trend_r.detach().cpu().float().numpy(),
                    q_recon_loss=q_recon_loss.detach().cpu().float().numpy(),
                    recon_loss=recon_loss.detach().cpu().float().numpy(),
                    parent=self,
                )
                loss_window.show()
                self.loss_windows.append(loss_window)

            out.append(AudioResult(
                label=os.path.basename(filepath)+data['model_labels'][0],
                audio=stitched))
        return InferenceResult(audios=out)



if __name__ == '__main__':
    import qdarktheme

    app = QApplication(sys.argv)
    qdarktheme.setup_theme()
    config = OmegaConf.load(CONFIG)
    window = MainWindow(config)
    window.show()
    app.exec_()