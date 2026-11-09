from omegaconf import OmegaConf
import pytorch_lightning as pl
from model import IntensityModel
from torch import nn
from dataset import dataset2, WhisperContext, PedalboardContext
from commons import sequence_mask
import torch
import numpy as np
import random
import ultimate_xc

import signal
import traceback
import sys

def debug_signal_handler(sig, frame):
    print("\n=== Stack trace ===")
    traceback.print_stack(frame)
    import pdb; pdb.set_trace()
    sys.exit(0)
signal.signal(signal.SIGINT, debug_signal_handler)

FRAMES_PER_SEC = 100  # interp2 gives 10 ms output frames

class TrainModule(pl.LightningModule):
    def __init__(self, model: IntensityModel, config: OmegaConf):
        super().__init__()
        self.model = model
        self.mse = nn.MSELoss()
        self.config = config

        self.crop_min_sec = float(config.train.get('crop_min_sec', 0.5))
        self.crop_max_sec = float(config.train.get('crop_max_sec', 2.0))
        self.window_loss_weight = float(config.train.get('window_loss_weight', 0.0))
        self.window_consistency_weight = float(config.train.get('window_consistency_weight', 0.0))
        self.num_windows = int(config.train.get('num_windows', 2))
        self.val_augment = bool(config.train.get('val_augment', False))

        self.pedalboard_context = PedalboardContext(augment=True)
        self.val_pedalboard_context = PedalboardContext(augment=self.val_augment)

        whisper_model = config.get('whisper_model', None)
        self.whisper_context = WhisperContext(whisper_model) if whisper_model else WhisperContext()
        d_model = self.whisper_context.model.config.d_model
        if d_model != config.model.in_channels:
            raise ValueError(
                f"whisper feature width ({d_model}) != model.in_channels "
                f"({config.model.in_channels}); set config.model.in_channels={d_model} "
                f"or config.whisper_model to match."
            )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.config.train.lr)

    def _random_crop(self, wave: np.ndarray) -> np.ndarray:
        """Random *length* crop so the model sees 0.5..crop_max_sec windows.

        Training on a single fixed 2 s window made the model unusable on longer
        audio (it only ever saw 2 s), while at inference it is scored on
        0.5-4 s windows.
        """
        lo = int(self.crop_min_sec * 16000)
        hi = int(self.crop_max_sec * 16000)
        n = wave.shape[0]
        if n <= lo:
            return wave
        length = random.randint(lo, min(hi, n))
        start = random.randint(0, n - length)
        return wave[start:start + length]

    def _window_pooled(self, interp_feats, feature_mask):
        """Pooled prediction for K random sub-windows per item -> (B, 1).

        The windows reuse the already-computed whisper features (slices of the
        crop) rather than re-running the encoder, so this costs only K small
        forwards through the conv stack. Slightly different from a standalone
        whisper pass on the window (boundary context), but it is what makes the
        per-window scores -- the ones actually used at inference -- trained.
        """
        B, T, C = interp_feats.shape
        lengths = feature_mask.sum(dim=1).to(torch.long).tolist()
        win_min = max(8, int(self.crop_min_sec * FRAMES_PER_SEC))
        win_max = max(win_min, int(self.crop_max_sec * FRAMES_PER_SEC))

        spans = []
        for b, L in enumerate(lengths):
            for _ in range(self.num_windows):
                if L <= win_min:
                    spans.append((b, 0, L))
                else:
                    w = random.randint(win_min, min(win_max, L))
                    s = random.randint(0, L - w)
                    spans.append((b, s, s + w))
        if not spans:
            return None

        max_w = max(e - s for _, s, e in spans)
        w_feats = interp_feats.new_zeros((len(spans), max_w, C))
        w_mask = torch.zeros((len(spans), max_w), dtype=feature_mask.dtype,
                             device=feature_mask.device)
        for i, (b, s, e) in enumerate(spans):
            w_feats[i, :e - s] = interp_feats[b, s:e]
            w_mask[i, :e - s] = 1

        w_intensity, w_attn = self.model(w_feats, w_mask)
        w_pooled = (w_intensity * w_attn).sum(dim=1).squeeze(-1)      # (B*K,)
        return w_pooled.view(B, self.num_windows).mean(dim=1, keepdim=True)


    def step(self, batch, batch_idx, is_train: bool):
        wave = batch['wave']
        wave_lengths = batch['wave_length']

        context = self.pedalboard_context if is_train else self.val_pedalboard_context
        wave_np = wave.detach().cpu().numpy()
        waves_processed = []
        for i, w in enumerate(wave_np):
            w = w[:int(wave_lengths[i])]
            if is_train:
                w = self._random_crop(w)
            waves_processed.append(context.process_wave(w))

        whisper_features, feature_len = self.whisper_context.extract_features_batched(
            waves_processed)
        whisper_features = whisper_features.to(self.dtype).to(self.device)
        interp_whisper_features = self.whisper_context.interp2(whisper_features)
        feature_mask = sequence_mask(feature_len).to(torch.long).to(self.device)
        interp_whisper_features = interp_whisper_features[:, :feature_len.max(), :]
        assert interp_whisper_features.shape[1] == feature_mask.shape[1], (
            "feature length convention changed: interp2 length must equal "
            "feature_len (10 ms frames)")

        intensity = batch['intensity'].to(self.dtype).to(self.device).unsqueeze(1)
        intensity = (intensity - 1) / 8 # (1-9) -> (0-1)
        intensity = (
            intensity +
            torch.randn_like(intensity) * self.config.train.get('label_noise', 0))

        intensity_pred, attn = self.model(interp_whisper_features, feature_mask)
        pooled = (intensity_pred * attn).sum(dim=1)
        loss = self.mse(pooled, intensity)

        logs = {
            'loss': loss.detach(),
            'pred_mean': (pooled.detach() * 8 + 1).mean(),
        }

        # Multi-scale supervision: also train the same prediction on random
        # sub-windows, so the window scores used at inference are calibrated.
        # Optionally pull them towards the whole-crop prediction so that
        # "clip score == mean of window scores" holds.
        if self.window_loss_weight > 0:
            window_pooled = self._window_pooled(interp_whisper_features, feature_mask)
            if window_pooled is not None:
                window_loss = self.mse(window_pooled, intensity)
                loss = loss + self.window_loss_weight * window_loss
                logs['window_loss'] = window_loss.detach()
                if self.window_consistency_weight > 0:
                    consistency = self.mse(window_pooled, pooled.detach())
                    loss = loss + self.window_consistency_weight * consistency
                    logs['window_consistency'] = consistency.detach()

        logs['loss_total'] = loss.detach()
        prefix = 'train' if is_train else 'val'
        for name, value in logs.items():
            self.log(f'{prefix}_{name}', value, prog_bar=(name == 'loss'),
                     batch_size=wave.shape[0])
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, is_train=True)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, is_train=False)

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/base.yaml")
    parser.add_argument("--resume_from", type=str)
    parser.add_argument("--max_epochs", type=int, default=-1,
                        help="stop after this many epochs (-1 = run forever)")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    model = IntensityModel(**config.model)
    train_module = TrainModule(model, config)

    train_dataset = dataset2(config.data.train_filelist, is_train=True,
                             max_crop_sec=config.train.get('crop_max_sec', 2.0))
    val_dataset = dataset2(config.data.val_filelist, is_train=False,
                           max_crop_sec=config.data.get('val_crop_sec', 30.0))
    train_dataloader = train_dataset.loader(
        batch_size=config.train.batch_size, num_workers=config.train.num_workers,
        persistent_workers=config.train.num_workers > 0)
    val_dataloader = val_dataset.loader(
        batch_size=config.train.batch_size, num_workers=config.train.num_workers,
        persistent_workers=config.train.num_workers > 0)

    val_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'checkpoints/{config.exp_name}',
        filename='best-checkpoint',
        save_top_k=config.train.keep_ckpts,
        mode='min',
        save_last=True
    )
    callbacks = [val_checkpoint_callback]
    logger = pl.loggers.TensorBoardLogger(
        config.get('log_dir', 'logs'), name=config.exp_name,
        version=0
    )

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.max_epochs,
        accelerator='gpu',
        precision='bf16-mixed',
        callbacks=callbacks)

    trainer.fit(train_module, train_dataloader, val_dataloader, ckpt_path=args.resume_from)

if __name__ == '__main__':
    main()