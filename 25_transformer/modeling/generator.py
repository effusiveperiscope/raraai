from einops import rearrange
from contextlib import contextmanager, nullcontext
import math
import torch
from modeling.vevo_repcodec import VevoRepCodec
from modeling.rvc.my_nsf import MyGeneratorNSF
from modeling.vits.models import PosteriorEncoder
from modeling.vits.commons import rand_slice_segments_with_pitch
import torch.nn as nn
import torch.nn.functional as F

# Normalization scale for pitch: log1p(f0) / PITCH_LOG_SCALE maps
# ~[0, 1000 Hz] to ~[0, 1] for stable codec training.
PITCH_LOG_SCALE = math.log1p(1000.0)

# Module-level parameter-name prefixes of the *pitch codec path*: the analysis
# conv, the pitch RVQ codec, the synthesis conv, the f0 output head and the
# voicing (V/UV) head. These are exactly the weights stage-1 ``--pitch_only``
# keeps trainable (see ``Generator.freeze_non_pitch``), so anything that needs
# to identify "the pitch codec weights" in a checkpoint (e.g.
# ``merge_pitch_codec.py``) must read this tuple instead of re-listing them.
# Note the decoder's own pitch conditioning (``dec.pit.cond.*``) is *not* part
# of this set: it belongs to the NSF decoder and stays frozen in pitch-only
# training.
PITCH_PATH_PREFIXES = (
    "pitch_downsample.",
    "pitch_codec.",
    "pitch_upsample.",
    "pitch_out_proj.",
    "pitch_uv_proj.",
)

# Module-level parameter-name prefixes of the *synthesis (decode) path*: the
# spectrogram codec decoder + its speaker projection, the resampling conv, the
# NSF decoder, the speaker table and the whole pitch decode path. This is
# exactly what `Generator.infer_from_codes` runs once it has codes -- i.e.
# everything that turns codes into audio -- and therefore exactly the trainable
# set of stage-1 `--acoustic_only` fine-tuning (see
# `Generator.freeze_code_encoders`): the code-producing half of the model
# (`enc_q`, `downsample`, both codec encoders/projectors and the quantisers) is
# frozen, so every previously extracted spec/pitch code tensor stays valid.
#
# It is an allow-list on purpose: a newly added module defaults to frozen,
# which is the safe direction for code identity. If you add a module that is
# used at inference time, add its prefix here (the freeze method prints the
# trainable parameter table, so a missing entry is visible on the first line of
# a run).
CODE_DECODE_PREFIXES = (
    "codec.decoder.",
    "codec.g_proj.",
    "upsample.",
    "dec.",
    "spk_emb.",
    "pitch_codec.decoder.",
    "pitch_codec.g_proj.",
    "pitch_upsample.",
    "pitch_out_proj.",
    "pitch_uv_proj.",
)


def _get_cfg(cfg, key, default):
    try:
        if cfg is None:
            return default
        if key in cfg:
            v = cfg[key]
            return v if v is not None else default
    except Exception:
        pass
    try:
        return getattr(cfg, key, default)
    except Exception:
        return default

class Generator(nn.Module):
    def __init__(self,
        hp,
        down_factor: int = 4):
        super().__init__()
        self.enc_q = PosteriorEncoder(
            hp.data.filter_length // 2 + 1,
            hp.vits.inter_channels,
            hp.vits.hidden_channels,
            5,
            1,
            16,
            gin_channels=hp.vits.gin_channels,
        )
        self.codec = VevoRepCodec(
            input_channels=hp.vits.inter_channels,
            output_channels=hp.vits.inter_channels,
            encode_channels=hp.vits.inter_channels,
            decode_channels=hp.vits.inter_channels,
            code_dim=hp.codec.code_dim,
            codebook_size=hp.codec.codebook_size,
            codebook_num=hp.codec.codebook_num,
            gin_channels=hp.vits.gin_channels,
            # EMA decay of the codebook update. The historical default (0.8,
            # i.e. a ~5-batch memory) was measured to rewrite 50-90 % of a
            # codebook's content every 20 steps, so the encoder was chasing a
            # moving target. 0.99 is the usual value.
            quantizer_decay=float(_get_cfg(hp.codec, "ema_decay", 0.99)),
        )
        self.dec = MyGeneratorNSF(hp=hp)
        self.spk_emb = nn.Embedding(hp.vits.max_spk_count, hp.vits.gin_channels)
        self.hp = hp
        # Set by `freeze_code_encoders()` (stage-1 `--acoustic_only`): the
        # code-producing half then also runs in eval() + no_grad, so the decoder
        # is trained on exactly the codes `2_preprocess.py` extracts (the codec
        # encoders contain dropout; extracting codes in train mode would train
        # the decoder on a distribution stage 2 never sees). Plain attribute --
        # no state_dict entry, so checkpoints are unchanged.
        self.code_frozen = False
        # Temporal resampling around the codec so that
        # code rate == decoder frame rate / down_factor (default 1/4).
        # Convs operate on (B, C, T).
        self.down_factor = down_factor
        self.downsample = nn.Conv1d(
            hp.vits.inter_channels,
            hp.vits.inter_channels,
            kernel_size=down_factor,
            stride=down_factor,
        )
        self.upsample = nn.ConvTranspose1d(
            hp.vits.inter_channels,
            hp.vits.inter_channels,
            kernel_size=down_factor,
            stride=down_factor,
        )

        # --- Low-parameter pitch codec ---
        # Same downsampling/upsampling factor as the spectrogram path so
        # that pitch codes are at the same rate as the spec codes.
        # Operates on (B, 1, T) f0 contours. Config overridable via
        # hp.pitch_codec: {hidden_channels, code_dim, codebook_size,
        # codebook_num}.
        pitch_codec_cfg = _get_cfg(hp, "pitch_codec", None)
        p_hidden = int(_get_cfg(pitch_codec_cfg, "hidden_channels", 32))
        p_code_dim = int(_get_cfg(pitch_codec_cfg, "code_dim", 32))
        p_codebook_size = int(_get_cfg(pitch_codec_cfg, "codebook_size", 32))
        p_codebook_num = int(_get_cfg(pitch_codec_cfg, "codebook_num", 2))
        # Overlapping analysis conv. With kernel_size == stride this is a plain
        # block decimation: code j sees frames 4j..4j+3 and nothing else, so it
        # is blind to its neighbours -- while the (overlapping) synthesis side
        # now spreads each code over 16 frames. Two problems with that:
        #   * the encoder cannot see the context it is contributing to, which
        #     makes neighbouring codes a non-orthogonal, coupled code;
        #   * k == s is decimation with no anti-alias filter, so pitch movement
        #     above the 12.5 Hz code-rate Nyquist folds into the block means.
        # A wider kernel (symmetric, stride still down_factor) fixes both and
        # keeps the block centre: with pad = (k - s)/2, code j covers frames
        # 4j-2..4j+5, i.e. centre 4j+1.5, exactly as the non-overlapping block
        # did, and the output length is still floor(T / down_factor).
        p_ds_kernel = int(_get_cfg(pitch_codec_cfg, "downsample_kernel",
                                   2 * down_factor))
        if p_ds_kernel < down_factor or (p_ds_kernel - down_factor) % 2:
            raise ValueError(
                "pitch_codec.downsample_kernel must be >= down_factor "
                f"({down_factor}) with the same parity, got {p_ds_kernel}")
        self.pitch_downsample_pad = (p_ds_kernel - down_factor) // 2
        self.pitch_downsample = nn.Conv1d(
            1,
            p_hidden,
            kernel_size=p_ds_kernel,
            stride=down_factor,
        )
        self.pitch_codec = VevoRepCodec(
            input_channels=p_hidden,
            output_channels=p_hidden,
            encode_channels=p_hidden,
            decode_channels=p_hidden,
            code_dim=p_code_dim,
            codebook_size=p_codebook_size,
            codebook_num=p_codebook_num,
            gin_channels=hp.vits.gin_channels,
            quantizer_decay=float(_get_cfg(pitch_codec_cfg, "ema_decay", 0.99)),
        )
        # Overlapping transposed conv back to the decoder frame rate.
        # With kernel_size == stride this layer is exactly block-diagonal: the
        # `down_factor` frames of a block are `down_factor` different linear
        # functionals of one code vector, and at every seam the path switches
        # to a different functional of the *next* code. Nothing ties
        # W[3] . h[i] to W[0] . h[i+1], so each seam is a discontinuity that
        # the codebook has to actively spend capacity avoiding -- and 12
        # bits/frame of L1-optimal VQ has no reason to. Note this is about the
        # seam, not about context: the codec decoder already has a ~17-code
        # (680 ms) receptive field, so a code already influences ~68 output
        # frames before this layer. An overlapping kernel blends
        # `kernel / stride` neighbouring codes into every frame, which makes
        # continuity structural instead of learnt. (kernel - stride) must be
        # even so that the output length stays exactly `down_factor * T`.
        p_up_kernel = int(_get_cfg(pitch_codec_cfg, "upsample_kernel",
                                   4 * down_factor))
        if p_up_kernel < down_factor or (p_up_kernel - down_factor) % 2:
            raise ValueError(
                "pitch_codec.upsample_kernel must be >= down_factor "
                f"({down_factor}) with the same parity, got {p_up_kernel}")
        self.pitch_upsample = nn.ConvTranspose1d(
            p_hidden,
            p_hidden,
            kernel_size=p_up_kernel,
            stride=down_factor,
            padding=(p_up_kernel - down_factor) // 2,
        )
        self.pitch_out_proj = nn.Conv1d(p_hidden, 1, kernel_size=1)
        # Voicing (V/UV) head. It predicts, per decoder frame, whether the
        # pitch code represents voiced speech. Trained with BCE against
        # (f0 > 0), so the pitch codes carry their own unvoiced decision and
        # synthesis no longer needs an external voicing mask.
        self.pitch_uv_proj = nn.Conv1d(p_hidden, 1, kernel_size=1)
        # Start biased towards "voiced" with near-zero weights: a checkpoint
        # that predates this head (so these weights are randomly initialized
        # by load_state_dict_mismatch) then reproduces the legacy all-voiced
        # behaviour instead of randomly dropping frames to silence.
        nn.init.normal_(self.pitch_uv_proj.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.pitch_uv_proj.bias, 4.0)
        # Decision threshold on sigmoid(uv_logit) used at inference.
        self.pitch_uv_threshold = float(_get_cfg(
            _get_cfg(hp, "train", None), "pitch_uv_threshold", 0.5))
        # Weight applied to unvoiced frames in the pitch reconstruction and
        # quantiser commitment losses (`train.c_pitch_qr_uv`, mirroring the
        # spectrogram path's `train.c_qr_uv`). Unvoiced frames are exactly zero,
        # so hard-masking them out left the codec's 64-d *decoder* output on
        # ~85 % of all frames unsupervised -- measured 0.0008 reconstruction
        # error on voiced frames vs 0.0132 on unvoiced. (`0` restores the hard
        # mask; the commitment itself is not unvoiced-dominated, so weighting it
        # is mainly what keeps the two terms referring to the same frames.)
        self.pitch_qr_uv_weight = float(_get_cfg(
            _get_cfg(hp, "train", None), "c_pitch_qr_uv", 0.5))

    def freeze_non_pitch(self):
        """Freeze every parameter except the pitch path components.

        Keeps ``pitch_downsample``, ``pitch_codec``, ``pitch_upsample``,
        ``pitch_out_proj`` and ``pitch_uv_proj`` trainable and freezes
        everything else (posterior encoder, spectrogram codec, resampling
        layers, NSF decoder and the speaker table). Used by stage-1
        ``--pitch_only`` fine-tuning so only the pitch path is updated.

        The prefix list lives in :data:`PITCH_PATH_PREFIXES` so that
        ``merge_pitch_codec.py`` merges exactly the weights this method
        unfreezes.
        """
        for name, param in self.named_parameters():
            param.requires_grad_(name.startswith(PITCH_PATH_PREFIXES))

    def freeze_code_encoders(self):
        """Freeze every parameter that *produces* RVQ codes (acoustic fine-tune).

        The trainable set is exactly :data:`CODE_DECODE_PREFIXES` -- the
        parameters ``infer_from_codes`` uses to turn codes into audio. The
        frozen complement is the whole code-producing half: ``enc_q``,
        ``downsample``, ``codec.encoder``, ``codec.projector``,
        ``codec.quantizer``, ``pitch_downsample``, ``pitch_codec.encoder``,
        ``pitch_codec.projector`` and ``pitch_codec.quantizer``.

        Two things are needed for the codes to actually stay identical:

        * ``requires_grad_(False)`` on the encoder-side parameters -- the
          obvious part;
        * ``Quantizer.freeze()`` on both codebooks -- ``requires_grad`` does not
          stop the quantiser's EMA update, which rewrites the ``embed`` buffer
          in place on every training step and would silently move the codes.

        The forward pass additionally switches the frozen half to ``eval()``
        under ``torch.no_grad()`` (see :meth:`_frozen_encode`), because the
        codec encoders contain dropout while ``encode_codes`` /
        ``2_preprocess.py`` run them in eval mode.

        The speaker embedding is *not* in the frozen set: the code path never
        consumes it (``enc_q`` is called without ``g`` and
        ``VevoRepCodec.encode_latent`` takes no ``g``), so training the decoder
        side of the table -- including new rows for a new speaker set -- cannot
        perturb a single code.

        No parameter, buffer or ``state_dict`` key changes, so ``--transfer_from``
        and ``--resume_from`` both keep working. Note that the *optimizer* is
        built from ``requires_grad`` parameters, so resuming a full-training
        checkpoint (which has state for the now-frozen tensors) fails loudly on
        the parameter-group sizes -- start a fine-tune from a new ``exp_name``
        with ``--transfer_from``.
        """
        for name, param in self.named_parameters():
            param.requires_grad_(name.startswith(CODE_DECODE_PREFIXES))
        self.codec.quantizer.freeze()
        self.pitch_codec.quantizer.freeze()
        self.code_frozen = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f'=== Frozen-code (acoustic) mode: {trainable / 1e6:.2f} M trainable '
              f'(decode path) / {frozen / 1e6:.2f} M frozen (code encoders) ===')
        for prefix in CODE_DECODE_PREFIXES:
            n = sum(p.numel() for name, p in self.named_parameters()
                    if name.startswith(prefix) and p.requires_grad)
            print(f'    train {prefix:22s} {n / 1e6:7.3f} M')

    def _code_producer_modules(self):
        """The modules that map specs/f0 to codes (frozen in acoustic mode)."""
        return (self.enc_q, self.downsample,
                self.codec.encoder, self.codec.projector, self.codec.quantizer,
                self.pitch_downsample, self.pitch_codec.encoder,
                self.pitch_codec.projector, self.pitch_codec.quantizer)

    @contextmanager
    def _frozen_encode(self):
        """``eval()`` + ``no_grad()`` over the code-producing half of the model.

        Used by the ``code_frozen`` branch of :meth:`forward`. ``eval()`` is not
        cosmetic: ``VevoRepCodec``'s ``Encoder`` has dropout, and
        ``encode_codes`` (what ``2_preprocess.py`` runs) refuses to run outside
        eval, so training the decoder on train-mode latents would target a code
        distribution stage 2 never sees. The previous ``training`` flags are
        restored afterwards, so this is safe to call from both the train and the
        validation path.
        """
        mods = self._code_producer_modules()
        prev = [m.training for m in mods]
        for m in mods:
            m.eval()
        try:
            with torch.no_grad():
                yield
        finally:
            for m, was_training in zip(mods, prev):
                m.train(was_training)

    def _pitch_downsample_conv(self, f0_target):
        """Overlapping analysis conv, replicate-padded at both ends.

        Expects a log-normalized ``(B, 1, T)`` contour whose length is already
        a multiple of ``down_factor``. Replicate padding (not zero) because a
        zero pad reads as ~0 Hz and would drag the first/last code towards
        unvoiced. Kept in one place so that training (``_pitch_codec_forward``)
        and code extraction (``encode_codes``) can never diverge -- a drift
        here would make stage-2 codes inconsistent with stage-1 training.
        """
        if self.pitch_downsample_pad > 0:
            f0_target = F.pad(
                f0_target,
                (self.pitch_downsample_pad, self.pitch_downsample_pad),
                mode='replicate')
        return self.pitch_downsample(f0_target)  # (B, H, T / down_factor)

    def _pitch_encode_inputs(self, f0, t_pad):
        """Pitch *encoder half*: ``f0 (Hz) -> (f0_target, px_ds, voiced_w_ds)``.

        Shared by stage-1 training (:meth:`_pitch_codec_forward`) and code
        extraction (:meth:`encode_codes`), so the pitch codes can never diverge
        between the two.
        """
        f0_in = f0.float().unsqueeze(1)  # (B, 1, T)
        f0_target = (torch.log1p(torch.clamp(f0_in, min=0.0))
                     / PITCH_LOG_SCALE)
        if t_pad > 0:
            f0_target = F.pad(f0_target, (0, t_pad))
        # Same down_factor as the spec path -> same code rate.
        px = self._pitch_downsample_conv(f0_target)  # (B, H, T_ds)
        px_ds = rearrange(px, "b c t -> b t c")
        # Per-frame weight for the quantiser's commitment loss, shared with
        # `pitch_q_recon_loss` in 1_training.py. Unvoiced frames are exactly
        # zero and collapse onto a couple of codes, so the codec's output
        # there is only weakly constrained; weighting them the same way in
        # both terms keeps them referring to the same frames.
        voiced_w = torch.where(
            f0_target > 0,
            torch.ones_like(f0_target),
            torch.full_like(f0_target, self.pitch_qr_uv_weight),
        )
        voiced_w_ds = F.interpolate(
            voiced_w, size=px_ds.shape[1], mode='area').squeeze(1)
        return f0_target, px_ds, voiced_w_ds

    def _pitch_codec_forward(self, f0, t_pad, g):
        """Run the low-parameter pitch codec.

        Args:
            f0: (B, T) f0 contour in Hz, 0 == unvoiced.
            t_pad: same right-pad used for the spectrogram path so that
                pitch codes land at exactly the same rate as spec codes.
            g: speaker embedding.
        Returns:
            f0_hat: (B, T_full) reconstructed log-normalized pitch at the
                decoder frame rate (trimmed of padding).
            pq_ds: (B, T_ds, H) quantized pitch codes at the code rate.
            px_ds: (B, T_ds, H) pre-quantization pitch features.
            p_vqloss, p_perplexity: VQ stats from the pitch codec.
            f0_target: (B, T_full) log-normalized pitch target.
            p_uv_logits: (B, T_full) voiced/unvoiced logits (higher ==
                voiced), decoded from the same quantized pitch codes.

        Note:
            The path runs in fp32 even when the trainer uses bf16-mixed
            precision. The contour is log-normalized to roughly [0, 1], where
            bf16 has an ulp of ~2e-3; in f0 terms that is ~24 cents, which is
            coarser than the per-frame movement the pitch codec is supposed to
            model and much coarser than the slopes a slope-based critic needs
            to see. The module is small (a 64-wide codec at a quarter of the
            frame rate), so fp32 costs practically nothing here.
        """
        with torch.autocast(device_type=f0.device.type, enabled=False):
            f0 = f0.float()
            g = g.float()
            t_full = f0.shape[-1]
            # Encode half (f0 -> pitch codes) and decode half (codes -> f0_hat)
            # are separated here for the same reason as on the spectrogram path:
            # in frozen-code (acoustic) mode the encode half runs in
            # eval()+no_grad via `encode_latent`, and the trainable decode half
            # goes through `decode_latent` -- the exact call infer_from_codes
            # makes.
            encode_ctx = self._frozen_encode() if self.code_frozen else nullcontext()
            with encode_ctx:
                f0_target, px_ds, voiced_w_ds = self._pitch_encode_inputs(f0, t_pad)
                if self.code_frozen:
                    _, pzq, p_vqloss, p_perplexity, _ = \
                        self.pitch_codec.encode_latent(px_ds, weight=voiced_w_ds)
            if self.code_frozen:
                pq = rearrange(self.pitch_codec.decode_latent(pzq, g),
                               "b c t -> b t c")
            else:
                pq, _, _, _, p_vqloss, p_perplexity = self.pitch_codec(
                    px_ds, do_uq=False, g=g, weight=voiced_w_ds)
            pq_ds = pq
            pq_ct = rearrange(pq, "b t c -> b c t")
            pq_up = self.pitch_upsample(pq_ct)
            # Force the upsampled contour onto exactly `t_full` frames. The
            # transposed conv lands on 4 * floor(T_ds), and T_ds comes from the
            # pitch path's own (downsample_kernel - down_factor) floor, so the
            # result is `t_full + t_pad` only when `t_pad` was derived from the
            # *pitch* length. `t_pad` actually comes from the spectrogram path,
            # so whenever it is non-zero the old `[..., :-t_pad]` trim returned
            # `f0_hat` t_pad frames short of `f0_target` (measured 1802 vs 1804
            # at t_pad=2). Training never hit it (segment/hop = 800 == 0 mod 4
            # => t_pad == 0), but infer(use_pitch_recon=True) on arbitrary
            # lengths did.
            if pq_up.shape[-1] > t_full:
                pq_up = pq_up[..., :t_full]
            elif pq_up.shape[-1] < t_full:
                pq_up = F.pad(pq_up, (0, t_full - pq_up.shape[-1]))
            f0_hat = self.pitch_out_proj(pq_up).squeeze(1)  # (B, T_full)
            p_uv_logits = self.pitch_uv_proj(pq_up).squeeze(1)  # (B, T_full)
            return (f0_hat, pq_ds, px_ds, p_vqloss, p_perplexity,
                    f0_target[..., :t_full].squeeze(1), p_uv_logits)

    def _spec_encode(self, spec, spec_l, with_indices=False):
        """Spectrogram code path: ``enc_q -> downsample -> codec.encode_latent``.

        Returns ``(t_frames, t_pad, x_ds, z, zq, vqloss, perplexity, indices)``
        where ``t_frames`` is the unpadded decoder frame count, ``x_ds`` is the
        ``(B, T_ds, C)`` downsampled latent (the ``q_recon_loss`` target) and
        the rest comes from ``VevoRepCodec.encode_latent``. The caller owns the
        eval()/no_grad() bookkeeping (see :meth:`_frozen_encode`); both
        :meth:`forward` and :meth:`encode_codes` call this, so the codes stage 1
        trains on are the codes stage 2 extracts.
        """
        x, _ = self.enc_q(spec, spec_l)
        t_frames = x.shape[-1]
        # Pad time so it is divisible by down_factor (pad on the right).
        t_pad = (-t_frames) % self.down_factor
        if t_pad > 0:
            x = F.pad(x, (0, t_pad))
        x_ds = rearrange(self.downsample(x), "b c t -> b t c")
        z, zq, vqloss, perplexity, indices = self.codec.encode_latent(
            x_ds, with_indices=with_indices)
        return t_frames, t_pad, x_ds, z, zq, vqloss, perplexity, indices

    @staticmethod
    def _align_f0(f0, t_frames):
        """Trim/pad an f0 contour to the (unpadded) decoder frame count."""
        if f0.shape[-1] < t_frames:
            return F.pad(f0, (0, t_frames - f0.shape[-1]))
        return f0[..., :t_frames]

    def forward(self, spec, spec_l, f0, sid):
        g = self.spk_emb(sid)
        # Code-producing half (enc_q -> downsample -> codec encoder/quantiser).
        # In frozen-code (acoustic) fine-tuning this runs in eval() + no_grad,
        # so the decoder is trained on exactly the codes 2_preprocess.py will
        # extract; a normal run keeps the graph so the quantiser/encoder learn.
        encode_ctx = self._frozen_encode() if self.code_frozen else nullcontext()
        with encode_ctx:
            (t_frames, t_pad, x_ds, _, zq,
             vqloss, perplexity, _) = self._spec_encode(spec, spec_l)

        # Decode half -- the same call infer_from_codes makes.
        yq = rearrange(self.codec.decode_latent(zq, g), "b c t -> b t c")
        yq_ds = yq

        # Upsample codec output back to decoder frame rate.
        yq = self.upsample(rearrange(yq, "b t c -> b c t"))
        if t_pad > 0:
            yq = yq[..., : yq.shape[-1] - t_pad]
        # Align f0 length with the (unpadded) encoder frame length so the
        # pitch branch sees the same T as the spec path.
        f0_aligned = self._align_f0(f0, t_frames)
        (f0_hat, pq_ds, px_ds, p_vqloss, p_perplexity,
            f0_target, p_uv_logits) = self._pitch_codec_forward(
                f0_aligned, t_pad, g)
        seg = self.hp.data.segment_size // self.hp.data.hop_length
        y_slice, f0_slice, ids_slice = rand_slice_segments_with_pitch(
            yq, f0, spec_l.clamp_min(seg), seg)
        audio = self.dec(g, y_slice, f0_slice)
        return (audio, ids_slice, yq_ds, x_ds, vqloss, perplexity,
                f0_hat, f0_target, pq_ds, px_ds, p_vqloss, p_perplexity,
                p_uv_logits)

    def infer(self, spec, spec_l, f0, sid, use_pitch_recon=False,
              predict_voicing=False, return_pitch=False):
        """Synthesise audio from a spectrogram + f0 contour.

        Args:
            spec: (B, C, T) linear spectrogram.
            spec_l: (B,) valid spectrogram lengths.
            f0: (B, T) f0 contour in Hz, 0 == unvoiced.
            sid: (B,) speaker ids.
            use_pitch_recon: replace the input f0 with the pitch codec's
                reconstruction (encode/decode through the codes) before
                synthesis.
            predict_voicing: decide voicing from the pitch codec's V/UV head
                instead of the input f0 mask (only used with
                ``use_pitch_recon``).
            return_pitch: also return the f0 contour (Hz) that was fed to
                the decoder, e.g. for visualisation.
        Returns:
            audio: (B, 1, T_audio) waveform, or ``(audio, f0_used)`` when
            ``return_pitch`` is True.
        """
        g = self.spk_emb(sid)
        # Same code path as training/encode_codes (`_spec_encode`) and the same
        # decode call as infer_from_codes (`decode_latent`); in frozen-code
        # (acoustic) mode this runs the encoder half in eval()+no_grad.
        encode_ctx = self._frozen_encode() if self.code_frozen else nullcontext()
        with encode_ctx:
            (t_frames, t_pad, _, _, zq,
             vqloss, perplexity, _) = self._spec_encode(spec, spec_l)
        yq = self.upsample(self.codec.decode_latent(zq, g))  # (B, C, T)
        if t_pad > 0:
            yq = yq[..., : yq.shape[-1] - t_pad]
        if use_pitch_recon:
            f0_aligned = self._align_f0(f0, t_frames)
            (f0_hat_norm, _, _, _, _, _,
                uv_logits) = self._pitch_codec_forward(f0_aligned, t_pad, g)
            # Denormalize log1p pitch back to Hz, keep unvoiced at 0.
            # `predict_voicing` uses the codec's own V/UV head; the default
            # keeps the legacy behaviour of trusting the input f0 mask.
            if predict_voicing:
                voiced = (torch.sigmoid(uv_logits)
                          >= self.pitch_uv_threshold).float()
            else:
                voiced = (f0_aligned > 0).float()
            f0 = (torch.expm1(torch.clamp(
                f0_hat_norm, min=0.0) * PITCH_LOG_SCALE) * voiced)
            if f0.shape[-1] < yq.shape[-1]:
                f0 = F.pad(f0, (0, yq.shape[-1] - f0.shape[-1]))
            else:
                f0 = f0[..., : yq.shape[-1]]
        audio = self.dec(g, yq, f0)
        if return_pitch:
            return audio, f0
        return audio

    @staticmethod
    def _normalize_code_indices(indices, codebook_num, name):
        """Accept int codes as (Q, B, T) or (B, T, Q); return (Q, B, T)."""
        idx = indices if torch.is_tensor(indices) else torch.tensor(indices)
        if idx.dim() != 3:
            raise ValueError(
                f"{name} must be 3D (Q, B, T) or (B, T, Q), "
                f"got shape {tuple(idx.shape)}")
        if idx.shape[0] == codebook_num:
            return idx.long()
        if idx.shape[-1] == codebook_num:
            return idx.permute(2, 0, 1).long()
        raise ValueError(
            f"{name} must have codebook dim {codebook_num} first or last, "
            f"got shape {tuple(idx.shape)}")

    @staticmethod
    def _residual_lookup(quantizer, indices_qbt):
        """Sum per-layer codebook embeddings for RVQ indices (Q, B, T)."""
        layers = quantizer.codebook.layers
        if indices_qbt.shape[0] != len(layers):
            raise ValueError(
                f"Expected {len(layers)} codebooks, got {indices_qbt.shape[0]}")
        out = None
        for q, layer in enumerate(layers):
            v = F.embedding(indices_qbt[q], layer.codebook)  # (B, T, D)
            out = v if out is None else out + v
        return out.transpose(1, 2)  # (B, D, T)

    def encode_codes(self, spec, spec_l, f0, sid):
        """Encode spec + f0 into integer codes at the shared code rate.

        Each path is the *encoder half* of ``VevoRepCodec.forward``, i.e.
        ``encoder -> projector -> nearest codebook entry``. The encoder call is
        mandatory, not optional decoration: stage 1 trains the codebooks and
        the decoder on ``projector(encoder(x))``, so collecting codes from
        ``projector(x)`` alone would hand stage 2 a latent distribution the
        decoder has never been trained on (measured max |dz| ~ 0.9 with the
        encoder skipped, i.e. a different function -- the codes are not a
        sub-sample of the trained ones, they are drawn from elsewhere).

        Run this under ``eval()``: the codec encoders contain dropout, which
        would jitter the extracted codes while stage-1 training sees the
        dropped variant.

        Returns (spec_codes, pitch_codes) as (Q, B, T_ds) int tensors.
        """
        if self.training:
            raise RuntimeError(
                "encode_codes must run under eval(): the codec encoders "
                "contain dropout, so codes extracted in train mode would not "
                "match the ones stage 1 optimised. 2_preprocess.py already "
                "calls net_g.eval(); do the same in ad-hoc scripts.")
        # `sid` is accepted for API compatibility but is never used: the code
        # path is speaker-independent (`enc_q` is called without `g` and
        # `VevoRepCodec.encode_latent` takes no speaker conditioning). That is
        # what lets acoustic fine-tuning retrain the speaker table -- including
        # new rows for a new speaker set -- without moving a single code.
        (t_frames, t_pad, _, _, _, _, _,
         spec_codes) = self._spec_encode(spec, spec_l, with_indices=True)

        f0_aligned = self._align_f0(f0, t_frames)
        _, px_ds, _ = self._pitch_encode_inputs(f0_aligned, t_pad)
        _, _, _, _, pitch_codes = self.pitch_codec.encode_latent(
            px_ds, with_indices=True)
        return spec_codes, pitch_codes

    def infer_from_codes(self, spec_codes, pitch_codes, sid,
                         out_len=None, voiced=None, uv_threshold=None,
                         return_pitch=False):
        """Synthesize audio directly from integer spec + pitch codes.

        Args:
            spec_codes: (Q_spec, B, T) or (B, T, Q_spec) int codes.
            pitch_codes: (Q_pitch, B, T) or (B, T, Q_pitch) int codes at
                the same rate/length as spec_codes.
            sid: (B,) speaker ids.
            out_len: optional frame length at the decoder rate to trim to.
            voiced: optional voicing mask, (B, T_frames) or (B, T_codes).
                If None, the pitch codec's voicing head decides which frames
                are voiced; unvoiced frames render as 0 Hz. Pass a mask to
                override the prediction.
            uv_threshold: optional override for the voicing decision
                threshold applied to ``sigmoid(uv_logits)``.
            return_pitch: also return the f0 contour (Hz) decoded from
                ``pitch_codes``, e.g. for stage-2 validation plots.
        Returns:
            audio: (B, 1, T_audio) waveform, or ``(audio, f0)`` when
            ``return_pitch`` is True. ``f0`` is ``(B, T_audio // hop_length)``
            in Hz at the audio frame rate, 0 == unvoiced.
        """
        g = self.spk_emb(sid)
        spec_idx = self._normalize_code_indices(
            spec_codes, len(self.codec.quantizer.codebook.layers),
            "spec_codes").to(g.device)
        pitch_idx = self._normalize_code_indices(
            pitch_codes, len(self.pitch_codec.quantizer.codebook.layers),
            "pitch_codes").to(g.device)
        if spec_idx.shape[1] != pitch_idx.shape[1]:
            raise ValueError(
                "spec/pitch batch mismatch: "
                f"{spec_idx.shape[1]} vs {pitch_idx.shape[1]}")
        if spec_idx.shape[2] != pitch_idx.shape[2]:
            raise ValueError(
                "spec/pitch code lengths must match (same rate): "
                f"{spec_idx.shape[2]} vs {pitch_idx.shape[2]}")
        if sid.shape[0] != spec_idx.shape[1]:
            raise ValueError(
                f"sid batch {sid.shape[0]} != codes batch {spec_idx.shape[1]}")

        # Spec path: indices -> quantized latent -> codec decoder -> upsample.
        zq = self._residual_lookup(self.codec.quantizer, spec_idx)
        y = self.codec.decode_latent(zq, g)  # (B, C, T)
        yq = self.upsample(y)  # (B, C, T * down_factor)
        t_full = yq.shape[-1]
        if out_len is not None:
            yq = yq[..., :out_len]
            t_full = yq.shape[-1]

        # Pitch path: indices -> quantized latent -> codec decoder ->
        # upsample -> 1-channel log-norm pitch -> Hz.
        pzq = self._residual_lookup(self.pitch_codec.quantizer, pitch_idx)
        py = self.pitch_codec.decode_latent(pzq, g)
        py_up = self.pitch_upsample(py)[..., :t_full]
        f0_norm = self.pitch_out_proj(py_up).squeeze(1)  # (B, T_full)
        uv_logits = self.pitch_uv_proj(py_up).squeeze(1)  # (B, T_full)

        if voiced is None:
            # No external mask: decode voicing from the pitch codes.
            thr = (self.pitch_uv_threshold if uv_threshold is None
                   else float(uv_threshold))
            v = (torch.sigmoid(uv_logits) >= thr).float()
        else:
            vt = (voiced if torch.is_tensor(voiced)
                  else torch.tensor(voiced, device=f0_norm.device))
            vt = vt.float().to(f0_norm.device)
            if vt.dim() != 2 or vt.shape[0] != f0_norm.shape[0]:
                raise ValueError(
                    "voiced must be (B, T_frames) or (B, T_codes), got "
                    f"{tuple(vt.shape)}")
            if vt.shape[1] == spec_idx.shape[2]:
                # Code-rate mask: expand by the downsampling factor.
                vt = vt.repeat_interleave(self.down_factor, dim=1)
            v = vt[..., :t_full]
            if v.shape[1] < t_full:
                v = F.pad(v, (0, t_full - v.shape[1]), value=1.0)
        f0 = (torch.expm1(torch.clamp(
            f0_norm, min=0.0) * PITCH_LOG_SCALE) * v)

        audio = self.dec(g, yq, f0)
        if return_pitch:
            return audio, f0
        return audio

    def infer_pitch(self, f0, sid, predict_voicing=False):
        """Encode/decode a raw f0 contour through the pitch codec only.

        Returns (f0_hat_hz, pq_ds, px_ds) where f0_hat_hz is at the input
        frame rate and codes are at 1/down_factor of it. With
        ``predict_voicing=True`` the codec's own V/UV head (rather than the
        input f0 mask) decides which frames are rendered.
        """
        self.eval()
        with torch.no_grad():
            g = self.spk_emb(sid)
            t_pad = (-f0.shape[-1]) % self.down_factor
            (f0_hat_norm, pq_ds, px_ds, _, _, _,
                uv_logits) = self._pitch_codec_forward(f0, t_pad, g)
            if predict_voicing:
                voiced = (torch.sigmoid(uv_logits)
                          >= self.pitch_uv_threshold).float()
            else:
                voiced = (f0 > 0).float()
            f0_hat = (torch.expm1(torch.clamp(
                f0_hat_norm, min=0.0) * PITCH_LOG_SCALE) * voiced)
        return f0_hat, pq_ds, px_ds


if __name__ == "__main__":
    from omegaconf import OmegaConf
    spec = torch.randn(1, 769, 356) # Apparently the model expects the channel to be first for spec
    sid = torch.randint(0, 5, (1,))
    spec_l = torch.tensor([356])
    f0 = torch.randn(1, 356)

    hp = OmegaConf.load("configs/base.yaml")
    decoder = Generator(hp)

    print(decoder.infer(spec, spec_l, f0, sid).shape)
