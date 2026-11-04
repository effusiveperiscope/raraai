"""Stage-2 transformer over discrete codec codes.

This module owns the whole sequence model *and* the training-specific
computations for it:

* ``PhonemeEncoder`` consumes the phoneme codes and speaker id produced by
  ``2_preprocess`` (the ``phone_ids`` and ``sid`` fields) and produces a
  memory the decoder can cross-attend to.
* ``DelayCodebookDecoder`` is a single causal transformer that predicts
  *all* codebooks (spectrogram RVQ codes + pitch RVQ codes) at once using a
  MusicGen-style **delay pattern**: codebook ``k`` is shifted ``k`` frames to
  the right so that every codebook can be trained with the exact same
  next-token objective while still conditioning on the already-decoded
  frames of the other codebooks.

Speaker conditioning is applied on **both** sides of the model.  The encoder
adds a speaker embedding to the phoneme tokens (so the memory is
speaker-specific), and — because cross-attention alone is a narrow, easily
averaged-away channel — the decoder has its own speaker embedding table that
is added to every decoder position.  Without the decoder-side embedding the
only path for ``sid`` is "constant vector -> 6 pre-LN encoder layers ->
cross-attention keys/values", which in practice produces speaker bleed
(same intonation/prosody for every voice) even when the encoder sees a
perfectly different ``sid``.  See ``transformer.decoder_speaker_cond`` in
``configs/transformer.yaml``.
* ``CodeTransformer`` glues them together and provides ``build_delay_pattern``,
  ``compute_batch_loss`` and ``generate``.  ``2_training.py`` therefore only
  needs to own the optimisation loop and logging.

The delay pattern implemented here is the training-time analogue of the
inference-time pattern used by MusicGen.  Given target codes ``X`` of shape
``(B, K, T)`` we build the delayed matrix ``D`` of shape ``(B, K, T + K - 1)``::

    D[b, k, k + t] = X[b, k, t]

A special start column (the per-codebook pad/BOS embedding, exactly like
MusicGen where ``bos_token_id == pad_token_id``) is prepended and the usual
``inputs = seq[..., :-1]`` / ``targets = seq[..., 1:]`` next-token split is
applied, so codebook ``k`` only ever needs the ``T + K - 1`` column context.

End of utterance (``transformer.eos_token``)
--------------------------------------------
Without an EOS class the model can only ever be asked for a fixed number of
frames: ``generate`` runs ``max_frames`` steps and *nothing inside the model*
decides when the sentence is over (the only non-code class is the pad/BOS
index, which is masked to ``-inf`` at sampling and is the cross-entropy
``ignore_index``, so it cannot double as a stop token).

With ``eos_token: true`` every codebook gets one extra output class at index
``vocab_size + 1`` (pad/BOS stays at ``vocab_size``) and the targets gain a
trailing EOS frame, so the decoder learns where the utterance ends.  At
inference ``generate`` stops at the first EOS emitted by codebook 0 -- it leads
the delay pattern, so its column index *is* its frame index -- keeps decoding
the ``K - 1`` tail columns the other codebooks still need, and returns the
codes trimmed to the predicted length.

The EOS target is only added for clips that actually reach their end:
``dataset_codes`` reports a per-item ``complete`` flag, because random training
crops are usually mid-clip and an EOS there would teach the model to fall silent
after ``max_frames``.  Enabling the key changes the parameter shapes (see
``expand_state_dict_with_eos``), so it needs a fresh run or a ``--transfer_from``
warm start -- a ``--resume_from`` of an EOS-less checkpoint cannot work.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_cfg(cfg, key, default):
    """Read ``key`` from an OmegaConf/dict/namespace cfg, falling back."""
    if cfg is None:
        return default
    try:
        if key in cfg:
            v = cfg[key]
            if v is not None:
                return v
    except Exception:
        pass
    try:
        v = getattr(cfg, key, None)
        return default if v is None else v
    except Exception:
        return default


class SinusoidalPositionalEncoding(nn.Module):
    """Classic fixed sinusoidal positional encoding (added to inputs)."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        return x + self.pe[:, : x.size(1)].to(dtype=x.dtype)


class PhonemeEncoder(nn.Module):
    """Encodes phoneme tokens + a speaker id into a cross-attention memory.

    The memory must stay *discriminative across phoneme positions*: the decoder
    reaches the text only through cross-attention, so if two positions produce
    near-identical keys the attention degenerates into a uniform average and
    the model becomes a text-blind acoustic LM (silently: the loss keeps
    falling while `phoneme -> code` conditioning is gone).  Use
    ``test_stage2_conditioning.py`` to check.

    The problem is the residual stream: pre-LN transformers do not bound it,
    and on this model it grows ~20 -> ~4000 over six layers.  With the stock
    final ``LayerNorm`` that is fatal, because the LayerNorm divides by the
    per-position std *across feature dimensions*: one runaway dimension sets
    that denominator, every informative dimension is crushed to ~1% of its size
    and the cross-attention keys become almost identical (measured attention
    entropy / log N ~= 0.99, i.e. uniform).

    ``memory_norm`` selects how the memory is normalised:

    * ``"temporal"`` (default since v0.4.1) — standardise every dimension over
      the *time* axis (masked instance norm + learnable per-dimension affine).
      Each dimension is divided by its own spread, so a runaway dimension can
      no longer suppress the others; directions that are constant over time
      (they carry no positional information) collapse into the bias.
    * ``"layer"`` — the historical single ``LayerNorm`` over the feature axis.
      Kept so pre-v0.4.1 checkpoints can still be loaded/evaluated (they were
      trained under the collapsed regime).
    * ``"none"`` — return the raw residual stream.

    ``residual_scale`` additionally scales the initial weights of the sublayer
    output projections (``self_attn.out_proj``, ``linear2``) by
    ``1/sqrt(2 * n_layers)`` (GPT-2 style), which stops the residual stream
    from growing in the first place.
    """

    def __init__(
        self,
        phone_vocab_size: int,
        n_speakers: int,
        hidden_size: int,
        n_layers: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        max_positions: int,
        memory_norm: str = "temporal",
        residual_scale: bool = True,
    ):
        super().__init__()
        if memory_norm not in ("temporal", "layer", "none"):
            raise ValueError(
                "transformer.memory_norm must be 'temporal', 'layer' or "
                f"'none', got {memory_norm!r}")
        self.memory_norm = str(memory_norm)
        self.memory_norm_eps = 1e-5

        self.phone_emb = nn.Embedding(phone_vocab_size, hidden_size)
        self.spk_emb = nn.Embedding(n_speakers, hidden_size)
        self.pos = SinusoidalPositionalEncoding(hidden_size, max_positions)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, n_layers,
            # With "temporal" the final LayerNorm is counter-productive (see
            # the class docstring); the per-dimension norm below replaces it.
            norm=nn.LayerNorm(hidden_size) if self.memory_norm == "layer" else None,
        )

        # Learnable affine of the per-dimension (temporal) normalisation.
        if self.memory_norm == "temporal":
            self.memory_gain = nn.Parameter(torch.ones(hidden_size))
            self.memory_bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter("memory_gain", None)
            self.register_parameter("memory_bias", None)

        if residual_scale and n_layers > 0:
            scale = 1.0 / math.sqrt(2.0 * n_layers)
            for enc_layer in self.encoder.layers:
                enc_layer.self_attn.out_proj.weight.data.mul_(scale)
                enc_layer.linear2.weight.data.mul_(scale)

    def _temporal_norm(
        self, x: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """Masked per-dimension standardisation of ``(B, N, H)`` over `N`."""
        mask = valid.unsqueeze(-1).to(x.dtype)
        count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (x * mask).sum(dim=1, keepdim=True) / count
        var = (((x - mean) ** 2) * mask).sum(dim=1, keepdim=True) / count
        x = (x - mean) / torch.sqrt(var + self.memory_norm_eps)
        x = x * self.memory_gain + self.memory_bias
        return x * mask

    def forward(
        self,
        phone_ids: torch.Tensor,  # (B, N) long
        phone_lengths: Optional[torch.Tensor],  # (B,) long
        sid: torch.Tensor,  # (B,) long
    ):
        B, N = phone_ids.shape
        if phone_lengths is None:
            phone_lengths = torch.full(
                (B,), N, device=phone_ids.device, dtype=torch.long
            )
        phone_lengths = phone_lengths.to(phone_ids.device).clamp(min=1, max=N)
        valid = torch.arange(N, device=phone_ids.device)[None, :] < phone_lengths[:, None]

        x = self.phone_emb(phone_ids)
        x = x + self.spk_emb(sid.to(phone_ids.device)).unsqueeze(1)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=~valid)
        x = x * valid.unsqueeze(-1).to(x.dtype)
        if self.memory_norm == "temporal":
            x = self._temporal_norm(x, valid)
        return x, valid


class DelayCodebookDecoder(nn.Module):
    """Causal transformer decoder that predicts every codebook per column.

    Each codebook has its own embedding table and output head because the
    spectrogram codebooks and pitch codebooks have different vocabulary
    sizes.  Every codebook gets one extra "pad/start" index equal to its
    vocabulary size (mirroring MusicGen's ``bos == pad`` token) and -- with
    ``eos=True``, i.e. ``transformer.eos_token`` -- a second extra class at
    ``vocab_size + 1`` for end-of-utterance.

    ``residual_scale`` scales the initial weights of the three residual output
    projections (``self_attn.out_proj``, ``multihead_attn.out_proj``,
    ``linear2``) by ``1/sqrt(3 * n_layers)``.  The decoder's residual stream
    otherwise grows by >30x over twelve layers, which shrinks the
    cross-attention contribution to a few percent of the residual and makes it
    even easier for the model to ignore the phoneme memory.
    """

    def __init__(
        self,
        vocab_sizes: Sequence[int],
        hidden_size: int,
        n_layers: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        max_positions: int,
        n_speakers: int = 0,
        speaker_cond: bool = True,
        residual_scale: bool = True,
        eos: bool = False,
    ):
        super().__init__()
        self.vocab_sizes = list(vocab_sizes)
        self.num_codebooks = len(self.vocab_sizes)
        self.hidden_size = hidden_size
        self.speaker_cond = bool(speaker_cond)
        # ``vocab_size`` is the pad/BOS class; ``vocab_size + 1`` is EOS and
        # only exists when the model was trained with ``transformer.eos_token``.
        self.eos = bool(eos)
        self.num_classes = [v + 1 + int(self.eos) for v in self.vocab_sizes]

        # Decoder-side speaker conditioning.  The encoder already puts `sid`
        # into the memory, but relying on cross-attention alone is fragile:
        # the decoder must recover a constant vector from the keys/values at
        # every step.  A dedicated embedding added to each decoder position
        # makes the voice a first-class input of the autoregressive model.
        if self.speaker_cond:
            if n_speakers <= 0:
                raise ValueError(
                    "speaker_cond=True requires n_speakers > 0, got "
                    f"{n_speakers}")
            self.spk_emb = nn.Embedding(n_speakers, hidden_size)
        else:
            self.spk_emb = None

        self.embeds = nn.ModuleList(
            [nn.Embedding(c, hidden_size) for c in self.num_classes]
        )
        self.pos = SinusoidalPositionalEncoding(hidden_size, max_positions)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, n_layers, norm=nn.LayerNorm(hidden_size))
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_size, c) for c in self.num_classes]
        )

        if residual_scale and n_layers > 0:
            scale = 1.0 / math.sqrt(3.0 * n_layers)
            for dec_layer in self.decoder.layers:
                dec_layer.self_attn.out_proj.weight.data.mul_(scale)
                dec_layer.multihead_attn.out_proj.weight.data.mul_(scale)
                dec_layer.linear2.weight.data.mul_(scale)

        self._causal_mask_cache = None
        self._causal_mask_key = None

    def _causal_mask(self, length: int, device, dtype) -> torch.Tensor:
        key = (length, device, dtype)
        if self._causal_mask_key != key:
            mask = torch.full((length, length), float("-inf"), device=device, dtype=dtype)
            mask = torch.triu(mask, diagonal=1)
            self._causal_mask_cache = mask
            self._causal_mask_key = key
        return self._causal_mask_cache

    def embed(self, codes: torch.Tensor) -> torch.Tensor:
        """Sum the per-codebook embeddings of ``codes`` (B, K, L)."""
        x = None
        for k, emb in enumerate(self.embeds):
            e = emb(codes[:, k])
            x = e if x is None else x + e
        return x

    def forward(
        self,
        codes: torch.Tensor,  # (B, K, L) long, pad index == vocab size
        memory: torch.Tensor,  # (B, N, H)
        memory_mask: Optional[torch.Tensor] = None,  # (B, N) bool, True == valid
        sid: Optional[torch.Tensor] = None,  # (B,) long
    ) -> List[torch.Tensor]:
        B, K, L = codes.shape
        x = self.embed(codes)
        x = self.pos(x)
        if self.spk_emb is not None:
            if sid is None:
                raise ValueError(
                    "DelayCodebookDecoder was built with speaker_cond=True "
                    "but no sid was passed")
            x = x + self.spk_emb(sid.to(codes.device)).unsqueeze(1)

        tgt_mask = self._causal_mask(L, x.device, x.dtype)
        mem_pad = None if memory_mask is None else ~memory_mask
        out = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=mem_pad,
        )
        return [head(out) for head in self.heads]


class CodeTransformer(nn.Module):
    """Phoneme + speaker conditioned transformer over spec/pitch codebooks."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        tcfg = _get_cfg(config, "transformer", config)

        hidden_size = int(_get_cfg(tcfg, "hidden_size", 768))
        n_heads = int(_get_cfg(tcfg, "n_heads", 12))
        enc_layers = int(_get_cfg(tcfg, "encoder_layers", 4))
        dec_layers = int(_get_cfg(tcfg, "decoder_layers", 10))
        ffn = int(_get_cfg(tcfg, "dim_feedforward", 2048))
        dropout = float(_get_cfg(tcfg, "dropout", 0.1))
        max_positions = int(_get_cfg(tcfg, "max_positions", 4096))
        phone_vocab = int(_get_cfg(tcfg, "phone_vocab_size", 1024))
        n_speakers = int(_get_cfg(tcfg, "n_speakers", 512))
        decoder_speaker_cond = bool(
            _get_cfg(tcfg, "decoder_speaker_cond", True))
        # End-of-utterance class (see the module docstring).  Off by default:
        # it changes the embedding/head shapes, so a checkpoint trained without
        # it cannot be resumed into a model built with it.
        eos_token = bool(_get_cfg(tcfg, "eos_token", False))
        # Encoder -> decoder conditioning (v0.4.1).  See PhonemeEncoder.
        memory_norm = str(_get_cfg(tcfg, "memory_norm", "temporal"))
        residual_scale = bool(_get_cfg(tcfg, "residual_scale", True))

        codec = _get_cfg(config, "codec", None)
        pitch_codec = _get_cfg(config, "pitch_codec", None)
        spec_codebook_num = int(_get_cfg(codec, "codebook_num", 8))
        spec_codebook_size = int(_get_cfg(codec, "codebook_size", 64))
        pitch_codebook_num = int(_get_cfg(pitch_codec, "codebook_num", 2))
        pitch_codebook_size = int(_get_cfg(pitch_codec, "codebook_size", 32))

        self.spec_codebook_num = spec_codebook_num
        self.pitch_codebook_num = pitch_codebook_num
        self.codebook_num = spec_codebook_num + pitch_codebook_num
        self.vocab_sizes = [spec_codebook_size] * spec_codebook_num + [
            pitch_codebook_size
        ] * pitch_codebook_num
        # Per-codebook pad/start index (== vocabulary size, cf. MusicGen bos==pad).
        self.pad_ids = list(self.vocab_sizes)
        # Per-codebook end-of-utterance index (== vocabulary size + 1), or None
        # when this model was built without EOS.  See the module docstring.
        self.eos_token = eos_token
        self.eos_ids = ([v + 1 for v in self.vocab_sizes] if eos_token else None)
        self.hidden_size = hidden_size
        self.n_speakers = n_speakers
        self.decoder_speaker_cond = decoder_speaker_cond

        # Coarse-to-fine loss weighting (see ``loss_weights``).  Stored as
        # plain attributes rather than buffers so they stay out of
        # ``state_dict`` and existing checkpoints keep loading unchanged.
        self.codebook_weights = self._parse_codebook_weights(
            _get_cfg(tcfg, "codebook_weights", None))
        self.codebook_weight_warmup_epochs = float(
            _get_cfg(tcfg, "codebook_weight_warmup_epochs", 0) or 0)

        self.encoder = PhonemeEncoder(
            phone_vocab_size=phone_vocab,
            n_speakers=n_speakers,
            hidden_size=hidden_size,
            n_layers=enc_layers,
            n_heads=n_heads,
            dim_feedforward=ffn,
            dropout=dropout,
            max_positions=max_positions,
            memory_norm=memory_norm,
            residual_scale=residual_scale,
        )
        self.decoder = DelayCodebookDecoder(
            vocab_sizes=self.vocab_sizes,
            hidden_size=hidden_size,
            n_layers=dec_layers,
            n_heads=n_heads,
            dim_feedforward=ffn,
            dropout=dropout,
            max_positions=max_positions,
            n_speakers=n_speakers,
            speaker_cond=decoder_speaker_cond,
            residual_scale=residual_scale,
            eos=eos_token,
        )
        self.memory_norm = memory_norm
        self.residual_scale = residual_scale

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, config) -> "CodeTransformer":
        return cls(config)

    def _parse_codebook_weights(self, raw) -> List[float]:
        """Validate ``transformer.codebook_weights`` into a ``(K,)`` list.

        ``None`` (key absent) means uniform weighting, i.e. the historical
        ``losses.mean()`` behaviour.
        """
        if raw is None:
            return [1.0] * self.codebook_num
        weights = [float(w) for w in raw]
        if len(weights) != self.codebook_num:
            raise ValueError(
                f"transformer.codebook_weights has {len(weights)} entries but "
                f"there are {self.codebook_num} codebooks "
                f"({self.spec_codebook_num} spec + {self.pitch_codebook_num} "
                f"pitch)")
        if any(w < 0 for w in weights):
            raise ValueError(
                f"transformer.codebook_weights must be >= 0, got {weights}")
        if sum(weights) <= 0:
            raise ValueError(
                "transformer.codebook_weights must not sum to 0")
        return weights

    def loss_weights(self, epoch: Optional[int] = None) -> torch.Tensor:
        """Effective ``(K,)`` loss weights for training epoch ``epoch``.

        ``loss = sum_k w_k * CE_k / sum_k w_k``, so only the *relative* weights
        matter.  With ``transformer.codebook_weight_warmup_epochs = N > 0`` the
        weights are linearly interpolated from uniform at epoch 0 to the
        configured hierarchy at epoch ``N`` — a soft curriculum: the coarse
        codebooks are not swamped by the fine-detail ones before the coarse
        structure exists.  ``epoch=None`` or ``N <= 0`` returns the configured
        weights unchanged.

        The tensor is float32 on CPU; ``code_loss`` moves it to the loss
        device/dtype itself.
        """
        target = torch.tensor(self.codebook_weights, dtype=torch.float32)
        warm = self.codebook_weight_warmup_epochs
        if epoch is None or warm <= 0:
            return target
        progress = min(max(float(epoch) / warm, 0.0), 1.0)
        return (1.0 - progress) + progress * target

    def _pad_column(self, batch_size: int, device, dtype=torch.long) -> torch.Tensor:
        """(B, K, 1) column of per-codebook pad/start ids."""
        col = torch.tensor(self.pad_ids, device=device, dtype=dtype).view(1, -1, 1)
        return col.expand(batch_size, -1, 1).contiguous()

    def _append_eos(
        self,
        codes: torch.Tensor,  # (B, K, T) long
        code_lengths: Optional[torch.Tensor] = None,  # (B,) long
        complete: Optional[torch.Tensor] = None,  # (B,) bool
    ):
        """Put one EOS frame at the end of each *complete* sample's streams.

        The EOS lands at frame ``code_lengths[b]`` (i.e. directly after the last
        real frame) for every codebook -- the delay pattern takes care of
        spreading it to codebook ``k``'s column ``n + k``.  Samples whose
        segment does not reach the end of the clip (``complete=False``, the
        usual case for random training crops) get no EOS: their code stream
        simply stops, which is exactly what the model should learn there.

        Everything stays vectorised on-device (no ``.item()``/``if tensor``), so
        this adds no GPU->CPU sync to the training step.

        Args:
            codes: ``(B, K, T)`` real codes (padding value irrelevant).
            code_lengths: ``(B,)`` valid frames per sample; ``None`` means all
                ``T`` frames are real.
            complete: ``(B,)`` bool, ``True`` where the segment ends the clip.

        Returns:
            ``(codes_with_eos, new_code_lengths)``: ``(B, K, T_out)`` and
            ``(B,)``, where ``T_out = max(code_lengths + complete)`` -- at most
            ``T + 1``, and never ``T + 1`` when nothing in the batch is
            complete, so no pure-padding column is kept.
        """
        B, K, T = codes.shape
        device = codes.device
        if code_lengths is None:
            code_lengths = torch.full((B,), T, device=device, dtype=torch.long)
        code_lengths = code_lengths.to(device).long().clamp(min=1, max=T)
        if complete is None:
            complete = torch.ones(B, dtype=torch.bool, device=device)
        else:
            complete = complete.to(device).bool().reshape(-1)

        lens = code_lengths + complete.long()  # +1 frame where EOS is added
        T_out = int(lens.max())  # no pure-padding column for short batches
        out = codes.new_zeros((B, K, T_out))
        keep = min(T, T_out)
        out[:, :, :keep] = codes[:, :, :keep]

        # out[b, k, code_lengths[b]] = eos_ids[k] for complete samples, and left
        # untouched for truncated ones (gather + where, so no data-dependent
        # control flow).  The gather index is redirected to 0 for the truncated
        # rows: their write is a no-op either way, but their length *can* equal
        # ``T_out`` (a full ``max_frames`` crop that is not the clip's end, in a
        # batch whose longest complete stream is one frame shorter), which would
        # otherwise index one past the last column.
        src = torch.where(complete, code_lengths, torch.zeros_like(code_lengths))
        rows = torch.arange(B, device=device)[:, None]
        cols = torch.arange(K, device=device)[None, :]
        idx = src[:, None].expand(B, K)
        eos = torch.tensor(self.eos_ids, device=device,
                           dtype=codes.dtype).view(1, K).expand(B, K)
        out[rows, cols, idx] = torch.where(complete[:, None],
                                           eos, out[rows, cols, idx])
        return out, lens

    # ------------------------------------------------------------------ #
    # Core computation
    # ------------------------------------------------------------------ #
    def encode(
        self,
        phone_ids: torch.Tensor,
        phone_lengths: Optional[torch.Tensor],
        sid: torch.Tensor,
    ):
        return self.encoder(phone_ids, phone_lengths, sid)

    def build_delay_pattern(
        self,
        codes: torch.Tensor,  # (B, K, T) long
        code_lengths: Optional[torch.Tensor] = None,  # (B,) long
    ) -> torch.Tensor:
        """Build the (B, K, T + K - 1) delayed code matrix.

        Positions outside each sample's valid window (either the delay ramp
        or trailing padding) are filled with the per-codebook pad id so they
        can be ignored by the cross-entropy.
        """
        B, K, T = codes.shape
        L = T + K - 1
        device = codes.device
        if code_lengths is None:
            code_lengths = torch.full((B,), T, device=device, dtype=torch.long)
        code_lengths = code_lengths.to(device).clamp(min=1, max=T)
        delayed = codes.new_full((B, K, L), 0)
        for k in range(K):
            delayed[:, k, k : k + T] = codes[:, k]

        col = torch.arange(L, device=device)
        row = torch.arange(K, device=device)
        frame = col[None, :] - row[:, None]  # (K, L) source frame index
        valid = (frame[None] >= 0) & (frame[None] < code_lengths[:, None, None])
        pad = torch.tensor(self.pad_ids, device=device, dtype=codes.dtype).view(1, K, 1)
        delayed = torch.where(valid, delayed, pad.expand(B, K, L))
        return delayed

    def code_loss(
        self,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor],
        codes: torch.Tensor,  # (B, K, T)
        code_lengths: Optional[torch.Tensor] = None,
        sid: Optional[torch.Tensor] = None,  # (B,) long
        epoch: Optional[int] = None,
        complete: Optional[torch.Tensor] = None,  # (B,) bool, for EOS
    ) -> dict:
        """Delayed next-token loss over all codebooks.

        With ``transformer.eos_token`` enabled, an EOS frame is appended to each
        sample's streams *before* the delay pattern is built, so "emit EOS" is
        just another next-token target and the delayed alignment is unchanged
        (see ``_append_eos``).  Only samples flagged ``complete`` get one.

        Per-codebook cross-entropies are combined as
        ``loss = sum_k w_k * CE_k / sum_k w_k`` with ``w = loss_weights(epoch)``
        (uniform unless ``transformer.codebook_weights`` is set), so the
        coarse codebooks can be prioritised over the fine residual ones.

        Returns a dict with the total loss plus per-group (spec/pitch) losses
        and diagnostics that make "is it learning?" answerable from
        TensorBoard alone:

        ``loss``/``spec_loss``/``pitch_loss``
            weighted mean cross-entropy over all / spec / pitch codebooks
            (weights renormalised inside the group, so the two group losses
            still average back to ``loss``).
        ``acc``/``spec_acc``/``pitch_acc``
            teacher-forced top-1 accuracy over non-pad target positions.
        ``code_losses`` (K,), ``code_accs`` (K,), ``code_usage_entropy`` (K,)
            the same per codebook, always *unweighted* so the raw per-book
            behaviour stays visible.  ``code_usage_entropy`` is the entropy of
            the batch-averaged predicted distribution, normalised to
            ``[0, 1]`` by ``log(vocab - 1)``; values near 0 mean the head has
            collapsed onto a handful of codes even if the loss still looks
            healthy.
        ``loss_weights`` (K,)
            the effective weights used for this batch, for logging.
        """
        B, K, T = codes.shape
        if self.eos_token:
            codes, code_lengths = self._append_eos(codes, code_lengths, complete)
        delayed = self.build_delay_pattern(codes, code_lengths)  # (B, K, L)
        L = delayed.shape[2]
        start = self._pad_column(B, codes.device)

        seq = torch.cat([start, delayed], dim=2)  # (B, K, L + 1)
        inputs = seq[..., :L]
        targets = seq[..., 1:]

        # list of (B, L, v + 1) -- or (B, L, v + 2) with the EOS class
        logits = self.decoder(inputs, memory, memory_mask, sid=sid)

        losses, accs, entropies = [], [], []
        for k in range(K):
            # fp32 for both CE and the diagnostics: under bf16-mixed the
            # logits come back bf16, which is fine for CE but too coarse for
            # a mean over ~1e5 positions.
            lg = logits[k].float()
            flat_logits = lg.reshape(-1, lg.size(-1))
            flat_targets = targets[:, k].reshape(-1)
            losses.append(F.cross_entropy(
                flat_logits, flat_targets, ignore_index=self.pad_ids[k]))

            valid = flat_targets != self.pad_ids[k]  # (B*L,)
            if valid.any():
                accs.append(
                    (flat_logits.argmax(-1)[valid] == flat_targets[valid])
                    .float().mean())
                probs = F.softmax(flat_logits[valid], dim=-1).mean(0)
                ent = -(probs.clamp_min(1e-9).log() * probs).sum()
                # `probs.size(-1)` includes the pad class, which is never a
                # target, so the attainable maximum is log(vocab - 1).  With EOS
                # the end-of-utterance class is included in the mass (it is a
                # real, if rare, target), so the normaliser stays as is.
                entropies.append(
                    ent / math.log(max(probs.size(-1) - 1, 2)))
            else:  # degenerate batch (everything masked)
                accs.append(lg.new_zeros(()))
                entropies.append(lg.new_zeros(()))

        losses = torch.stack(losses)  # (K,)
        accs = torch.stack(accs)  # (K,)
        entropies = torch.stack(entropies)  # (K,)

        # Coarse-to-fine combination.  ``code_losses`` above stays unweighted
        # (raw diagnostics); only the optimised total is weighted.
        weights = self.loss_weights(epoch).to(device=losses.device,
                                             dtype=losses.dtype)
        n_spec = self.spec_codebook_num
        return {
            "loss": self._weighted_mean(losses, weights),
            "spec_loss": self._weighted_mean(losses[:n_spec],
                                             weights[:n_spec]),
            "pitch_loss": self._weighted_mean(losses[n_spec:],
                                              weights[n_spec:]),
            "acc": accs.mean(),
            "spec_acc": accs[:n_spec].mean(),
            "pitch_acc": accs[n_spec:].mean(),
            "code_losses": losses,
            "code_accs": accs,
            "code_usage_entropy": entropies,
            "loss_weights": weights,
        }

    @staticmethod
    def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """``sum(w * v) / sum(w)``.

        ``numel()`` checks are pure Python so this adds no GPU sync per step;
        the ``clamp_min`` guards a degenerate all-zero weight group (schedule
        intentionally muting a group), where ``sum(w)`` would be 0.
        """
        if values.numel() == 0:
            return values.new_zeros(())
        return (values * weights).sum() / weights.sum().clamp_min(1e-8)

    def forward(
        self,
        phone_ids: torch.Tensor,
        phone_lengths: Optional[torch.Tensor],
        sid: torch.Tensor,
        spec_codes: torch.Tensor,
        pitch_codes: torch.Tensor,
        code_lengths: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
        complete: Optional[torch.Tensor] = None,
    ) -> dict:
        """Training forward pass: phonemes + speaker -> spec/pitch code loss.

        ``epoch`` only selects the codebook weighting schedule (see
        :meth:`loss_weights`); it does not affect the model's computation.
        ``complete`` (``(B,)`` bool) marks the samples whose segment reaches the
        end of the clip and therefore get an EOS target (no-op unless
        ``transformer.eos_token`` is set).
        """
        memory, memory_mask = self.encode(phone_ids, phone_lengths, sid)
        codes = torch.cat([spec_codes, pitch_codes], dim=1).long()
        return self.code_loss(
            memory, memory_mask, codes, code_lengths, sid=sid, epoch=epoch,
            complete=complete)

    def compute_batch_loss(self, batch: dict,
                           epoch: Optional[int] = None) -> dict:
        """Convenience wrapper used by the Lightning module.

        Expects ``phone_ids``, ``phone_lengths``, ``sid``, ``spec_codes``,
        ``pitch_codes`` and optional ``code_lengths`` / ``complete`` entries.
        ``epoch`` drives the codebook weight curriculum.
        """
        return self.forward(
            phone_ids=batch["phone_ids"],
            phone_lengths=batch.get("phone_lengths"),
            sid=batch["sid"],
            spec_codes=batch["spec_codes"],
            pitch_codes=batch["pitch_codes"],
            code_lengths=batch.get("code_lengths"),
            epoch=epoch,
            complete=batch.get("complete"),
        )

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(
        self,
        phone_ids: torch.Tensor,
        phone_lengths: Optional[torch.Tensor],
        sid: torch.Tensor,
        max_frames: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        stop_on_eos: Optional[bool] = None,
        return_lengths: bool = False,
    ):
        """Autoregressively sample ``(B, K, n)`` delayed codes.

        The returned tensor has the spec codebooks first (``spec_codebook_num``
        rows) followed by the pitch codebooks, matching the layout used for
        training.  Uses the same delay pattern as :meth:`code_loss`.

        When the model was trained with EOS (``transformer.eos_token``) and
        ``stop_on_eos`` is left at ``None``/``True``, decoding stops at the first
        EOS predicted by codebook 0 -- it leads the delay pattern, so its column
        index *is* its frame index -- and the result is trimmed to the predicted
        length.  ``n`` is then what the model chose (``max_frames`` is only an
        upper bound); without EOS ``n == max_frames``, as before.

        The ``K - 1`` columns that fill the *other* codebooks' delay tails are
        still decoded after the stop.  Frames past a *shorter* sample's end in a
        batch are filled by repeating that sample's last frame, never the pad id
        (``vocab_size`` is out of range for the stage-1 codec).  For the same
        reason the EOS class is masked in every codebook except the leading one:
        only codebook 0 terminates the utterance, so the returned codes are
        always in ``[0, vocab_size)``.

        Args:
            max_frames: hard upper bound on the number of code frames.
            temperature, top_k: decoding knobs (``top_k=None`` disables).
            stop_on_eos: ``None`` -> ``self.eos_token``.  Ignored when the model
                was built without EOS.
            return_lengths: also return the ``(B,)`` per-sample frame counts.

        Returns:
            ``(B, K, n)`` codes, or ``(codes, lengths)`` if ``return_lengths``.
        """
        was_training = self.training
        self.eval()
        use_eos = self.eos_token if stop_on_eos is None else bool(stop_on_eos)
        use_eos = use_eos and self.eos_token
        try:
            memory, memory_mask = self.encode(phone_ids, phone_lengths, sid)
            B = phone_ids.shape[0]
            K = self.codebook_num
            L = max_frames + K - 1
            device = phone_ids.device

            seq = self._pad_column(B, device)
            cols = []
            # Column at which codebook 0 emitted EOS == the predicted frame
            # count.  ``L`` is the "still running" sentinel, never a column index.
            stop_col = torch.full((B,), L, dtype=torch.long, device=device)
            for i in range(L):
                logits = self.decoder(seq, memory, memory_mask, sid=sid)
                col = []
                for k in range(K):
                    lg = logits[k][:, -1] / max(temperature, 1e-6)
                    # Never emit the pad/start class as a real token.
                    lg = lg.clone()
                    lg[..., self.pad_ids[k]] = float("-inf")
                    # Only codebook 0 decides where the utterance ends (it
                    # leads the delay pattern), and every book shares that
                    # length.  Mask the EOS class in the other books -- and in
                    # book 0 too when stopping is disabled -- so an EOS id can
                    # never end up in the output: it is one above the codebook
                    # size, i.e. out of range for the stage-1 codec.
                    if self.eos_token and (k > 0 or not use_eos):
                        lg[..., self.eos_ids[k]] = float("-inf")
                    if top_k is not None:
                        kth = torch.topk(lg, min(top_k, lg.size(-1)), dim=-1).values[..., -1:]
                        lg = torch.where(lg < kth, torch.full_like(lg, float("-inf")), lg)
                    probs = F.softmax(lg, dim=-1)
                    col.append(torch.multinomial(probs, num_samples=1).squeeze(-1))
                col = torch.stack(col, dim=1)  # (B, K)

                frame = i - torch.arange(K, device=device)  # (K,)
                valid = (frame >= 0) & (frame < max_frames)
                pad = torch.tensor(self.pad_ids, device=device).view(1, K)
                col = torch.where(valid.view(1, K), col, pad)

                if use_eos and i < max_frames:
                    # Codebook 0 leads the pattern (frame == column), so its EOS
                    # is the utterance end.  Record only the first one.
                    hit = (col[:, 0] == self.eos_ids[0]) & (stop_col == L)
                    stop_col = torch.where(hit, torch.full_like(stop_col, i), stop_col)

                seq = torch.cat([seq, col.unsqueeze(-1)], dim=2)
                cols.append(col)

                # Each codebook needs its own tail: codebook k's frame t lives in
                # column t + k, so frame ``stop - 1`` in the last book needs
                # column ``stop + K - 2``.  Running samples keep the bound at
                # ``L + K - 2``, which the loop never reaches.
                if use_eos and i >= int((stop_col + K - 2).max()):
                    break

            cols = torch.stack(cols, dim=2)  # (B, K, <=L)
            if use_eos:
                # At least one frame, so callers never see a zero-length clip.
                lengths = stop_col.clamp(min=1, max=max_frames)
            else:
                lengths = torch.full((B,), max_frames, dtype=torch.long,
                                     device=device)
            n = int(lengths.max())
            out = cols.new_zeros((B, K, n))
            for k in range(K):
                take = min(n, cols.shape[2] - k)
                if take > 0:
                    out[:, k, :take] = cols[:, k, k : k + take]
            if use_eos:
                for b in range(B):
                    # Repeat the last frame rather than leaving pad ids behind:
                    # the pad id equals the codebook size, which the stage-1
                    # codec would index out of range on.
                    l = int(lengths[b])
                    if l < n:
                        out[b, :, l:] = out[b, :, l - 1 : l]
            if return_lengths:
                return out, lengths
            return out
        finally:
            if was_training:
                self.train()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> str:
        n = self.num_parameters()
        spk = ("encoder+decoder" if self.decoder_speaker_cond else "encoder only")
        return (
            f"CodeTransformer(hidden={self.hidden_size}, codebooks="
            f"{self.spec_codebook_num} spec + {self.pitch_codebook_num} pitch, "
            f"speaker={spk}, memory_norm={self.memory_norm}, "
            f"residual_scale={self.residual_scale}, eos={self.eos_token}, "
            f"params={n / 1e6:.2f}M)"
        )


def expand_state_dict_with_eos(model: "CodeTransformer", state_dict: dict,
                               prefix: str = "net.") -> dict:
    """Grow EOS-less ``v + 1`` codebook weights to this model's ``v + 2`` layout.

    Enabling ``transformer.eos_token`` widens every codebook embedding and output
    head by one row (and each head's bias by one element), so
    :func:`commons.load_state_dict_mismatch` would *skip the whole tensor* when
    warm-starting from an EOS-less checkpoint -- silently re-initialising all
    input embeddings and output heads, i.e. much of what the stage-2 model knows
    about codes.

    This returns a copy of ``state_dict`` whose matching entries are padded with
    one extra zero row (zero bias for the EOS class), so the old rows load and
    only the new EOS class is fresh.  Zero (not a copy of another row) is
    deliberate: the EOS *logit* then starts at the head's own bias, i.e. 0, an
    unremarkable ``1/(v+2)`` probability instead of inheriting the pad class's
    logit, which would make a fresh model stop at frame 0.

    Args:
        model: the (EOS-enabled) model being loaded into.
        state_dict: checkpoint state dict, with or without ``prefix``.
        prefix: key prefix used by the checkpoint (``"net."`` for Lightning
            stage-2 checkpoints, ``""`` for a bare ``CodeTransformer``).

    Returns:
        A new dict; the input is returned unchanged when the model has no EOS
        class or nothing matches (so a stale/other checkpoint still loads).
    """
    if not getattr(model, "eos_token", False):
        return state_dict
    out = dict(state_dict)
    for k, vocab in enumerate(model.vocab_sizes):
        names = (f"embeds.{k}.weight", f"heads.{k}.weight", f"heads.{k}.bias")
        for name in names:
            for key in (prefix + f"decoder.{name}", f"decoder.{name}"):
                src = out.get(key)
                if src is None or src.shape[0] != vocab + 1:
                    continue
                grown = src.new_zeros((vocab + 2, *src.shape[1:]))
                grown[: vocab + 1] = src
                out[key] = grown
    return out


if __name__ == "__main__":
    import os

    from omegaconf import OmegaConf

    cfg_path = "configs/transformer.yaml"
    if not os.path.exists(cfg_path):
        cfg_path = "configs/base.yaml"
    hp = OmegaConf.load(cfg_path)
    # Add transformer hyper-params if the chosen config lacks them.
    if "transformer" not in hp:
        hp.transformer = OmegaConf.create(
            dict(
                hidden_size=768,
                n_heads=12,
                encoder_layers=4,
                decoder_layers=10,
                dim_feedforward=2048,
                dropout=0.1,
                max_positions=4096,
                phone_vocab_size=1024,
                n_speakers=512,
            )
        )

    model = CodeTransformer(hp)
    print(model.describe())

    B, N, T = 2, 40, 64
    phone_ids = torch.randint(0, 1024, (B, N))
    phone_lengths = torch.tensor([N, N - 5])
    sid = torch.randint(0, 51, (B,))
    spec_codes = torch.randint(0, hp.codec.codebook_size, (B, hp.codec.codebook_num, T))
    pitch_codes = torch.randint(
        0, hp.pitch_codec.codebook_size, (B, hp.pitch_codec.codebook_num, T)
    )
    code_lengths = torch.tensor([T, T - 7])

    out = model(phone_ids, phone_lengths, sid, spec_codes, pitch_codes, code_lengths)
    print({k: (float(v.detach()) if v.numel() == 1 else v.detach().tolist())
           for k, v in out.items()})

    gen = model.generate(phone_ids[:1], phone_lengths[:1], sid[:1], max_frames=16)
    print("generated", tuple(gen.shape))
