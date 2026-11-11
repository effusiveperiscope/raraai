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
``expand_state_dict_with_special``), so it needs a fresh run or a
``--transfer_from`` warm start -- a ``--resume_from`` of an EOS-less checkpoint
cannot work.

Fill-in-the-middle (``transformer.fim``)
----------------------------------------
Adds two sentinel classes per codebook (SUF, MID).  ``apply_fim`` reorders a
training item to ``prefix <SUF> suffix <MID> middle`` (frame level, before the
delay pattern) so the model learns to regenerate a span given its two sides;
plain AR items in the same batch are untouched.  ``transformer.region_cond``
additionally adds a decoder-side prefix/suffix/middle region tag.  At inference
``infill`` builds the ``prefix <SUF> suffix <MID>`` prompt and samples a
fixed-length middle.  See ``configs/transformer.yaml`` and ``test_fim.py``.

Length conditioning (``transformer.length_cond``)
-------------------------------------------------
Without it nothing in the model's input depends on *how long* the span to
regenerate is: the prompt, the region tags and the positions are identical for a
6-frame and a 72-frame hole, there is no ``MID_END`` terminator, and the middle
is always the *tail* of the reordered sequence (so the decoder cannot even see
that the span ends).  The caller's ``middle_frames`` only decides when the loop
stops sampling, i.e. the model emits its best continuation and gets chopped.
The only remaining route was "infer the duration from the phonemes plus the
suffix", which a weakly text-conditioned model cannot do.

``length_cond`` adds one more per-column additive embedding
(``decoder.len_emb``, ``n_len_buckets`` rows, exactly like ``region_emb``):
every column inside a middle span is tagged with a log2 bucket of the number of
frames left in the span (bucket 1 = 1 frame left, 2 = 2-3, 3 = 4-7, ...; bucket 0
= "not inside a span", which is what plain AR columns and the prefix/suffix
get).  ``apply_fim`` fills the tags from the drawn span at train time and
``infill`` from the caller's ``middle_frames``, so the decoder knows both the
budget and its progress through it, and can pace the content instead of
overrunning (repetition at the tail) or stopping mid-word.  It changes the
parameter set -- one ``(n_len_buckets, hidden_size)`` table -- so a checkpoint
trained without it needs a fresh ``exp_name`` + ``--transfer_from``.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:                                    # repo root on sys.path (training/tests)
    from hints.levels import (DEFAULT_PITCH_EDGES, DEFAULT_SLOT_BUCKETS,
                              load_levels as load_hint_levels)
except ImportError:                     # `python modeling/transformer.py`
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(
        _os.path.abspath(__file__))))
    from hints.levels import (DEFAULT_PITCH_EDGES, DEFAULT_SLOT_BUCKETS,
                              load_levels as load_hint_levels)


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
        hint_vec: Optional[torch.Tensor] = None,  # (B, H) clip-level hints
        pos_hint: Optional[torch.Tensor] = None,  # (B, N, H) per-phone hints
    ):
        """Encode phonemes to a memory.

        ``hint_vec`` (intensity + register + semantic, one vector per clip) and
        ``pos_hint`` (word pitch + word duration, one vector per phone) are
        **additive side channels** produced by :meth:`CodeTransformer.encode`
        from ``transformer.hints``.  They are added before the positional
        encoding, exactly like ``spk_emb``, so their contribution follows the
        same path through the residual stream; ``None`` (the default) is a plain
        unconditional encode.
        """
        B, N = phone_ids.shape
        if phone_lengths is None:
            phone_lengths = torch.full(
                (B,), N, device=phone_ids.device, dtype=torch.long
            )
        phone_lengths = phone_lengths.to(phone_ids.device).clamp(min=1, max=N)
        valid = torch.arange(N, device=phone_ids.device)[None, :] < phone_lengths[:, None]

        x = self.phone_emb(phone_ids)
        x = x + self.spk_emb(sid.to(phone_ids.device)).unsqueeze(1)
        if hint_vec is not None:
            x = x + hint_vec.to(x.dtype).unsqueeze(1)
        if pos_hint is not None:
            x = x + pos_hint.to(x.dtype)[:, :N]
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
        fim: bool = False,
        region_cond: bool = False,
        n_regions: int = 4,
        length_cond: bool = False,
        n_len_buckets: int = 9,
    ):
        super().__init__()
        self.vocab_sizes = list(vocab_sizes)
        self.num_codebooks = len(self.vocab_sizes)
        self.hidden_size = hidden_size
        self.speaker_cond = bool(speaker_cond)
        # Extra per-codebook classes: ``vocab_size`` is pad/BOS, ``+1`` is EOS
        # (``transformer.eos_token``), ``+2``/``+3`` are the FIM SUF/MID
        # sentinels (``transformer.fim``).  See the module docstring.
        self.eos = bool(eos)
        self.fim = bool(fim)
        self.num_classes = [v + 1 + int(self.eos) + 2 * int(self.fim)
                            for v in self.vocab_sizes]

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

        # FIM region tag (prefix / suffix / middle), added to every decoder
        # column exactly like ``spk_emb``.  Region ids: 0 = normal (plain AR
        # item), 1 = prefix, 2 = suffix, 3 = middle.  This is a *side channel*,
        # not a sequence token: it tells the decoder the role of a position
        # (dense, per column), whereas the SUF/MID sentinel tokens only mark
        # *where* the boundaries are (sparse, and smeared across K columns by
        # the delay pattern).  See ``CodeTransformer.apply_fim``.
        if region_cond:
            self.region_emb = nn.Embedding(n_regions, hidden_size)
        else:
            self.region_emb = None

        # FIM span-length budget (see the module docstring).  Same mechanism as
        # ``region_emb``: bucket ``0`` means "this column is not inside a middle
        # span" (plain AR, prefix, suffix, sentinels), buckets ``1..n-1`` are a
        # log2 bucket of *frames left in the span*.  Without it the decoder has
        # no way to know how much audio the requested span is supposed to hold.
        self.n_len_buckets = int(n_len_buckets)
        if length_cond:
            if self.n_len_buckets < 2:
                raise ValueError(
                    "length_cond=True requires n_len_buckets >= 2, got "
                    f"{self.n_len_buckets}")
            self.len_emb = nn.Embedding(self.n_len_buckets, hidden_size)
        else:
            self.len_emb = None

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

    def _forward_impl(
        self,
        codes: torch.Tensor,  # (B, K, L) long, pad index == vocab size
        memory: torch.Tensor,  # (B, N, H)
        memory_mask: Optional[torch.Tensor] = None,  # (B, N) bool, True == valid
        sid: Optional[torch.Tensor] = None,  # (B,) long
        region: Optional[torch.Tensor] = None,  # (B, L) long, FIM region ids
        length: Optional[torch.Tensor] = None,  # (B, L) long, span-length buckets
        hint_vec: Optional[torch.Tensor] = None,  # (B, H) clip-level hints
        use_mask_cache: bool = True,
    ) -> List[torch.Tensor]:
        """Body of :meth:`forward`, with the causal-mask source selectable.

        ``use_mask_cache=True`` reuses ``_causal_mask``'s cached ``(L, L)``
        tensor, which is what eager training wants (the shape is constant at
        ``max_frames + K - 1``).

        ``use_mask_cache=False`` rebuilds the mask inside the call instead, so
        the whole body is a pure function of its inputs with no Python-side
        mutable state.  ``2_training.py`` compiles *that* variant
        (``train.compile_decoder``): the cache's ``if self._causal_mask_key !=
        key`` is a Python branch on a shape-dependent key, which under
        ``torch.compile``/dynamo is a guard (and recompilation) at best and a
        graph break at worst.
        """
        B, K, L = codes.shape
        x = self.embed(codes)
        x = self.pos(x)
        if self.spk_emb is not None:
            if sid is None:
                raise ValueError(
                    "DelayCodebookDecoder was built with speaker_cond=True "
                    "but no sid was passed")
            x = x + self.spk_emb(sid.to(codes.device)).unsqueeze(1)

        # FIM region tag (prefix/suffix/middle), a per-column additive side
        # channel.  ``region`` is optional: a plain AR caller (or a model built
        # without ``transformer.region_cond``) gets column region 0 ("normal"),
        # which is what AR training uses, so nothing changes by omission.
        if self.region_emb is not None:
            if region is None:
                region = torch.zeros(B, x.size(1), dtype=torch.long,
                                     device=codes.device)
            x = x + self.region_emb(region.to(codes.device))

        # FIM span-length budget, same shape/convention as ``region``: bucket 0
        # ("not inside a middle span") is what every AR/prefix/suffix column
        # gets, so a plain AR caller changes nothing by omitting it.
        if self.len_emb is not None:
            if length is None:
                length = torch.zeros(B, x.size(1), dtype=torch.long,
                                     device=codes.device)
            x = x + self.len_emb(length.to(codes.device))

        # Clip-level conditioning hints (intensity + semantic), a constant added
        # to every column like ``spk_emb``: the decoder needs to know the target
        # loudness/affect directly, not only through the encoded memory.  Built
        # by ``CodeTransformer._dec_hint``; ``None`` means "this model has no
        # hint tables, or the caller passed none".
        if hint_vec is not None:
            x = x + hint_vec.to(x.dtype).unsqueeze(1)

        if use_mask_cache:
            tgt_mask = self._causal_mask(L, x.device, x.dtype)
        else:
            tgt_mask = torch.full((L, L), float("-inf"), device=x.device,
                                  dtype=x.dtype)
            tgt_mask = torch.triu(tgt_mask, diagonal=1)
        mem_pad = None if memory_mask is None else ~memory_mask
        out = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=mem_pad,
        )
        return [head(out) for head in self.heads]

    def forward(
        self,
        codes: torch.Tensor,  # (B, K, L) long, pad index == vocab size
        memory: torch.Tensor,  # (B, N, H)
        memory_mask: Optional[torch.Tensor] = None,  # (B, N) bool, True == valid
        sid: Optional[torch.Tensor] = None,  # (B,) long
        region: Optional[torch.Tensor] = None,  # (B, L) long, FIM region ids
        length: Optional[torch.Tensor] = None,  # (B, L) long, length buckets
        hint_vec: Optional[torch.Tensor] = None,  # (B, H) clip-level hints
    ) -> List[torch.Tensor]:
        return self._forward_impl(codes, memory, memory_mask, sid, region, length,
                                  hint_vec, use_mask_cache=True)


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
        # Fill-in-the-middle (v0.6-fim).  Adds two sentinel classes per codebook
        # (SUF, MID) and lets training mix prefix-suffix-middle reordered
        # samples with normal AR samples.  Like ``eos_token`` it changes the
        # embedding/head shapes, so a checkpoint trained without it cannot be
        # resumed into a model built with it (use --transfer_from).
        fim = bool(_get_cfg(tcfg, "fim", False))
        # Decoder-side prefix/suffix/middle region tag (see ``apply_fim``).
        # Cheap additive embedding, only meaningful together with ``fim`` but
        # independent to allow the "sentinel-only" ablation.
        region_cond = bool(_get_cfg(tcfg, "region_cond", False))
        # FIM span-length budget (see the module docstring): a second additive
        # per-column embedding telling the decoder how many frames are left in
        # the span it is filling.  Independent switch so the "no length info"
        # variant stays available as an ablation.
        length_cond = bool(_get_cfg(tcfg, "length_cond", False))
        n_len_buckets = int(_get_cfg(tcfg, "n_len_buckets", 9))
        # Encoder -> decoder conditioning (v0.4.1).  See PhonemeEncoder.
        memory_norm = str(_get_cfg(tcfg, "memory_norm", "temporal"))
        residual_scale = bool(_get_cfg(tcfg, "residual_scale", True))

        # Coarse conditioning hints (v0.8).  The tables are built below; the
        # level counts come from the *same* ``hints.json`` the dataset used, so
        # a level id means the same thing on both sides by construction.
        hints_on = bool(_get_cfg(tcfg, "hints", False))
        hints_json = _get_cfg(tcfg, "hints_json", None)
        decoder_hints = bool(_get_cfg(tcfg, "decoder_hints", True))
        semantic_dim = int(_get_cfg(tcfg, "semantic_dim", 384))
        pitch_edges = list(_get_cfg(tcfg, "hint_pitch_edges", None)
                           or DEFAULT_PITCH_EDGES)
        n_slot_buckets = int(_get_cfg(tcfg, "n_slot_buckets",
                                      DEFAULT_SLOT_BUCKETS)
                             or DEFAULT_SLOT_BUCKETS)

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
        # FIM sentinel indices: SUF and MID sit directly above the EOS class.
        # With EOS off they shift down to vocab_size + 1 / + 2.
        self.fim = fim
        self.region_cond = region_cond
        self.n_regions = 4
        self.length_cond = length_cond
        self.n_len_buckets = max(int(n_len_buckets), 2)
        _special_base = 1 + int(eos_token)
        self.suf_ids = ([v + _special_base for v in self.vocab_sizes]
                        if fim else None)
        self.mid_ids = ([v + _special_base + 1 for v in self.vocab_sizes]
                        if fim else None)
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
            fim=fim,
            region_cond=region_cond,
            n_regions=self.n_regions,
            length_cond=length_cond,
            n_len_buckets=self.n_len_buckets,
        )
        self.memory_norm = memory_norm
        self.residual_scale = residual_scale

        # ---- conditioning-hint tables (v0.8) ------------------------------- #
        # Every channel is a small ``(n_levels + 1, hidden)`` embedding indexed
        # by a level id, row 0 being the trained "not provided" state (the
        # dataset applies per-channel dropout, so that row is always exercised).
        # They live here rather than on the encoder/decoder so the checkpoint
        # has one obvious place for every hint parameter, and so the semantic
        # projection can feed both halves without being registered twice.
        self.hints = bool(hints_on)
        self.decoder_hints = bool(decoder_hints) and self.hints
        self.hints_json = str(hints_json) if hints_json else None
        self.semantic_dim = int(semantic_dim)
        self.pitch_edges = [float(e) for e in pitch_edges]
        self.n_slot_buckets = max(int(n_slot_buckets), 1)
        self.n_intensity_levels = 0
        self.n_register_levels = 0
        # Voiced pitch bins; row ``n_pitch_levels + 1`` is "this word is
        # unvoiced" (see ``hints.levels.word_pitch_levels``), hence ``+ 2`` rows.
        self.n_pitch_levels = len(self.pitch_edges) + 1
        if self.hints:
            if not self.hints_json:
                raise ValueError(
                    "transformer.hints is on but transformer.hints_json is not "
                    "set. It must point at the hints.json written by "
                    "hints/build.py -- the dataset reads the same file, so the "
                    "two sides cannot disagree about a level id.")
            levels = load_hint_levels(self.hints_json)
            for name, attr in (("intensity", "n_intensity_levels"),
                               ("register", "n_register_levels")):
                spec = levels.get(name) or {}
                if not spec.get("edges"):
                    raise ValueError(
                        f"{self.hints_json} has no fitted edges for hint "
                        f"{name!r} (found: {sorted(levels)}); re-run "
                        f"hints/build.py without --no-edges.")
                setattr(self, attr, int(spec["n_levels"]))
            self.int_emb = nn.Embedding(self.n_intensity_levels + 1, hidden_size)
            self.reg_emb = nn.Embedding(self.n_register_levels + 1, hidden_size)
            self.pitch_emb = nn.Embedding(self.n_pitch_levels + 2, hidden_size)
            self.slot_emb = nn.Embedding(self.n_slot_buckets + 1, hidden_size)
            self.sem_proj = nn.Linear(self.semantic_dim, hidden_size)
            self.sem_null = nn.Parameter(torch.zeros(hidden_size))
            self.dec_int_emb = (nn.Embedding(self.n_intensity_levels + 1,
                                             hidden_size)
                                if self.decoder_hints else None)
        else:
            self.int_emb = None
            self.reg_emb = None
            self.pitch_emb = None
            self.slot_emb = None
            self.sem_proj = None
            self.sem_null = None
            self.dec_int_emb = None

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
    # Fill-in-the-middle (v0.6-fim)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _len_buckets(remaining: torch.Tensor, n_buckets: int) -> torch.Tensor:
        """Map "frames left in the span" to log2 bucket ids.

        ``0`` means "not inside a middle span" and is preserved (a span never
        has 0 frames left: the last frame of a span is bucket 1).  For ``r >= 1``
        the bucket is ``1 + floor(log2(r))`` clamped into ``[1, n_buckets - 1]``,
        i.e. 1 / 2-3 / 4-7 / 8-15 / 16-31 / 32-63 / 64-127 / 128+ for the default
        ``n_buckets = 9``.  Log spacing because "how much is left" needs to be
        coarse far from the end and precise near it; the clamp is a ``where``
        on ``r > 0`` so the trailing-padding frames of a short item (which get
        bucket 0) cost no branch.

        Pure elementwise tensor maths: no GPU->CPU sync, so this can run once per
        batch inside ``apply_fim``.
        """
        r = remaining.to(torch.float32)
        bucket = torch.floor(torch.log2(r.clamp(min=1.0))) + 1.0
        bucket = torch.where(r > 0, bucket, torch.zeros_like(bucket))
        return bucket.clamp(min=0.0, max=float(n_buckets - 1)).long()

    def apply_fim(self, codes, code_lengths, fim_mask, prefix_len, mid_len):
        """Reorder each FIM item to ``prefix <SUF> suffix <MID> middle``.

        The reordering is done at the **frame** level (all K codebooks move
        together), because the delay pattern requires frame-aligned codebooks;
        :meth:`build_delay_pattern` then handles the per-codebook shift
        unchanged.  Items with ``fim_mask=False`` are left in plain AR order
        (region 0) and merely padded to the same width, so AR and FIM samples
        can share a batch.

        Fully vectorised on-device: ``src`` is a per-(item, frame) gather index
        shared by all codebooks, sentinels are written with a ``where`` and the
        region tag is built from the same masks, so no GPU->CPU sync is added.

        Args:
            codes: ``(B, K, T)`` real code frames.
            code_lengths: ``(B,)`` valid frames per item.
            fim_mask: ``(B,)`` which items to reorder.
            prefix_len, mid_len: ``(B,)`` span lengths (suffix = len - p - m).

        Returns:
            ``(codes_new, lengths_new, region, len_bucket)``: ``(B, K, T+2)``,
            ``(B,)`` (``T+2`` for FIM items, the original length for AR items),
            ``(B, T+2)`` region ids (0 normal, 1 prefix, 2 suffix, 3 middle) and
            ``(B, T+2)`` span-length buckets (``transformer.length_cond``):
            ``0`` outside the middle, else the log2 bucket of the number of
            frames left in the span *including* the current one, so ``MID``
            itself is ``m + 1`` and the last middle frame is ``1``.
        """
        B, K, T = codes.shape
        device = codes.device
        Tn = T + 2
        pos = torch.arange(Tn, device=device)                 # (Tn,)
        cl = code_lengths.to(device).long().clamp(min=1, max=T)
        p = prefix_len.to(device).long().clamp(min=0, max=T)[:, None]
        p = torch.minimum(p, (cl - 1).clamp(min=0)[:, None])
        m = mid_len.to(device).long().clamp(min=0, max=T)[:, None]
        m = torch.minimum(m, cl[:, None] - p)
        s = (cl[:, None] - p - m).clamp(min=0)                # suffix length
        isfim = fim_mask.to(device).bool()[:, None]           # (B, 1)

        pos2 = pos[None, :]                                   # (1, Tn)
        in_prefix = pos2 < p
        is_suf = pos2 == p
        in_suffix = (pos2 > p) & (pos2 <= p + s)
        is_mid = pos2 == p + s + 1
        in_mid = (pos2 >= p + s + 2) & (pos2 < p + s + 2 + m)

        src_fim = torch.where(
            in_prefix, pos2,
            torch.where(in_suffix, pos2 - 1 + m,
                        torch.where(in_mid, pos2 - s - 2,
                                    torch.zeros_like(pos2))))
        src_ar = pos2.expand(B, Tn)
        valid_ar = pos2 < cl[:, None]
        # A FIM item's reordered stream is exactly ``cl + 2`` frames long; any
        # frame past that is collator padding (short item in a longer batch) and
        # must stay masked, exactly like the AR padding.
        valid_fim = pos2 < (cl + 2)[:, None]

        reg_fim = torch.zeros((B, Tn), dtype=torch.long, device=device)
        reg_fim = torch.where(in_prefix, torch.ones_like(reg_fim), reg_fim)
        reg_fim = torch.where(in_suffix | is_suf,
                              torch.full_like(reg_fim, 2), reg_fim)
        reg_fim = torch.where(in_mid | is_mid,
                              torch.full_like(reg_fim, 3), reg_fim)
        reg_ar = torch.zeros((B, Tn), dtype=torch.long, device=device)

        src = torch.where(isfim, src_fim, src_ar)
        region = torch.where(isfim, reg_fim, reg_ar)
        valid = torch.where(isfim, valid_fim, valid_ar)

        src_c = src.clamp(0, T - 1)
        gathered = torch.gather(codes, 2, src_c[:, None, :].expand(B, K, Tn))
        sent = torch.full((B, K, Tn), -1, dtype=codes.dtype, device=device)
        suf_base = torch.tensor(self.suf_ids, device=device, dtype=codes.dtype)
        mid_base = torch.tensor(self.mid_ids, device=device, dtype=codes.dtype)
        sent = torch.where(is_suf[:, None, :].expand(B, K, Tn),
                           suf_base.view(1, K, 1).expand(B, K, Tn), sent)
        sent = torch.where(is_mid[:, None, :].expand(B, K, Tn),
                           mid_base.view(1, K, 1).expand(B, K, Tn), sent)
        out = torch.where(sent >= 0, sent, gathered)

        pad = torch.tensor(self.pad_ids, device=device,
                           dtype=codes.dtype).view(1, K, 1)
        out = torch.where(valid[:, None, :].expand(B, K, Tn),
                          out, pad.expand(B, K, Tn))
        lengths_new = torch.where(fim_mask.to(device).bool(), cl + 2, cl)

        # Span-length budget (``transformer.length_cond``): frames left in the
        # middle *including* the current frame, so the ``MID`` sentinel reports
        # ``m + 1`` and the last middle frame reports ``1`` -- the same formula
        # ``infill`` uses (``Tn - pos``), which is what makes the tag mean the
        # same thing at train and inference time.
        mid_first = p + s + 2                       # first middle frame
        remaining = torch.where(
            (in_mid | is_mid),
            (mid_first + m) - pos2,
            torch.zeros_like(pos2))
        len_bucket = torch.where(
            isfim,
            self._len_buckets(remaining, self.n_len_buckets),
            torch.zeros((B, Tn), dtype=torch.long, device=device))
        return out, lengths_new.long(), region, len_bucket

    def _tag_to_columns(self, tag: torch.Tensor, out_width: int,
                        L: int) -> torch.Tensor:
        """Map a per-frame tag ``(B, Tn)`` to per-column ids ``(B, L)``.

        Column ``c`` of the delayed matrix carries (among others) codebook 0's
        frame ``c``, so its tag is ``tag[:, c]`` (clamped into the last frame for
        the ``K - 1`` delay-tail columns).  ``tag`` is first grown to the
        post-EOS ``out_width`` by repeating its last frame, so the EOS frame
        inherits the tag of the frame it terminates.

        Used for both the FIM **region** ids and the FIM **span-length** buckets:
        the two side channels share the alignment rule, which is what keeps
        training and :meth:`infill` consistent (see ``CodeTransformer.infill``).
        """
        Tr = tag.shape[1]
        if Tr < out_width:
            tail = tag[:, -1:].expand(-1, out_width - Tr)
            tag = torch.cat([tag, tail], dim=1)
        elif Tr > out_width:
            tag = tag[:, :out_width]
        idx = torch.arange(L, device=tag.device).clamp(max=out_width - 1)
        return tag[:, idx]

    def _region_to_columns(self, region: torch.Tensor, out_width: int,
                           L: int) -> torch.Tensor:
        """Backwards-compatible alias of :meth:`_tag_to_columns` for regions."""
        return self._tag_to_columns(region, out_width, L)

    def _middle_target_mask(self, region: torch.Tensor, targets: torch.Tensor,
                            L: int) -> torch.Tensor:
        """Boolean ``(B, K, L)`` mask of middle-region, non-pad targets.

        Diagnostics only (drives ``fim_middle_loss``).  Target column ``j`` of
        codebook ``k`` maps to reordered frame ``j - k`` (the same relationship
        ``build_delay_pattern`` uses), so the mask is gathered through it.
        """
        B, K, _ = targets.shape
        Tr = region.shape[1]
        if Tr < L:
            region = torch.cat([region, region[:, -1:].expand(-1, L - Tr)],
                               dim=1)
        else:
            region = region[:, :L]
        frame = (torch.arange(L, device=region.device)[None, :]
                 - torch.arange(K, device=region.device)[:, None])  # (K, L)
        frame = frame.clamp(0, L - 1)
        reg_t = region[:, frame]  # (B, L)[B, frame] -> (B, K, L)
        pad = torch.tensor(self.pad_ids, device=region.device).view(1, K, 1)
        return (reg_t == 3) & (targets != pad)

    # ------------------------------------------------------------------ #
    # Core computation
    # ------------------------------------------------------------------ #
    def encode(
        self,
        phone_ids: torch.Tensor,
        phone_lengths: Optional[torch.Tensor],
        sid: torch.Tensor,
        hints: Optional[dict] = None,
    ):
        """Phonemes + speaker (+ hints) -> ``(memory, memory_mask)``.

        ``hints`` is the collated batch (or any dict with the same keys); the
        hint channels are added to the phoneme embeddings inside the encoder.
        Omitting it is identical to a pre-v0.8 model.
        """
        hint_vec, pos_hint = self._enc_hints(hints, phone_ids.device)
        return self.encoder(phone_ids, phone_lengths, sid,
                            hint_vec=hint_vec, pos_hint=pos_hint)

    # ------------------------------------------------------------------ #
    # Conditioning hints (v0.8)
    # ------------------------------------------------------------------ #
    def _semantic_hint(self, hints) -> Optional[torch.Tensor]:
        """Project the 384-d sentence vector to ``hidden_size``.

        Absent / dropped semantic hints go through ``sem_null`` rather than
        through ``sem_proj(0)``: a dropped channel has to be a *distinct,
        trained* state, otherwise "no hint" would look like an arbitrary real
        sentence (the projection has a bias).
        """
        if not self.hints or hints is None:
            return None
        sem = hints.get("hint_sem")
        if sem is None or sem.numel() == 0:
            return None
        if sem.size(-1) != self.semantic_dim:
            raise ValueError(
                f"transformer.semantic_dim={self.semantic_dim} but the batch's "
                f"hint_sem is {tuple(sem.shape)}; the .cond files and the "
                f"config disagree (re-run hints/build.py --hints semantic, or "
                f"fix semantic_dim).")
        x = self.sem_proj(sem.to(self.sem_null.device, torch.float32))
        valid = hints.get("hint_sem_valid")
        if valid is not None:
            keep = valid.to(x.device).bool().unsqueeze(-1)
            x = torch.where(keep, x, self.sem_null.to(x.dtype).expand_as(x))
        return x

    def _enc_hints(self, hints, device):
        """``(clip_vec (B,H) | None, per_phone (B,N,H) | None)`` for the encoder.

        Clip-level: intensity + register + semantic.  Per phone: word pitch
        (including the explicit unvoiced class) + word duration bucket.  Word
        hints are attached to *phones*, not words, because that is the sequence
        the encoder actually sees.
        """
        if not self.hints or hints is None:
            return None, None
        int_ids = hints.get("hint_int")
        if int_ids is None:
            return None, None
        clip = self.int_emb(int_ids.to(device).long())
        reg = hints.get("hint_reg")
        if reg is not None:
            clip = clip + self.reg_emb(reg.to(device).long())
        sem = self._semantic_hint(hints)
        if sem is not None:
            clip = clip + sem

        pos = None
        pitch = hints.get("hint_pitch")
        if pitch is not None and pitch.numel() > 0:
            pos = self.pitch_emb(pitch.to(device).long())
            slot = hints.get("hint_slot")
            if slot is not None:
                pos = pos + self.slot_emb(slot.to(device).long())
        return clip, pos

    def _dec_hint(self, hints) -> Optional[torch.Tensor]:
        """``(B, H)`` constant added to every decoder column (or ``None``).

        Only the *clip-level* hints are repeated here.  Per-phone word hints
        stay encoder-only on purpose: they are a property of the text's pacing,
        and the decoder already receives them through cross-attention; adding
        them per column would have to invent a phone<->frame alignment that does
        not exist at generation time.
        """
        if not self.decoder_hints or hints is None:
            return None
        int_ids = hints.get("hint_int")
        if int_ids is None:
            return None
        out = self.dec_int_emb(int_ids.to(self.dec_int_emb.weight.device).long())
        sem = self._semantic_hint(hints)
        if sem is not None:
            out = out + sem
        return out

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
        diagnostics: bool = True,
        fim_mask: Optional[torch.Tensor] = None,       # (B,) bool
        fim_prefix_len: Optional[torch.Tensor] = None,  # (B,) long
        fim_mid_len: Optional[torch.Tensor] = None,     # (B,) long
        ablate_region: bool = False,
        ablate_length: bool = False,
        hints: Optional[dict] = None,
    ) -> dict:
        """Delayed next-token loss over all codebooks.

        With ``transformer.eos_token`` enabled, an EOS frame is appended to each
        sample's streams *before* the delay pattern is built, so "emit EOS" is
        just another next-token target and the delayed alignment is unchanged
        (see ``_append_eos``).  Only samples flagged ``complete`` get one.

        Per-codebook cross-entropies are combined as
        ``loss = sum_k w_k * CE_k / sum(w)`` with ``w = loss_weights(epoch)``
        (uniform unless ``transformer.codebook_weights`` is set), so the
        coarse codebooks can be prioritised over the fine residual ones.

        Args:
            diagnostics: compute the per-codebook accuracy / usage entropy as
                well.  Those cost a second full softmax over every codebook's
                logits plus an argmax pass, i.e. real GPU time that is only
                useful on the steps whose result is actually logged (see
                ``2_training.py:training_step``, which passes ``False`` on the
                steps between two ``train.log_interval`` log points).  The
                returned dict then simply omits ``acc``/``spec_acc``/
                ``pitch_acc``/``code_accs``/``code_usage_entropy``; the loss
                scalars are bit-identical either way.
            ablate_region, ablate_length: zero the FIM region tags / span-length
                buckets for this call.  Only ``2_training.py``'s FIM probe uses
                these, as the "is this side channel load-bearing?" control arms:
                everything else (masks, targets, spans) stays identical, so the
                resulting ``fim_middle_loss`` difference is directly attributable
                to that channel.
            hints: the collated batch (or any dict with ``hint_int`` /
                ``hint_reg`` / ``hint_sem`` / ``hint_sem_valid``) whose
                clip-level channels are added to every decoder column
                (``transformer.hints``).  ``None`` is a no-op.

        Returns a dict with the total loss plus per-group (spec/pitch) losses
        and diagnostics that make "is it learning?" answerable from
        TensorBoard alone:

        ``loss``/``spec_loss``/``pitch_loss``
            weighted mean cross-entropy over all / spec / pitch codebooks
            (weights renormalised inside the group, so the two group losses
            still average back to ``loss``).
        ``acc``/``spec_acc``/``pitch_acc``
            teacher-forced top-1 accuracy over non-pad target positions.
            Omitted when ``diagnostics=False``.
        ``code_losses`` (K,), ``code_accs`` (K,), ``code_usage_entropy`` (K,)
            the same per codebook, always *unweighted* so the raw per-book
            behaviour stays visible.  ``code_usage_entropy`` is the entropy of
            the batch-averaged predicted distribution, normalised to
            ``[0, 1]`` by ``log(vocab - 1)``; values near 0 mean the head has
            collapsed onto a handful of codes even if the loss still looks
            healthy.  ``code_accs``/``code_usage_entropy`` are omitted when
            ``diagnostics=False``; ``code_losses`` never is.
        ``fim_middle_loss``/``fim_middle_acc`` (+ ``_early``/``_mid``/``_late``)
            diagnostics over the *middle* targets only, i.e. "can it fill the
            hole?" rather than "can it continue?".  The accuracy is split into
            thirds of the span because that is where a model with no sense of the
            span's budget shows up: flat across the span means it is pacing the
            content, a rise at the end means it is only good at wrapping up.
            Diagnostics only.
        ``fim_frac``
            fraction of the batch that was FIM (no-op items included).
        ``loss_weights`` (K,)
            the effective weights used for this batch, for logging.
        """
        B, K, T = codes.shape
        region = None
        len_bucket = None
        if self.fim and fim_mask is not None:
            # Reorder FIM items to prefix <SUF> suffix <MID> middle.  When the
            # suffix is empty the middle reaches the clip's real end, so EOS
            # still belongs after the middle; a mid-clip middle (suffix present)
            # ends at MID_END-by-end-of-sequence and gets no EOS, like a
            # mid-clip AR crop.
            if complete is None:
                complete = torch.ones(B, dtype=torch.bool, device=codes.device)
            suffix_zero = (code_lengths - fim_prefix_len - fim_mid_len) <= 0
            complete = complete & (~fim_mask.bool() | suffix_zero)
            codes, code_lengths, region, len_bucket = self.apply_fim(
                codes, code_lengths, fim_mask, fim_prefix_len, fim_mid_len)
        if self.eos_token:
            codes, code_lengths = self._append_eos(codes, code_lengths, complete)
        delayed = self.build_delay_pattern(codes, code_lengths)  # (B, K, L)
        L = delayed.shape[2]
        start = self._pad_column(B, codes.device)

        seq = torch.cat([start, delayed], dim=2)  # (B, K, L + 1)
        inputs = seq[..., :L]
        targets = seq[..., 1:]

        region_col = None
        len_col = None
        mid_target_mask = None
        if region is not None:
            region_col = self._tag_to_columns(region, codes.shape[2], L)
            if diagnostics:
                mid_target_mask = self._middle_target_mask(region, targets, L)
        if len_bucket is not None and self.decoder.len_emb is not None:
            # Span-length budget, aligned exactly like the region tags (entry j
            # = tag of frame j, see ``_tag_to_columns``).  ``ablate_length``
            # zeroes it for the diagnostic control arm in ``2_training.py``.
            len_col = self._tag_to_columns(len_bucket, codes.shape[2], L)
        if ablate_region:
            region_col = None
        if ablate_length:
            len_col = None

        # list of (B, L, v + 1) -- or (B, L, v + 2) with the EOS class
        logits = self.decoder(inputs, memory, memory_mask, sid=sid,
                              region=region_col, length=len_col,
                              hint_vec=self._dec_hint(hints))

        losses, accs, entropies = [], [], []
        fim_num = None
        fim_den = None
        fim_acc_num = None
        fim_acc_den = None
        fim_bin_num = None
        fim_bin_den = None
        for k in range(K):
            # fp32 for both CE and the diagnostics: under bf16-mixed the
            # logits come back bf16, which is fine for CE but too coarse for
            # a mean over ~1e5 positions.
            lg = logits[k].float()
            flat_logits = lg.reshape(-1, lg.size(-1))
            flat_targets = targets[:, k].reshape(-1)
            losses.append(F.cross_entropy(
                flat_logits, flat_targets, ignore_index=self.pad_ids[k]))
            if not diagnostics:
                continue

            # Extra per-codebook diagnostics (accuracy + predicted-distribution
            # entropy). These are the expensive part of this function -- a
            # second full softmax, an argmax pass and two masking reductions
            # *per codebook* -- so `diagnostics=False` skips them on the
            # training steps that Lightning would not log anyway (see
            # `train.log_interval` and `2_training.py:training_step`).
            #
            # Everything here stays on-device: no `if valid.any()` (that is a
            # GPU->CPU sync per codebook, i.e. 10 syncs per step in the old
            # code) and no boolean-mask gather (which materialises another
            # (n_valid, v) copy). Pad positions are zeroed by multiplication
            # instead and the denominator is `n_valid.clamp_min(1)`, so a
            # degenerate all-masked batch yields 0 rather than NaN.
            mask = (flat_targets != self.pad_ids[k]).unsqueeze(-1).to(lg.dtype)
            n_valid = mask.sum().clamp_min(1.0)
            correct = (flat_logits.argmax(-1) == flat_targets)
            accs.append((correct.to(lg.dtype).unsqueeze(-1) * mask).sum()
                        / n_valid)
            probs = (F.softmax(flat_logits, dim=-1) * mask).sum(0) / n_valid
            ent = -(probs.clamp_min(1e-9).log() * probs).sum()
            # `probs.size(-1)` includes the pad class, which is never a target,
            # so the attainable maximum is log(vocab - 1).  With EOS the
            # end-of-utterance class is included in the mass (it is a real, if
            # rare, target), so the normaliser stays as is.
            entropies.append(ent / math.log(max(probs.size(-1) - 1, 2)))

            # ``fim_middle_loss``: mean CE over middle-region targets only, the
            # "can it fill the hole?" yardstick.  Diagnostics-only (a second,
            # reduction='none' CE per codebook), so it costs nothing on the
            # unlogged steps.  ``fim_middle_acc`` (plus its early/mid/late
            # split) is the meaningful companion: the CE alone cannot say
            # *where* in the span the model fails, and the position profile is
            # exactly the "does it have a sense of the span's budget?" signal
            # (a profile that only looks good at the end of the span means the
            # model is good at wrapping up, not at pacing the content).
            if mid_target_mask is not None:
                ce_none = F.cross_entropy(
                    flat_logits, flat_targets,
                    ignore_index=self.pad_ids[k], reduction="none")
                mk_b = mid_target_mask[:, k]                    # (B, L) bool
                mk = mk_b.reshape(-1).to(ce_none.dtype)
                num = (ce_none * mk).sum()
                den = mk.sum()
                fim_num = num if fim_num is None else fim_num + num
                fim_den = den if fim_den is None else fim_den + den

                # Position within the span, derived from the mask itself (the
                # middle targets of one item/codebook are one contiguous run),
                # so no extra per-frame tensor has to be carried through
                # `apply_fim`.  ``run`` counts 0..m-1 from the span's start.
                cnt = mk_b.to(lg.dtype)
                run = torch.cumsum(cnt, dim=1) - 1.0
                total = cnt.sum(-1, keepdim=True).clamp_min(1.0)
                bin_id = ((run / total) * 3).clamp(min=0, max=2).long()
                ok = correct.to(lg.dtype).reshape(cnt.shape) * cnt
                fim_acc_num = (ok.sum() if fim_acc_num is None
                               else fim_acc_num + ok.sum())
                fim_acc_den = (cnt.sum() if fim_acc_den is None
                               else fim_acc_den + cnt.sum())
                bin_num = []
                bin_den = []
                for b in range(3):
                    sel = cnt * (bin_id == b).to(lg.dtype)
                    bin_num.append((ok * sel).sum())
                    bin_den.append(sel.sum())
                if fim_bin_num is None:
                    fim_bin_num, fim_bin_den = bin_num, bin_den
                else:
                    fim_bin_num = [a + b for a, b in zip(fim_bin_num, bin_num)]
                    fim_bin_den = [a + b for a, b in zip(fim_bin_den, bin_den)]

        losses = torch.stack(losses)  # (K,)
        weights = self.loss_weights(epoch).to(device=losses.device,
                                             dtype=losses.dtype)
        n_spec = self.spec_codebook_num
        out = {
            "loss": self._weighted_mean(losses, weights),
            "spec_loss": self._weighted_mean(losses[:n_spec],
                                             weights[:n_spec]),
            "pitch_loss": self._weighted_mean(losses[n_spec:],
                                              weights[n_spec:]),
            "code_losses": losses,
            "loss_weights": weights,
        }
        if diagnostics:
            accs = torch.stack(accs)  # (K,)
            entropies = torch.stack(entropies)  # (K,)
            out.update({
                "acc": accs.mean(),
                "spec_acc": accs[:n_spec].mean(),
                "pitch_acc": accs[n_spec:].mean(),
                "code_accs": accs,
                "code_usage_entropy": entropies,
            })
            if fim_den is not None:
                out["fim_middle_loss"] = fim_num / fim_den.clamp_min(1e-6)
            if fim_acc_den is not None:
                out["fim_middle_acc"] = (fim_acc_num
                                         / fim_acc_den.clamp_min(1e-6))
                for name, num, den in zip(("early", "mid", "late"),
                                          fim_bin_num, fim_bin_den):
                    out[f"fim_middle_acc_{name}"] = num / den.clamp_min(1e-6)
        if fim_mask is not None:
            out["fim_frac"] = fim_mask.to(losses.dtype).mean()
        return out

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
        diagnostics: bool = True,
        fim_mask: Optional[torch.Tensor] = None,
        fim_prefix_len: Optional[torch.Tensor] = None,
        fim_mid_len: Optional[torch.Tensor] = None,
        hints: Optional[dict] = None,
    ) -> dict:
        """Training forward pass: phonemes + speaker -> spec/pitch code loss.

        ``epoch`` only selects the codebook weighting schedule (see
        :meth:`loss_weights`); it does not affect the model's computation.
        ``complete`` (``(B,)`` bool) marks the samples whose segment reaches the
        end of the clip and therefore get an EOS target (no-op unless
        ``transformer.eos_token`` is set).  ``diagnostics`` is forwarded to
        :meth:`code_loss`.  ``fim_mask``/``fim_prefix_len``/``fim_mid_len``
        select the prefix-suffix-middle reorder (no-op unless
        ``transformer.fim``).
        """
        memory, memory_mask = self.encode(phone_ids, phone_lengths, sid,
                                          hints=hints)
        codes = torch.cat([spec_codes, pitch_codes], dim=1).long()
        return self.code_loss(
            memory, memory_mask, codes, code_lengths, sid=sid, epoch=epoch,
            complete=complete, diagnostics=diagnostics,
            fim_mask=fim_mask, fim_prefix_len=fim_prefix_len,
            fim_mid_len=fim_mid_len, hints=hints)

    def compute_batch_loss(self, batch: dict,
                           epoch: Optional[int] = None,
                           diagnostics: bool = True) -> dict:
        """Convenience wrapper used by the Lightning module.

        Expects ``phone_ids``, ``phone_lengths``, ``sid``, ``spec_codes``,
        ``pitch_codes`` and optional ``code_lengths`` / ``complete`` entries.
        ``epoch`` drives the codebook weight curriculum; ``diagnostics``
        enables the per-codebook accuracy/usage-entropy extras (see
        :meth:`code_loss`).
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
            diagnostics=diagnostics,
            fim_mask=batch.get("fim"),
            fim_prefix_len=batch.get("fim_prefix_len"),
            fim_mid_len=batch.get("fim_mid_len"),
            hints=batch if self.hints else None,
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
        hints: Optional[dict] = None,
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
            memory, memory_mask = self.encode(phone_ids, phone_lengths, sid,
                                              hints=hints)
            dec_hint = self._dec_hint(hints)
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
                logits = self.decoder(seq, memory, memory_mask, sid=sid,
                                      hint_vec=dec_hint)
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
                    # FIM sentinels are never real output tokens: mask them in
                    # every codebook so they can't leak into the stage-1 codec.
                    if self.fim:
                        lg[..., self.suf_ids[k]] = float("-inf")
                        lg[..., self.mid_ids[k]] = float("-inf")
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

    @torch.no_grad()
    def infill(
        self,
        phone_ids: torch.Tensor,
        phone_lengths: Optional[torch.Tensor],
        sid: torch.Tensor,
        prefix_codes: torch.Tensor,  # (B, K, p) known prefix
        suffix_codes: torch.Tensor,  # (B, K, s) known suffix
        middle_frames: int,          # fixed number of middle frames to generate
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        hints: Optional[dict] = None,
    ) -> torch.Tensor:
        """Fixed-length span regeneration: ``(prefix, suffix) -> middle``.

        Builds the delayed prompt ``prefix <SUF> suffix <MID>`` and then samples
        exactly ``middle_frames`` fresh frames, conditioned on both sides.  This
        is the inference counterpart of :meth:`apply_fim` for the "regenerate
        this span" UX: the caller supplies the untouched prefix and suffix and
        decides the length, so the duration problem disappears and no MID_END
        terminator is needed.  The result must be spliced back into the *time*
        order ``prefix + middle + suffix`` (the PSM order puts the middle last in
        the *sequence* only).

        The prefix/suffix sentinel is never sampled; sentinels and EOS are
        masked out of every sampled column.

        **The per-column side channels are built exactly like training.**  In
        :meth:`code_loss` column ``j`` of the delayed matrix is tagged with the
        region/length of frame ``j`` (``_tag_to_columns``), i.e. with the tag of
        the token it is about to predict -- *not* of the token it holds.  Doing
        it the other way round (tagging each column with the tag of the token
        just appended, as this method used to) shifts the whole side channel one
        column: the model sees "prefix" where training saw "suffix" and "suffix"
        where training saw "middle", i.e. it is asked to emit ``SUF``/``MID`` and
        to hand off into the span under the wrong region.  The tags are therefore
        laid out once for the whole column range and sliced by ``seq`` length,
        which reproduces the training alignment to within float noise
        (``test_fim.py`` asserts that).

        Args:
            prefix_codes, suffix_codes: ``(B, K, p)`` / ``(B, K, s)`` known codes.
                Either may be empty (``p``/``s`` == 0).
            middle_frames: number of frames to generate (> 0).
            temperature, top_k: decoding knobs (``top_k=None`` disables).

        Returns:
            ``(B, K, middle_frames)`` freshly generated middle codes.
        """
        was_training = self.training
        if not self.fim:
            raise ValueError(
                "CodeTransformer.infill requires transformer.fim (the SUF/MID "
                "sentinel classes); this model was built without it.")
        self.eval()
        try:
            memory, memory_mask = self.encode(phone_ids, phone_lengths, sid,
                                              hints=hints)
            dec_hint = self._dec_hint(hints)
            B = phone_ids.shape[0]
            K = self.codebook_num
            device = phone_ids.device
            m = int(middle_frames)
            p = int(prefix_codes.shape[-1])
            s = int(suffix_codes.shape[-1])
            if m <= 0:
                return prefix_codes.new_zeros((B, K, 0))

            suf_col = torch.tensor(self.suf_ids, device=device,
                                   dtype=torch.long).view(1, K, 1).expand(B, K, 1)
            mid_col = torch.tensor(self.mid_ids, device=device,
                                   dtype=torch.long).view(1, K, 1).expand(B, K, 1)
            known = torch.cat([prefix_codes.long(), suf_col,
                               suffix_codes.long(), mid_col], dim=2)  # (B,K,Kp)
            Kp = known.shape[2]
            Tn = Kp + m
            L = Tn + K - 1
            pad = torch.tensor(self.pad_ids, device=device).view(1, K)

            # Region per reordered frame: prefix 1, SUF+suffix 2, MID+middle 3.
            region_frames = torch.full((B, Tn), 3, dtype=torch.long,
                                       device=device)
            if p > 0:
                region_frames[:, :p] = 1
            region_frames[:, p:p + s + 1] = 2  # SUF at p, suffix p+1..p+s

            # Per-column tags, indexed by *column*: entry j is the tag of frame j
            # (the frame that column j predicts), with the ``K - 1`` delay-tail
            # columns repeating the last one -- the layout ``_tag_to_columns``
            # produces at train time.  The BOS column gets frame 0's tag, again
            # like training.
            region_cols = torch.cat([region_frames,
                                     region_frames[:, -1:].expand(B, K - 1)], 1)
            len_cols = None
            if self.decoder.len_emb is not None:
                # Frames left in the span *including* the current frame, i.e.
                # ``Tn - pos``, so ``MID`` at ``Kp - 1`` reports ``m + 1`` and the
                # last middle frame reports 1 -- the convention ``apply_fim``
                # uses.  Before the span the tag is bucket 0 ("no span").
                pos = torch.arange(Tn, device=device).view(1, Tn)
                remaining = torch.where(pos >= (Kp - 1), Tn - pos,
                                        torch.zeros_like(pos))
                len_frames = self._len_buckets(remaining.expand(B, Tn),
                                               self.n_len_buckets)
                len_cols = torch.cat([len_frames,
                                      len_frames[:, -1:].expand(B, K - 1)], 1)

            seq = self._pad_column(B, device)
            cols = []
            ar = torch.arange(K, device=device)
            for i in range(L):
                frame = i - ar
                if i < Kp:
                    # Pure prefill column (known frames only): no decoder call.
                    fk = frame.clamp(0, Kp - 1)
                    col = known[:, ar, fk]
                    valid = (frame >= 0) & (frame < Kp)
                    col = torch.where(valid.view(1, K), col, pad)
                else:
                    n_cols = seq.shape[2]      # 1 (BOS) + i columns so far
                    logits = self.decoder(
                        seq, memory, memory_mask, sid=sid,
                        region=region_cols[:, :n_cols],
                        length=None if len_cols is None else len_cols[:, :n_cols],
                        hint_vec=dec_hint)
                    col = []
                    for k in range(K):
                        lg = logits[k][:, -1] / max(temperature, 1e-6)
                        lg = lg.clone()
                        lg[..., self.pad_ids[k]] = float("-inf")
                        lg[..., self.suf_ids[k]] = float("-inf")
                        lg[..., self.mid_ids[k]] = float("-inf")
                        if self.eos_token:
                            lg[..., self.eos_ids[k]] = float("-inf")
                        if top_k is not None:
                            kth = torch.topk(lg, min(top_k, lg.size(-1)),
                                             dim=-1).values[..., -1:]
                            lg = torch.where(lg < kth,
                                             torch.full_like(lg, float("-inf")), lg)
                        probs = F.softmax(lg, dim=-1)
                        col.append(torch.multinomial(probs, num_samples=1).squeeze(-1))
                    col = torch.stack(col, dim=1)
                    # Keep known tokens where this column still touches the
                    # prefix/suffix (delay boundary).
                    is_known = (frame >= 0) & (frame < Kp)
                    fk = frame.clamp(0, Kp - 1)
                    known_tok = known[:, ar, fk]
                    col = torch.where(is_known.view(1, K), known_tok, col)
                    valid = (frame >= 0) & (frame < Tn)
                    col = torch.where(valid.view(1, K), col, pad)
                cols.append(col)
                seq = torch.cat([seq, col.unsqueeze(-1)], dim=2)

            cols = torch.stack(cols, dim=2)  # (B, K, L)
            out = cols.new_zeros((B, K, m))
            for k in range(K):
                out[:, k, :] = cols[:, k, Kp + k : Kp + k + m]
            return out
        finally:
            if was_training:
                self.train()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> str:
        n = self.num_parameters()
        spk = ("encoder+decoder" if self.decoder_speaker_cond else "encoder only")
        if self.hints:
            hints = (f", hints={self.n_intensity_levels} int/{self.n_pitch_levels}"
                     f" pitch/{self.n_register_levels} reg/{self.n_slot_buckets}"
                     f" slot+sem{self.semantic_dim}"
                     + (" (decoder too)" if self.decoder_hints else " (encoder only)"))
        else:
            hints = ", hints=off"
        return (
            f"CodeTransformer(hidden={self.hidden_size}, codebooks="
            f"{self.spec_codebook_num} spec + {self.pitch_codebook_num} pitch, "
            f"speaker={spk}, memory_norm={self.memory_norm}, "
            f"residual_scale={self.residual_scale}, eos={self.eos_token}, "
            f"fim={self.fim}, region_cond={self.region_cond}, "
            f"length_cond={self.length_cond}"
            + (f" (n_len_buckets={self.n_len_buckets})"
               if self.length_cond else "")
            + hints
            + f", params={n / 1e6:.2f}M)"
        )


def expand_state_dict_with_special(model: "CodeTransformer", state_dict: dict,
                                   prefix: str = "net.") -> dict:
    """Grow a checkpoint's codebook weights to this model's class count.

    Enabling ``transformer.eos_token`` and/or ``transformer.fim`` widens every
    codebook embedding and output head (and each head's bias), so
    :func:`commons.load_state_dict_mismatch` would *skip the whole tensor* when
    warm-starting from a narrower checkpoint -- silently re-initialising all
    input embeddings and output heads, i.e. much of what the stage-2 model knows
    about codes.

    This returns a copy of ``state_dict`` whose matching entries are zero-padded
    to ``model.decoder.num_classes[k]`` (one row per new special class: EOS,
    SUF, MID), so the old rows load and only the new classes are fresh.  Zero
    (not a copy of another row) is deliberate: the new logit then starts at the
    head's own bias, i.e. 0, an unremarkable ``1/num_classes`` probability
    instead of inheriting the pad class's logit (which would make a fresh model
    stop/sentinel at frame 0).  A checkpoint *wider* than the model (e.g. EOS/FIM
    going the other way) is left untouched; ``check_special_layout`` in
    ``2_training.py`` reports it.

    Args:
        model: the model being loaded into.
        state_dict: checkpoint state dict, with or without ``prefix``.
        prefix: key prefix used by the checkpoint (``"net."`` for Lightning
            stage-2 checkpoints, ``""`` for a bare ``CodeTransformer``).

    Returns:
        A new dict (the input when nothing matches, so a stale/other checkpoint
        still loads).
    """
    out = dict(state_dict)
    num_classes = model.decoder.num_classes
    changed = False
    for k, vocab in enumerate(model.vocab_sizes):
        target = int(num_classes[k])
        names = (f"embeds.{k}.weight", f"heads.{k}.weight", f"heads.{k}.bias")
        for name in names:
            for key in (prefix + f"decoder.{name}", f"decoder.{name}"):
                src = out.get(key)
                if src is None or src.shape[0] >= target:
                    continue
                grown = src.new_zeros((target, *src.shape[1:]))
                grown[: src.shape[0]] = src
                out[key] = grown
                changed = True
    return out if changed else state_dict


# Backwards-compatible alias (the pre-FIM name, still used by older scripts).
expand_state_dict_with_eos = expand_state_dict_with_special


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
