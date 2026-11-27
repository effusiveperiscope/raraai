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
``adapt_state_dict_special``), so it needs a fresh run or a
``--transfer_from`` warm start -- a ``--resume_from`` of an EOS-less checkpoint
cannot work.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:                                    # repo root on sys.path (training/tests)
    from hints.levels import (DEFAULT_PITCH_EDGES, DEFAULT_SLOT_BUCKETS,
                              load_levels as load_hint_levels,
                              load_pitch_anchors, load_word_bert_dim,
                              load_word_bert_model)
except ImportError:                     # `python modeling/transformer.py`
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(
        _os.path.abspath(__file__))))
    from hints.levels import (DEFAULT_PITCH_EDGES, DEFAULT_SLOT_BUCKETS,
                              load_levels as load_hint_levels,
                              load_pitch_anchors, load_word_bert_dim,
                              load_word_bert_model)

try:                                    # LoRA fine-tuning (--finetune)
    from modeling.lora import inject_lora, qkv_delta
except ImportError:                     # `python modeling/transformer.py`
    from lora import inject_lora, qkv_delta


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


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """``[x1, x2] -> [-x2, x1]`` over the last (head-dim) axis."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary position embedding (RoPE) applied inside attention.

    Unlike :class:`SinusoidalPositionalEncoding` this is *not* added to the
    embeddings: it rotates the per-head queries and keys, so the attention
    score depends on the **relative** offset of the two positions and there is
    no lookup table whose rows the model may never have trained on.  That is
    exactly the failure this exists to remove: the absolute encoding made every
    decoder column past ``train.max_frames + K - 1`` effectively a different
    function (measured: CE 2.0 -> 3.6 nats and text conditioning -> 0 at the
    first unseen column, and shifting the table by +128 on a *short* item alone
    reproduced it).

    No parameters and only a non-persistent ``inv_freq`` buffer, so the
    checkpoint layout is unchanged.
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim % 2:
            raise ValueError(f"RoPE needs an even head dim, got {dim}")
        inv_freq = 1.0 / (
            float(base) ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor):
        """``x``: (B, H, L, D) -> ``(cos, sin)`` each (1, 1, L, D)."""
        length = x.size(-2)
        pos = torch.arange(length, device=x.device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq.to(x.device))  # (L, D/2)
        emb = torch.cat((freqs, freqs), dim=-1)               # (L, D)
        return (emb.cos().to(x.dtype)[None, None, :, :],
                emb.sin().to(x.dtype)[None, None, :, :])


class RotaryMultiheadAttention(nn.MultiheadAttention):
    """``nn.MultiheadAttention`` with RoPE applied to the queries and keys.

    Kept as a subclass so the state dict stays *exactly* the parent's
    (``in_proj_weight`` / ``in_proj_bias`` / ``out_proj.*``): a checkpoint
    written by the sinusoidal model loads into this one with no key or shape
    change, which is what makes a warm start (``--transfer_from``) possible.

    The attention itself is computed explicitly (projection -> per-head split ->
    RoPE -> SDPA), because the rotary transform has to sit between the
    projection and the score.  ``_qkv_same_embed_dim`` is forced to ``False``
    only to stop ``TransformerEncoderLayer``'s fused eval fast path, which
    bypasses ``forward`` entirely and would silently drop the rotation outside
    training.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0,
                 rope_base: float = 10000.0, **kwargs):
        kwargs.setdefault("batch_first", True)
        super().__init__(embed_dim, num_heads, dropout=dropout, **kwargs)
        if embed_dim % num_heads:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads "
                f"({num_heads})")
        self.rope = RotaryEmbedding(embed_dim // num_heads, base=rope_base)
        self._qkv_same_embed_dim = False  # see the class docstring

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask=None,
        need_weights: bool = True,
        attn_mask=None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        if not self.batch_first:
            query, key, value = (t.transpose(0, 1) for t in (query, key, value))
        B, L, E = query.shape
        S = key.shape[1]
        H = self.num_heads
        D = E // H
        dev, dt = query.device, query.dtype

        w, b = self.in_proj_weight, self.in_proj_bias
        q = F.linear(query, w[:E], None if b is None else b[:E])
        k = F.linear(key, w[E:2 * E], None if b is None else b[E:2 * E])
        v = F.linear(value, w[2 * E:], None if b is None else b[2 * E:])
        # LoRA qkv adapter (``--finetune``, see modeling/lora.py).  Only this
        # class gets one: the stock ``nn.MultiheadAttention`` used by the
        # encoder and the decoder cross-attention reads its fused weight inside
        # ``F.multi_head_attention_forward``, so adapting it would mean copying
        # torch internals.  Absent on a normal run.
        if getattr(self, "lora_qkv_A", None) is not None:
            dq, dk, dv = qkv_delta(self, query, key, value)
            q, k, v = q + dq, k + dk, v + dv
        q = q.view(B, L, H, D).transpose(1, 2)
        k = k.view(B, S, H, D).transpose(1, 2)
        v = v.view(B, S, H, D).transpose(1, 2)

        cos, sin = self.rope(q)
        q = q * cos + _rotate_half(q) * sin
        cos, sin = self.rope(k)
        k = k * cos + _rotate_half(k) * sin

        mask = attn_mask
        if is_causal and mask is None:
            mask = torch.full((L, S), float("-inf"), device=dev, dtype=dt)
            mask = torch.triu(mask, diagonal=1)
        if mask is not None and mask.dtype == torch.bool:
            # SDPA convention: True == allowed to attend.
            mask = torch.zeros((1, 1, L, S), device=dev, dtype=dt).masked_fill(
                ~mask.view(1, 1, L, S), float("-inf"))
        if key_padding_mask is not None:
            pad = torch.zeros((B, 1, 1, S), device=dev, dtype=dt).masked_fill(
                key_padding_mask.to(torch.bool).view(B, 1, 1, S), float("-inf"))
            mask = pad if mask is None else mask.view(1, 1, L, S) + pad

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=float(self.dropout) if self.training else 0.0,
            is_causal=bool(is_causal and mask is None),
        )
        out = out.transpose(1, 2).reshape(B, L, E)
        out = self.out_proj(out)

        weights = None
        if need_weights:
            score = (q @ k.transpose(-2, -1)) / math.sqrt(D)
            if mask is not None:
                score = score + mask
            weights = torch.softmax(score.float(), dim=-1).to(dt)
            if average_attn_weights:
                weights = weights.mean(dim=1)
        if not self.batch_first:
            out = out.transpose(0, 1)
        return out, weights


def uses_rope(state_dict) -> bool:
    """Whether a checkpoint was trained with ``transformer.rope: true``.

    RoPE adds no parameters, so the persistent ``rope_marker`` buffer is the
    only trace it leaves in ``state_dict``.  Accepts both raw and ``net.``-
    prefixed keys.
    """
    return any(k == "rope_marker" or k.endswith(".rope_marker")
               for k in state_dict)


class _OmitHints(dict):
    """Marker for "pass **no** hint vectors at all" (the pre-Phase-0 path).

    It has to be an object rather than ``None`` because ``None`` means "the
    trained null state" everywhere else (see
    :meth:`CodeTransformer.resolve_hints`); collapsing the two turned the
    ``omit`` diagnostic arm into a copy of the ``null`` one, which hid the very
    difference it exists to measure.  Being an empty ``dict`` makes
    ``_enc_hints``/``_dec_hint`` do the right thing for free: no ``hint_int``
    key -> no vectors.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<omit hints>"


#: The singleton returned by ``resolve_hints("omit")``.
OMIT_HINTS = _OmitHints()


class SpeakerEmbedding(nn.Module):
    """Low-rank / regularised speaker conditioning (optional).

    The default model uses a full ``nn.Embedding(n_speakers, hidden)`` per site.
    With this module (``transformer.spk_rank > 0``) each speaker instead owns a
    short code ``(rank,)`` and a **shared** ``rank -> hidden`` projection turns
    it into the conditioning vector.  The projection is trained by every
    speaker, so a 1-5-clip speaker only has to learn ``rank`` numbers (16-32)
    rather than 768, with the structure supplied by the shared projection --
    this is the parameter-sharing mechanism the per-id table lacks.

    ``spk_norm: l2`` normalises the code to ``sqrt(rank)`` (directions only, no
    free norm); ``spk_init: zero`` starts every speaker at the neutral vector so
    rare ids begin at the population prior instead of a random direction.

    With ``spk_kl_weight > 0`` a per-speaker ``logvar`` table is added and the
    code is reparameterised (``z = mu + eps * exp(logvar/2)``) during training;
    ``kl`` is the standard Gaussian KL.  At inference the mean is used.  This is
    the "smooth latent" option: rare speakers stay at the prior (zero code),
    high-data speakers move, and no discrete lookup can jump between unrelated
    directions.
    """

    def __init__(self, n_speakers: int, hidden_size: int, rank: int,
                 norm: str = "none", init: str = "default",
                 kl_weight: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("SpeakerEmbedding needs rank > 0")
        if norm not in ("none", "l2"):
            raise ValueError(f"spk_norm must be none|l2, got {norm!r}")
        if init not in ("default", "zero"):
            raise ValueError(f"spk_init must be default|zero, got {init!r}")
        self.rank = int(rank)
        self.norm = norm
        self.code = nn.Embedding(n_speakers, self.rank)
        self.proj = nn.Linear(self.rank, hidden_size, bias=False)
        if init == "zero":
            nn.init.zeros_(self.code.weight)
        self.logvar = (nn.Embedding(n_speakers, self.rank)
                       if kl_weight > 0 else None)
        if self.logvar is not None:
            nn.init.zeros_(self.logvar.weight)

    def vector(self, sid: torch.Tensor, stochastic: bool = True) -> torch.Tensor:
        """``(B,)`` ids -> ``(B, hidden)`` conditioning vectors."""
        mu = self.code(sid)
        if self.logvar is not None and stochastic and self.training:
            std = torch.exp(0.5 * self.logvar(sid))
            mu = mu + std * torch.randn_like(mu)
        if self.norm == "l2":
            mu = F.normalize(mu, dim=-1) * math.sqrt(self.rank)
        return self.proj(mu)

    def kl(self, sid: torch.Tensor) -> torch.Tensor:
        """Gaussian KL for the ids in ``sid`` (0 when KL is disabled)."""
        if self.logvar is None:
            return torch.zeros((), device=sid.device)
        mu = self.code(sid)
        logvar = self.logvar(sid)
        return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(-1).mean()


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
        hint_film: bool = False,
        spk_rank: int = 0,
        spk_norm: str = "none",
        spk_init: str = "default",
        spk_kl_weight: float = 0.0,
        spk_film: bool = False,
    ):
        super().__init__()
        if memory_norm not in ("temporal", "layer", "none"):
            raise ValueError(
                "transformer.memory_norm must be 'temporal', 'layer' or "
                f"'none', got {memory_norm!r}")
        self.memory_norm = str(memory_norm)
        self.memory_norm_eps = 1e-5

        self.phone_emb = nn.Embedding(phone_vocab_size, hidden_size)
        if spk_rank > 0:
            self.spk_lowrank = SpeakerEmbedding(
                n_speakers, hidden_size, spk_rank, norm=spk_norm,
                init=spk_init, kl_weight=spk_kl_weight)
            self.spk_emb = None
        else:
            self.spk_lowrank = None
            self.spk_emb = nn.Embedding(n_speakers, hidden_size)
            if spk_init == "zero":
                nn.init.zeros_(self.spk_emb.weight)
        # Post-norm FiLM of the speaker vector.  The additive path above is
        # added *before* ``_temporal_norm``, whose per-dimension mean
        # subtraction cancels a per-clip constant; this is the injection site
        # that survives (the speaker analogue of ``hint_film_memory``).
        self.spk_film = bool(spk_film)
        if self.spk_film:
            self.spk_film_gamma = nn.Linear(hidden_size, hidden_size)
            self.spk_film_beta = nn.Linear(hidden_size, hidden_size)
            nn.init.zeros_(self.spk_film_gamma.weight)
            nn.init.zeros_(self.spk_film_gamma.bias)
            nn.init.zeros_(self.spk_film_beta.weight)
            nn.init.zeros_(self.spk_film_beta.bias)
        else:
            self.spk_film_gamma = None
            self.spk_film_beta = None
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

        # Post-normalisation FiLM of the clip-level hints
        # (``transformer.hint_film_memory``).  ``_temporal_norm`` standardises
        # every dimension over *time*, which is exactly invariant to adding a
        # per-clip constant (and to a FiLM that is computed from one): the mean
        # subtraction cancels it.  That is why the additive clip-level channels
        # measured ~0 nats -- intensity 0 -> 4 moved the pre-norm residual by
        # 4.0 % and its CE by 0.0000.  Applying the modulation *after* the norm
        # puts it beyond the cancellation point, where the decoder sees it.
        # Zero-initialised, so a ``--transfer_from`` model reproduces its source
        # exactly and the two projections must earn their effect.
        if hint_film:
            self.film_gamma = nn.Linear(hidden_size, hidden_size)
            self.film_beta = nn.Linear(hidden_size, hidden_size)
            for lin in (self.film_gamma, self.film_beta):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
        else:
            self.film_gamma = None
            self.film_beta = None

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

    def _spk_vector(self, sid: torch.Tensor) -> torch.Tensor:
        """``(B,)`` ids -> ``(B, H)`` (low-rank + optional VAE sampling)."""
        if self.spk_lowrank is not None:
            return self.spk_lowrank.vector(sid)
        return self.spk_emb(sid.to(sid.device))

    def spk_kl(self, sid: torch.Tensor) -> torch.Tensor:
        if self.spk_lowrank is None:
            return torch.zeros((), device=sid.device)
        return self.spk_lowrank.kl(sid)

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

        With ``hint_film`` the clip-level hints are *also* applied as a
        post-normalisation FiLM on the returned memory (see ``__init__``), which
        is the only place a per-clip constant can survive ``_temporal_norm``.
        """
        B, N = phone_ids.shape
        if phone_lengths is None:
            phone_lengths = torch.full(
                (B,), N, device=phone_ids.device, dtype=torch.long
            )
        phone_lengths = phone_lengths.to(phone_ids.device).clamp(min=1, max=N)
        valid = torch.arange(N, device=phone_ids.device)[None, :] < phone_lengths[:, None]

        x = self.phone_emb(phone_ids)
        spk = self._spk_vector(sid.to(phone_ids.device))
        x = x + spk.unsqueeze(1)
        if hint_vec is not None:
            x = x + hint_vec.to(x.dtype).unsqueeze(1)
        if pos_hint is not None:
            x = x + pos_hint.to(x.dtype)[:, :N]
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=~valid)
        x = x * valid.unsqueeze(-1).to(x.dtype)
        if self.memory_norm == "temporal":
            x = self._temporal_norm(x, valid)
        if self.spk_film:
            # Post-norm, so the per-clip speaker vector is not removed by the
            # temporal standardisation (zero-init: identity at start).
            x = (x * (1.0 + self.spk_film_gamma(spk).unsqueeze(1))
                 + self.spk_film_beta(spk).unsqueeze(1))
            x = x * valid.unsqueeze(-1).to(x.dtype)
        if self.film_gamma is not None and hint_vec is not None:
            h = hint_vec.to(x.dtype)
            x = (x * (1.0 + self.film_gamma(h).unsqueeze(1))
                 + self.film_beta(h).unsqueeze(1))
            x = x * valid.unsqueeze(-1).to(x.dtype)
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
        hint_postnorm: bool = False,
        rope: bool = False,
        rope_base: float = 10000.0,
        input_dropout: float = 0.0,
        spk_rank: int = 0,
        spk_norm: str = "none",
        spk_init: str = "default",
        spk_kl_weight: float = 0.0,
    ):
        super().__init__()
        self.vocab_sizes = list(vocab_sizes)
        self.num_codebooks = len(self.vocab_sizes)
        self.hidden_size = hidden_size
        self.speaker_cond = bool(speaker_cond)
        # Extra per-codebook classes: ``vocab_size`` is pad/BOS, ``+1`` is EOS
        # (``transformer.eos_token``).
        self.eos = bool(eos)
        self.num_classes = [v + 1 + int(self.eos) for v in self.vocab_sizes]
        # Training-only input-token dropout (``transformer.input_dropout``).
        # A plain float, so ``state_dict`` is unchanged and a checkpoint can be
        # resumed with the knob on or off (it changes the *function* during
        # training, like ``dropout``, not the parameter set).
        self.input_dropout = float(input_dropout)

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
            if spk_rank > 0:
                self.spk_lowrank = SpeakerEmbedding(
                    n_speakers, hidden_size, spk_rank, norm=spk_norm,
                    init=spk_init, kl_weight=spk_kl_weight)
                self.spk_emb = None
            else:
                self.spk_lowrank = None
                self.spk_emb = nn.Embedding(n_speakers, hidden_size)
                if spk_init == "zero":
                    nn.init.zeros_(self.spk_emb.weight)
        else:
            self.spk_emb = None
            self.spk_lowrank = None

        self.embeds = nn.ModuleList(
            [nn.Embedding(c, hidden_size) for c in self.num_classes]
        )
        # Constant decoder-input bias.  Pre-FIM checkpoints trained with
        # `region_cond`/`length_cond` added `region_emb[0] + len_emb[0]` to every
        # AR column; those tables are gone now, so `adapt_state_dict_special`
        # folds their row 0 into this one vector.  It is not FIM functionality:
        # it is just the per-position bias the old model learned for plain AR.
        self.input_bias = nn.Parameter(torch.zeros(hidden_size))
        # Post-norm hint projection (``transformer.hint_decoder_proj``): the
        # clip-level hints are added to the *output* of the final decoder norm
        # instead of (only) to the input.  The input path runs the hint through
        # twelve pre-LN residual blocks, where a constant is quickly washed out
        # (measured: the semantic vector moved the logits by 2.6x their std and
        # changed the CE by -0.002 nats, i.e. a direction the heads map to a
        # near-uniform offset).  Rank-H, shared by all codebooks, zero-init ->
        # identity at init.  See ``hint_film_memory`` for the encoder-side twin.
        if hint_postnorm:
            self.hint_out = nn.Linear(hidden_size, hidden_size)
            nn.init.zeros_(self.hint_out.weight)
            nn.init.zeros_(self.hint_out.bias)
        else:
            self.hint_out = None
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
        # RoPE (``transformer.rope``): replace the decoder's *self*-attention
        # with the rotary variant and stop adding the sinusoidal table.  The
        # cross-attention is deliberately left alone -- the encoder memory
        # carries the (absolute) phone positions, and a relative decoder->phone
        # rotation would not mean "this phone at this time".  Done before the
        # residual scaling below, so the new ``out_proj`` gets the same init
        # scaling as every other layer.  No parameters are added: the subclass
        # keeps ``nn.MultiheadAttention``'s state-dict keys exactly.
        self.rope = bool(rope)
        self.rope_base = float(rope_base)
        if self.rope:
            for dec_layer in self.decoder.layers:
                dec_layer.self_attn = RotaryMultiheadAttention(
                    hidden_size, n_heads, dropout=dropout,
                    rope_base=self.rope_base)
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

    def _spk_vector(self, sid: torch.Tensor) -> torch.Tensor:
        if self.spk_lowrank is not None:
            return self.spk_lowrank.vector(sid)
        return self.spk_emb(sid)

    def spk_kl(self, sid: torch.Tensor) -> torch.Tensor:
        if self.spk_lowrank is None:
            return torch.zeros((), device=sid.device)
        return self.spk_lowrank.kl(sid)

    def _forward_impl(
        self,
        codes: torch.Tensor,  # (B, K, L) long, pad index == vocab size
        memory: torch.Tensor,  # (B, N, H)
        memory_mask: Optional[torch.Tensor] = None,  # (B, N) bool, True == valid
        sid: Optional[torch.Tensor] = None,  # (B,) long
        hint_vec: Optional[torch.Tensor] = None,  # (B, H) clip-level hints
        use_mask_cache: bool = True,
        return_hidden: bool = False,
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
        # Input-token dropout (``transformer.input_dropout``): zero the code
        # embedding of a random subset of positions, so the next-token target
        # has to be predicted from a *partially missing* history.  This is the
        # cheap analogue of scheduled sampling -- no sampling loop, no new
        # parameters, no state_dict change -- and it targets the failure mode a
        # purely teacher-forced AR model has in free-running decoding: one bad
        # frame is an input the model has never seen.  Only the code content is
        # dropped; position, speaker and the hint vectors are added afterwards
        # and stay intact.  Column 0 is the start-of-sequence column (always
        # real at inference), so it is never masked.
        if self.input_dropout > 0.0 and self.training:
            keep_pos = torch.arange(L, device=x.device).view(1, L, 1) > 0
            drop = (torch.rand(B, L, 1, device=x.device) < self.input_dropout)
            x = x.masked_fill(drop & keep_pos, 0.0)
        if not self.rope:
            # With RoPE the position information lives inside self-attention;
            # adding the absolute table as well would reintroduce exactly the
            # unseen-position cliff RoPE exists to remove.
            x = self.pos(x)
        if self.spk_emb is not None or self.spk_lowrank is not None:
            if sid is None:
                raise ValueError(
                    "DelayCodebookDecoder was built with speaker_cond=True "
                    "but no sid was passed")
            x = x + self._spk_vector(sid.to(codes.device)).unsqueeze(1)
        x = x + self.input_bias.to(x.dtype)

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
        # Post-norm hint injection (beyond the norm, so nothing can cancel it).
        if self.hint_out is not None and hint_vec is not None:
            out = out + self.hint_out(hint_vec.to(out.dtype)).unsqueeze(1)
        logits = [head(out) for head in self.heads]
        if return_hidden:
            # Used by the word-progress auxiliary loss, which needs the shared
            # pre-head state rather than the per-codebook logits.
            return logits, out
        return logits


    def forward(
        self,
        codes: torch.Tensor,  # (B, K, L) long, pad index == vocab size
        memory: torch.Tensor,  # (B, N, H)
        memory_mask: Optional[torch.Tensor] = None,  # (B, N) bool, True == valid
        sid: Optional[torch.Tensor] = None,  # (B,) long
        hint_vec: Optional[torch.Tensor] = None,  # (B, H) clip-level hints
    ) -> List[torch.Tensor]:
        return self._forward_impl(codes, memory, memory_mask, sid,
                                  hint_vec, use_mask_cache=True)

    def forward_with_hidden(
        self,
        codes: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
        sid: Optional[torch.Tensor] = None,
        hint_vec: Optional[torch.Tensor] = None,
    ):
        """Decoder logits plus the shared pre-head hidden state.

        Only the word-progress auxiliary loss uses this.  It deliberately calls
        ``_forward_impl`` directly (with the eager mask cache), so it stays
        correct when ``train.compile_decoder`` has replaced ``forward`` with a
        compiled wrapper; the new config leaves that off anyway.
        """
        return self._forward_impl(codes, memory, memory_mask, sid,
                                  hint_vec, use_mask_cache=True,
                                  return_hidden=True)


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
        # Rotary position embedding in the *decoder self-attention* (see
        # ``RotaryMultiheadAttention``).  It replaces the sinusoidal table
        # instead of adding to it, adds no parameters (checkpoint keys are
        # unchanged), and is recorded by a ``rope_marker`` buffer so a
        # checkpoint can be matched to the flag: a model trained with RoPE
        # loaded into a sinusoidal build (or vice versa) would silently score
        # every position differently while every tensor shape still matches.
        rope = bool(_get_cfg(tcfg, "rope", False))
        rope_base = float(_get_cfg(tcfg, "rope_base", 10000.0) or 10000.0)
        # Training-only input-token dropout on the decoder (see
        # ``DelayCodebookDecoder._forward_impl``).  Zero keeps the historical
        # teacher-forced behaviour bit-identically; it adds no parameters, so it
        # needs neither a fresh ``exp_name`` nor re-preprocessing.
        input_dropout = float(_get_cfg(tcfg, "input_dropout", 0.0) or 0.0)
        if not 0.0 <= input_dropout < 1.0:
            raise ValueError(
                "transformer.input_dropout must be in [0, 1), got "
                f"{input_dropout}")

        # Speaker-embedding structure (optional; see SpeakerEmbedding).
        # Defaults reproduce the historical full per-id table exactly, so a
        # checkpoint written before these keys existed still loads.
        spk_rank = max(int(_get_cfg(tcfg, "spk_rank", 0) or 0), 0)
        spk_norm = str(_get_cfg(tcfg, "spk_norm", "none") or "none").lower()
        spk_init = str(_get_cfg(tcfg, "spk_init", "default")
                       or "default").lower()
        spk_kl_weight = float(_get_cfg(tcfg, "spk_kl_weight", 0.0) or 0.0)
        spk_film = bool(_get_cfg(tcfg, "spk_film_memory", False))
        if spk_norm not in ("none", "l2"):
            raise ValueError(
                f"transformer.spk_norm must be none|l2, got {spk_norm!r}")
        if spk_init not in ("default", "zero"):
            raise ValueError("transformer.spk_init must be default|zero, got "
                             f"{spk_init!r}")
        if spk_kl_weight > 0 and spk_rank <= 0:
            raise ValueError(
                "transformer.spk_kl_weight needs transformer.spk_rank > 0 "
                "(the Gaussian prior is on the low-rank speaker code)")
        if spk_norm != "none" and spk_rank <= 0:
            raise ValueError(
                "transformer.spk_norm needs transformer.spk_rank > 0 (the "
                "full per-id table has no separate code to normalise)")
        self.spk_kl_weight = spk_kl_weight
        self.spk_layout = (spk_rank, spk_norm, spk_init, spk_film,
                           spk_kl_weight > 0)
        # A parameter-free marker (like rope_marker) only for non-default
        # layouts: default checkpoints keep exactly their old key set, while a
        # structured run records the contract for check_special_layout.
        if any((spk_rank > 0, spk_norm != "none", spk_init != "default",
                spk_film, spk_kl_weight > 0)):
            self.register_buffer(
                "spk_layout_marker",
                torch.tensor([float(spk_rank),
                              {"none": 0.0, "l2": 1.0}[spk_norm],
                              {"default": 0.0, "zero": 1.0}[spk_init],
                              float(spk_film), float(spk_kl_weight > 0)]))

        # Coarse conditioning hints (v0.8).  The tables are built below; the
        # level counts come from the *same* ``hints.json`` the dataset used, so
        # a level id means the same thing on both sides by construction.
        hints_on = bool(_get_cfg(tcfg, "hints", False))
        hints_json = _get_cfg(tcfg, "hints_json", None)
        decoder_hints = bool(_get_cfg(tcfg, "decoder_hints", True))
        # *Where* the clip-level channels are injected (Phase 2).  The additive
        # path above ``_temporal_norm`` cannot express a per-clip constant, so
        # these two switches add a post-norm path on each side of the model:
        #   hint_film_memory  -> PhonemeEncoder, ``mem*(1+gamma(h)) + beta(h)``
        #                        applied *after* the temporal norm
        #   hint_decoder_proj -> DelayCodebookDecoder, ``out + W h`` applied
        #                        *after* the final decoder norm, before the heads
        # Both zero-initialise (identity at init), so a ``--transfer_from`` model
        # starts as an exact copy of its source.
        hint_film = bool(_get_cfg(tcfg, "hint_film_memory", False))
        hint_postnorm = bool(_get_cfg(tcfg, "hint_decoder_proj", False))
        # Classifier-free guidance over the hint channels, at generation time
        # only (see ``CodeTransformer.generate``).  ``None``/1.0 = off.
        self.hint_guidance = self._parse_guidance(
            _get_cfg(tcfg, "hint_guidance", None))
        self.hint_guidance_books = self._parse_guidance_books(
            _get_cfg(tcfg, "hint_guidance_books", None))

        pitch_edges = list(_get_cfg(tcfg, "hint_pitch_edges", None)
                           or DEFAULT_PITCH_EDGES)
        n_slot_buckets = int(_get_cfg(tcfg, "n_slot_buckets",
                                      DEFAULT_SLOT_BUCKETS)
                             or DEFAULT_SLOT_BUCKETS)
        # Register parameterisation.  ``hints/build.py --relative-register``
        # stores one semitone offset per speaker in ``hints.json`` and fits the
        # edges on the residual, which makes the channel orthogonal to ``sid``
        # (HINTS_PLAN.md §8.5).  The *switch* is the ``speaker_means`` block in
        # that file -- the dataset reads it from there -- and this flag exists
        # only to assert that the config agrees, because a level id changing
        # meaning is invisible to the row-count check in
        # ``check_special_layout`` (same table width either way).
        relative_register = bool(_get_cfg(tcfg, "relative_register", False))

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

        # Duration / word-progress auxiliary loss.  A small head on the decoder
        # hidden state predicts the normalised position of the word currently
        # being spoken; the target comes from the forced-alignment word timings.
        # The monotonicity term penalises going *backwards* in the sentence,
        # which is the "skips a word / repeats a word" failure mode, while the
        # regression term penalises pacing drift (stretched or compressed
        # words).  The head is a separate parameter set, so a warm start from a
        # checkpoint without it works through ``--transfer_from`` (the head is
        # freshly initialised and only the auxiliary loss uses it).
        self.word_progress = bool(_get_cfg(tcfg, "word_progress", False))
        self.word_progress_weight = float(
            _get_cfg(tcfg, "word_progress_weight", 0.1) or 0.0)
        self.word_progress_mono_weight = float(
            _get_cfg(tcfg, "word_progress_mono_weight", 0.02) or 0.0)
        self.word_progress_margin = float(
            _get_cfg(tcfg, "word_progress_margin", 0.0) or 0.0)
        if self.word_progress and self.word_progress_weight < 0:
            raise ValueError(
                "transformer.word_progress_weight must be >= 0")

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
            hint_film=hint_film,
            spk_rank=spk_rank,
            spk_norm=spk_norm,
            spk_init=spk_init,
            spk_kl_weight=spk_kl_weight,
            spk_film=spk_film,
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
            hint_postnorm=hint_postnorm,
            rope=rope,
            rope_base=rope_base,
            input_dropout=input_dropout,
            spk_rank=spk_rank,
            spk_norm=spk_norm,
            spk_init=spk_init,
            spk_kl_weight=spk_kl_weight,
        )
        self.memory_norm = memory_norm
        self.residual_scale = residual_scale
        self.hint_film_memory = hint_film
        self.hint_decoder_proj = hint_postnorm
        self.input_dropout = input_dropout
        self.rope = rope
        self.rope_base = rope_base
        # Word-progress head (``transformer.word_progress``): one scalar per
        # code frame, read from the same decoder hidden state the codebook heads
        # use.  A plain linear layer is enough -- the target is a smooth
        # monotone curve and the decoder already carries the timing state.
        if self.word_progress:
            self.word_progress_head = nn.Linear(hidden_size, 1)
            # Small init keeps the auxiliary loss from perturbing the pretrained
            # decoder before it has learned the target.
            nn.init.normal_(self.word_progress_head.weight, std=0.02)
            nn.init.zeros_(self.word_progress_head.bias)
        else:
            self.word_progress_head = None
        if self.rope:
            # Persistent marker (a buffer is the only thing a checkpoint can
            # carry for a parameter-free functional flag).  Readers detect it
            # with ``"rope_marker" in state``; see ``stage2_repl`` and
            # ``2_training.check_special_layout``.
            self.register_buffer("rope_marker", torch.ones(1),
                                 persistent=True)

        # ---- LoRA fine-tuning adapters (--finetune) ------------------------ #
        # ``finetune.enabled`` wraps the configured linear layers in
        # ``LoRALinear`` *here*, i.e. at construction time, so every reader
        # (training, REPL, eval scripts, probes) that builds the model from the
        # same config gets the same parameter names.  ``lora_B`` is zero, so an
        # injected model is exactly its source function; see modeling/lora.py
        # and ``2_training.check_special_layout`` for the layout guards.
        # ``finetune`` lives at the config root (next to ``train``), because it
        # is a training mode rather than a model definition -- but a run that
        # *saves* adapters has to keep the flag on to reload them.
        ft_cfg = _get_cfg(config, "finetune", None)
        self.lora_enabled = False
        self.lora_config = None
        if bool(_get_cfg(ft_cfg, "enabled", False)):
            info = inject_lora(self, ft_cfg)
            self.lora_enabled = True
            self.lora_config = dict(info["config"])
            # Persistent marker *and* contract: rank and alpha change no
            # parameter shape (alpha only scales the delta), so a checkpoint
            # whose adapters were trained with another alpha would load
            # completely and silently score a different function.  The values
            # live in the buffer so ``check_special_layout`` can compare them.
            self.register_buffer(
                "lora_marker",
                torch.tensor([float(self.lora_config["rank"]),
                              float(self.lora_config["alpha"]),
                              float(len(self.lora_config["targets"]))]),
                persistent=True)

        # ---- conditioning-hint tables (v0.8) ------------------------------- #
        # ``hint_monotone`` (v1.1) replaces the two *clip-level* level tables
        # with an ordered dial: ``direction * z(level)`` where ``z`` is the level
        # mapped linearly onto [-1, 1], plus a trained "absent" vector for level
        # 0.  The free per-level rows it replaces were measured to stay small and
        # mutually arbitrary (``||int_emb[8] - int_emb[0]|| = 0.61`` in a 768-d
        # space whose memory norm is ~27.7, *unchanged* after 433 k steps), so
        # the "level" was a tiny random perturbation rather than a dial.  An
        # ordered direction is monotone by construction and shares one gradient
        # across the whole range instead of splitting it n ways.  The per-phone
        # channels (pitch/slot) keep their tables -- they are categorical
        # (bins/unvoiced) and they already work.  See HINTS_V1_PLAN.md §5.6.
        self.hint_monotone = bool(_get_cfg(tcfg, "hint_monotone", False))

        # Hints V2 pitch representation.
        #   ``table``   -> v1: one level id per phone, ``(n_pitch_levels+2, H)``
        #                  embedding.  Kept as the default so every pre-V2
        #                  checkpoint stays loadable without editing a config.
        #   ``anchors`` -> up to A ``(value, position)`` anchors per *word*,
        #                  repeated over the word's phones (see
        #                  ``dataset_codes`` and ``hints.levels``).  Chosen by
        #                  ``configs/transformer.yaml``.
        # Measured (Genshin, words >= 8 code frames): one word mean reconstructs
        # the contour to 2.1 st, 3 anchors to 0.7 st, but *projecting* those
        # anchors to one value per phone is worth nothing beyond 2 (1.62 st) --
        # a 320-440 ms word has ~2.5 phones.  Hence the repeat-over-phones
        # vector plus ``hint_phone_frac``.
        self.hint_pitch_mode = str(
            _get_cfg(tcfg, "hint_pitch_mode", "table") or "table").strip().lower()
        if self.hint_pitch_mode not in ("anchors", "table"):
            raise ValueError(
                "transformer.hint_pitch_mode must be 'anchors' or 'table', "
                f"got {self.hint_pitch_mode!r}")
        self.hint_pitch_anchors = max(
            int(_get_cfg(tcfg, "hint_pitch_anchors", 3) or 3), 1)

        # Hints V2.1: per-word BERT states, repeated over the word's phones.
        # The sentence CLS is dead (HINTS_PLAN.md §11), but word-level states
        # do predict word prosody beyond the phones: +0.018 out-of-fold R2 on
        # duration/pitch/range, with the gain still growing at 256 dims -- so
        # the full 768 are stored and *this* projection learns what matters
        # (§12). ``word_dim`` is part of the checkpoint contract (a projection
        # trained on 768-d inputs cannot be fed another width, and a different
        # encoder would make every dimension mean something else).
        self.hint_word_sem = bool(_get_cfg(tcfg, "hint_word_sem", False))
        self.hint_word_dim = int(_get_cfg(tcfg, "hint_word_dim", 768) or 768)

        # Every remaining channel is a small ``(n_levels + 1, hidden)``
        # embedding indexed by a level id, row 0 being the trained "not
        # provided" state (the dataset applies per-channel dropout, so that row
        # is always exercised).
        # They live here rather than on the encoder/decoder so the checkpoint
        # has one obvious place for every hint parameter.
        self.hints = bool(hints_on)
        self.decoder_hints = bool(decoder_hints) and self.hints
        self.hints_json = str(hints_json) if hints_json else None
        self.pitch_edges = [float(e) for e in pitch_edges]
        self.n_slot_buckets = max(int(n_slot_buckets), 1)
        self.relative_register = relative_register
        self.n_intensity_levels = 0
        self.n_register_levels = 0
        # Voiced pitch bins (``len(edges) + 1`` of them, the outer bins
        # saturate); row ``n_pitch_levels + 1`` is "this word is unvoiced" (see
        # ``hints.levels.word_pitch_levels``), hence ``+ 2`` rows.
        #
        # Before Phase 0 the unvoiced class was emitted as ``len(edges) + 1``,
        # which *collides* with the top voiced bin, so the last row never
        # received a gradient (measured on transformer_v0.9: exactly 0.0) and
        # unvoiced words were indistinguishable from ">= +6 st".  The table
        # width was always right; the level id was not.
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
            # The relative-register switch and this flag must agree: the level
            # ids are identical either way, so nothing else can catch it.
            reg_means = (levels.get("register") or {}).get("speaker_means")
            if self.relative_register and not reg_means:
                raise ValueError(
                    f"transformer.relative_register is true, but "
                    f"{self.hints_json} has no levels.register.speaker_means. "
                    f"Re-run `hints/build.py --relative-register "
                    f"--fit-edges-only` on this corpus to fit the speaker "
                    f"means and the residual edges, or set relative_register: "
                    f"false to use the absolute level.")
            if not self.relative_register and reg_means:
                raise ValueError(
                    f"{self.hints_json} is a *relative* register corpus "
                    f"({len(reg_means)} speaker offset(s)) but "
                    f"transformer.relative_register is false: identical level "
                    f"ids would mean different things.  Set "
                    f"transformer.relative_register: true, or point "
                    f"hints_json at an absolute corpus.")
            self.pitch_emb = nn.Embedding(self.n_pitch_levels + 2, hidden_size)
            self.slot_emb = nn.Embedding(self.n_slot_buckets + 1, hidden_size)
            if self.hint_word_sem:
                stored_dim = load_word_bert_dim(self.hints_json)
                if stored_dim is not None and int(stored_dim) != self.hint_word_dim:
                    raise ValueError(
                        f"transformer.hint_word_dim={self.hint_word_dim} but "
                        f"{self.hints_json} was built with word_bert dim "
                        f"{stored_dim}.  Re-run hints/build.py --word-bert "
                        f"(the width is whatever --word-bert-model emits), or "
                        f"match the config to the corpus.")
                self._word_bert_model = load_word_bert_model(self.hints_json)
                self.word_proj = nn.Linear(self.hint_word_dim, hidden_size)
                self.word_null = nn.Parameter(torch.zeros(hidden_size))
            else:
                self._word_bert_model = None
                self.word_proj = None
                self.word_null = None
            if self.hint_pitch_mode == "anchors":
                # The anchor count is part of the checkpoint contract: slot
                # index a means a different point of the contour if it moves.
                stored = load_pitch_anchors(self.hints_json)
                if stored is not None and int(stored) != self.hint_pitch_anchors:
                    raise ValueError(
                        f"transformer.hint_pitch_anchors="
                        f"{self.hint_pitch_anchors} but {self.hints_json} was "
                        f"built with {stored} anchors per word.  Re-run "
                        f"hints/build.py --words --pitch-anchors "
                        f"{self.hint_pitch_anchors}, or match the config to the "
                        f"corpus (the anchor slots would otherwise mean "
                        f"different points of the contour).")
                # One direction per anchor *slot*, so slot index a always means
                # "the a-th point of the word's contour" -- the same ordering
                # the detector emitted.  ``pitch_slot_null`` covers words with
                # fewer than A anchors (so "absent slot" != "0 st"), and
                # ``pitch_absent`` / ``pitch_unvoiced`` are the two trained
                # states the dropout design relies on.
                A = self.hint_pitch_anchors
                self.pitch_emb = None
                self.pitch_val_dir = nn.Parameter(torch.zeros(A, hidden_size))
                self.pitch_pos_dir = nn.Parameter(torch.zeros(A, hidden_size))
                self.pitch_slot_null = nn.Parameter(torch.zeros(A, hidden_size))
                self.pitch_unvoiced = nn.Parameter(torch.zeros(hidden_size))
                self.pitch_absent = nn.Parameter(torch.zeros(hidden_size))
                self.pitch_frac_dir = nn.Parameter(torch.zeros(hidden_size))
            else:
                self.pitch_val_dir = None
                self.pitch_pos_dir = None
                self.pitch_slot_null = None
                self.pitch_unvoiced = None
                self.pitch_absent = None
                self.pitch_frac_dir = None
            if self.hint_monotone:
                self.int_emb = None
                self.reg_emb = None
                self.dec_int_emb = None
                self.int_dir = nn.Parameter(torch.zeros(hidden_size))
                self.int_null = nn.Parameter(torch.zeros(hidden_size))
                self.reg_dir = nn.Parameter(torch.zeros(hidden_size))
                self.reg_null = nn.Parameter(torch.zeros(hidden_size))
                self.dec_int_dir = (nn.Parameter(torch.zeros(hidden_size))
                                    if self.decoder_hints else None)
                self.dec_int_null = (nn.Parameter(torch.zeros(hidden_size))
                                     if self.decoder_hints else None)
            else:
                self.int_dir = self.int_null = None
                self.reg_dir = self.reg_null = None
                self.dec_int_dir = self.dec_int_null = None
                self.int_emb = nn.Embedding(self.n_intensity_levels + 1,
                                            hidden_size)
                self.reg_emb = nn.Embedding(self.n_register_levels + 1,
                                            hidden_size)
                self.dec_int_emb = (nn.Embedding(self.n_intensity_levels + 1,
                                                 hidden_size)
                                    if self.decoder_hints else None)
            # Zero-initialise every hint table.  ``encode`` adds
            # ``int_emb(int) + reg_emb(reg)`` to *every*
            # phoneme, so a randomly-initialised table is a large random
            # perturbation of the encoder memory -- applied to 90 % of training
            # items, at exactly the moment a ``--transfer_from`` model's text
            # alignment is most fragile.  With zeros a warm-started model
            # reproduces the source model *exactly* and the tables have to earn
            # their effect; measured on the v0.8 run, whose teacher-forced loss
            # improved while its free-running WER tripled (HINTS_PLAN.md §9).
            for emb in (self.int_emb, self.reg_emb, self.pitch_emb,
                        self.slot_emb, self.dec_int_emb):
                if emb is not None:
                    nn.init.zeros_(emb.weight)
            if self.word_proj is not None:
                nn.init.zeros_(self.word_proj.weight)
                nn.init.zeros_(self.word_proj.bias)
        else:
            self.int_emb = None
            self.reg_emb = None
            self.pitch_emb = None
            self.slot_emb = None
            self.dec_int_emb = None
            self.int_dir = self.int_null = None
            self.reg_dir = self.reg_null = None
            self.dec_int_dir = self.dec_int_null = None
            self.pitch_val_dir = None
            self.pitch_pos_dir = None
            self.pitch_slot_null = None
            self.pitch_unvoiced = None
            self.pitch_absent = None
            self.pitch_frac_dir = None
            self.word_proj = None
            self.word_null = None
            self._word_bert_model = None

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

    @staticmethod
    def _parse_guidance(raw) -> Optional[float]:
        """``transformer.hint_guidance`` -> a float, or ``None`` (off).

        ``None`` (the key absent) is *off*, not 1.0: 1.0 is the exact identity
        and would still pay for the second (unconditional) decoder pass, so the
        two are mapped together and the cost is skipped entirely.
        """
        if raw is None:
            return None
        value = float(raw)
        if value < 0:
            raise ValueError(
                f"transformer.hint_guidance must be >= 0, got {value}")
        return None if abs(value - 1.0) < 1e-9 else value

    def _parse_guidance_books(self, raw) -> Optional[List[float]]:
        """Per-codebook guidance weights (``transformer.hint_guidance_books``)."""
        if raw is None:
            return None
        weights = [float(w) for w in raw]
        if len(weights) != self.codebook_num:
            raise ValueError(
                f"transformer.hint_guidance_books has {len(weights)} entries "
                f"but there are {self.codebook_num} codebooks "
                f"({self.spec_codebook_num} spec + {self.pitch_codebook_num} "
                f"pitch)")
        if any(w < 0 for w in weights):
            raise ValueError(
                f"transformer.hint_guidance_books must be >= 0, got {weights}")
        return None if all(abs(w - 1.0) < 1e-9 for w in weights) else weights

    def spk_kl(self, sid: torch.Tensor) -> torch.Tensor:
        """Gaussian-prior KL of the speaker code (``transformer.spk_kl_weight``).

        Only non-zero for a low-rank model with ``spk_kl_weight > 0``; the
        encoder and decoder keep separate tables, so both KLs are summed.  Use
        the *mean* over the batch (a per-sample KL would scale with batch size).
        """
        kl = self.encoder.spk_kl(sid) + self.decoder.spk_kl(sid)
        return kl

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
        ``None`` on a hint-enabled model resolves to the **trained null state**
        (all level 0, see :meth:`resolve_hints`), which is what
        every generation path should use; ``"omit"`` reproduces the historical
        "no vectors at all" behaviour for ablations.  On a model without hint
        tables both are a no-op, identical to a pre-v0.8 model.
        """
        hints = self.resolve_hints(hints, phone_ids.shape[0], phone_ids.shape[1],
                                   phone_ids.device)
        hint_vec, pos_hint = self._enc_hints(hints, phone_ids.device)
        return self.encoder(phone_ids, phone_lengths, sid,
                            hint_vec=hint_vec, pos_hint=pos_hint)


    # ------------------------------------------------------------------ #
    # Conditioning hints (v0.8)
    # ------------------------------------------------------------------ #
    def null_hints(self, batch_size: int, n_phone: int, device=None,
                   dtype: torch.dtype = torch.long) -> Optional[dict]:
        """The trained "not provided" state, as a collator-shaped dict.

        Every hint table has a level-0 row (``word_null`` is the semantic
        state's equivalent), and ``train.hint_dropout`` / ``train.hint_null_all``
        exercise exactly this state during training.  **Omitting the vectors instead**
        is a *different* function: measured on the finished
        ``transformer_v0.9`` (step 433 455) it costs +0.136 nats against the
        real hints, of which the trained null state accounts for +0.023 -- i.e.
        0.113 nats are pure out-of-distribution error -- and it changes the
        encoder memory by 52 %.  So generation must send this dict, not
        ``None``.


        Args:
            batch_size, n_phone: leading dimensions of the batch.
            device: where to allocate; defaults to the hint tables' device.

        Returns:
            A dict with the same keys ``dataset_codes.CodeCollator`` produces,
            or ``None`` when the model has no hint tables.
        """
        if not self.hints:
            return None
        if device is None:
            device = (self.pitch_emb.weight.device if self.pitch_emb is not None
                      else self.slot_emb.weight.device)
        dev = device

        def zeros(*shape, dt=dtype):
            return torch.zeros(*shape, dtype=dt, device=dev)

        return {
            "hint_int": zeros(batch_size),
            "hint_reg": zeros(batch_size),
            "hint_pitch": zeros(batch_size, n_phone),
            "hint_slot": zeros(batch_size, n_phone),
            # Hints V2 anchors: state 0 = the trained "absent" path, so these
            # zeros are the null state (not an out-of-distribution input).
            "hint_pitch_val": zeros(batch_size, n_phone,
                                    self.hint_pitch_anchors, dt=torch.float32),
            "hint_pitch_pos": zeros(batch_size, n_phone,
                                    self.hint_pitch_anchors, dt=torch.float32),
            "hint_pitch_n": zeros(batch_size, n_phone),
            "hint_pitch_state": zeros(batch_size, n_phone),
            "hint_phone_frac": zeros(batch_size, n_phone, dt=torch.float32),
            # V2.1: valid=False is the trained "absent" state -> word_null.
            "hint_word": zeros(batch_size, n_phone, self.hint_word_dim,
                               dt=torch.float32),
            "hint_word_valid": zeros(batch_size, n_phone, dt=torch.bool),
        }

    def resolve_hints(self, hints, batch_size: int, n_phone: int, device):
        """Normalise a ``hints`` argument; ``None`` -> the trained null state.

        Accepts ``None`` (or ``"null"`` / ``"zero"``) for the trained "not
        provided" state, ``"omit"`` for the pre-Phase-0 behaviour of passing no
        vectors at all (kept only for ablations -- it is *out of distribution*
        for a dict/batch.  A dict with no ``hint_int`` (a batch from a hints-less
        dataset) also means "no hints available", i.e. the null state, so a
        hint-enabled model never silently falls back to the out-of-distribution
        path -- except for the explicit ``OMIT_HINTS`` marker, which is the
        *deliberate* "no vectors at all".
        """
        if not self.hints:
            return None
        if hints is OMIT_HINTS:
            return hints
        if hints is None:
            return self.null_hints(batch_size, n_phone, device)
        if isinstance(hints, str):
            if hints in ("null", "zero"):
                return self.null_hints(batch_size, n_phone, device)
            if hints == "omit":
                return OMIT_HINTS
            raise ValueError(
                f"unknown hints mode {hints!r}; expected 'null'/'zero' (the "
                f"trained null state), 'omit' (no vectors at all), or a hint "
                f"dict")
        if isinstance(hints, dict) and hints.get("hint_int") is None:
            return self.null_hints(batch_size, n_phone, device)
        return hints




    @staticmethod
    def monotone_level_vec(levels, direction, null, n_levels):
        """``(B, H)`` ordered-dial encoding of a clip-level hint.

        ``level`` 1..``n_levels`` maps linearly onto ``z`` in [-1, 1] and is
        scaled by one learned ``direction``; level 0 (absent) is the learned
        ``null`` vector and contributes no direction term.  This is the
        difference from the free ``(n_levels + 1, H)`` table it replaces: a
        *single* direction is trained by every level, so the channel is monotone
        by construction and the gradient is not split across mutually arbitrary
        rows (see HINTS_V1_PLAN.md §5.6 for the measurement that motivated it).
        """
        lv = levels.long()
        present = lv > 0
        denom = max(int(n_levels) - 1, 1)
        z = (lv.float() - 1.0) / denom * 2.0 - 1.0        # 1..n -> [-1, 1]
        z = torch.where(present, z, torch.zeros_like(z))
        dt = direction.dtype
        out = z.unsqueeze(-1).to(dt) * direction
        if null is not None:
            out = out + (~present).unsqueeze(-1).to(dt) * null
        return out

    def _enc_hints(self, hints, device):
        """``(clip_vec (B,H) | None, per_phone (B,N,H) | None)`` for the encoder.

        Clip-level: intensity + register.  Per phone: word pitch
        (including the explicit unvoiced class) + word duration bucket.  Word
        hints are attached to *phones*, not words, because that is the sequence
        the encoder actually sees.
        """
        if not self.hints or hints is None:
            return None, None
        int_ids = hints.get("hint_int")
        if int_ids is None:
            return None, None
        int_ids = int_ids.to(device).long()
        if self.hint_monotone:
            clip = self.monotone_level_vec(int_ids, self.int_dir, self.int_null,
                                           self.n_intensity_levels)
        else:
            clip = self.int_emb(int_ids)
        reg = hints.get("hint_reg")
        if reg is not None:
            reg = reg.to(device).long()
            if self.hint_monotone:
                clip = clip + self.monotone_level_vec(
                    reg, self.reg_dir, self.reg_null, self.n_register_levels)
            else:
                clip = clip + self.reg_emb(reg)

        pos = None
        if self.hint_pitch_mode == "anchors":
            pos = self._pitch_anchor_vec(hints, device)
        else:
            pitch = hints.get("hint_pitch")
            if pitch is not None and pitch.numel() > 0:
                pos = self.pitch_emb(pitch.to(device).long())
        slot = hints.get("hint_slot")
        if slot is not None and slot.numel() > 0:
            slot_vec = self.slot_emb(slot.to(device).long())
            pos = slot_vec if pos is None else pos + slot_vec
        word = self._word_hint(hints, device)
        if word is not None:
            pos = word if pos is None else pos + word
        return clip, pos

    def _word_hint(self, hints, device):
        """``(B, N, H)`` per-phone vector from each word's BERT state.

        The dataset repeats a word's vector over its phones, so this is a plain
        per-phone projection -- no pooling of its own.  ``hint_word_valid`` is
        what carries "absent"/"dropped": those rows go through the learned
        ``word_null`` rather than ``word_proj(0)``, because an all-zero input is
        a legitimate (if unusual) encoder vector and the two must stay distinct
        for the dropout design to mean what it says.
        """
        if not self.hint_word_sem:
            return None
        w = hints.get("hint_word")
        if w is None or w.numel() == 0:
            return None
        w = w.to(device, torch.float32)
        if int(w.shape[-1]) != self.hint_word_dim:
            raise ValueError(
                f"hint_word has {int(w.shape[-1])} dims but "
                f"transformer.hint_word_dim={self.hint_word_dim}")
        x = self.word_proj(w)
        valid = hints.get("hint_word_valid")
        if valid is None:
            return x
        keep = valid.to(device).bool().unsqueeze(-1)
        null = self.word_null.to(x.dtype).expand_as(x)
        return torch.where(keep, x, null)

    def _pitch_anchor_vec(self, hints, device):
        """``(B, N, H)`` per-phone vector from each word's pitch anchors.

        Every phone of a word carries the word's **whole** anchor vector; a
        scalar per phone is not enough (measured: projecting 3 anchors to one
        value per phone is no better than 2, because a 320-440 ms word has ~2.5
        phones).  The phone's normalised position inside its word and the anchor
        positions are carried alongside, so the contour *and* the phone's place
        in it are both expressible.

        Three trained states are kept apart, because hint dropout relies on it:

        * absent (state 0) -> ``pitch_absent``;
        * voiced (state 1) -> the anchor sum, padded slots -> ``pitch_slot_null``;
        * unvoiced (state 2) -> the same plus ``pitch_unvoiced``.
        """
        val = hints.get("hint_pitch_val")
        pos = hints.get("hint_pitch_pos")
        num = hints.get("hint_pitch_n")
        state = hints.get("hint_pitch_state")
        if (val is None or val.numel() == 0 or pos is None or num is None
                or state is None):
            return None
        A = int(self.hint_pitch_anchors)
        val = val.to(device, torch.float32)
        pos = pos.to(device, torch.float32)
        if val.shape[-1] < A:
            pad = torch.zeros(val.shape[:-1] + (A - val.shape[-1],),
                              dtype=val.dtype, device=val.device)
            val = torch.cat((val, pad), dim=-1)
            pos = torch.cat((pos, pad), dim=-1)
        val = val[..., :A]
        pos = pos[..., :A]
        num = num.to(device).long().clamp_(min=0, max=A)
        state = state.to(device).long()
        frac = hints.get("hint_phone_frac")
        frac = (frac.to(device, torch.float32) if frac is not None
                else torch.zeros(num.shape, dtype=torch.float32, device=device))

        ar = torch.arange(A, device=device).view(1, 1, A)
        mask = ar < num.unsqueeze(-1)                     # (B, N, A)
        out = torch.einsum("bna,ah->bnh", torch.tanh(val / 6.0) * mask,
                           self.pitch_val_dir)
        out = out + torch.einsum("bna,ah->bnh", (pos - 0.5) * mask,
                                 self.pitch_pos_dir)
        out = out + ((~mask).unsqueeze(-1) * self.pitch_slot_null).sum(dim=-2)
        out = out + (frac - 0.5).unsqueeze(-1) * self.pitch_frac_dir
        present = (state != 0).unsqueeze(-1)
        out = (out * present.to(out.dtype)
               + (~present).to(out.dtype) * self.pitch_absent)
        out = out + (state == 2).unsqueeze(-1).to(out.dtype) * self.pitch_unvoiced
        return out

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
        if self.hint_monotone:
            if self.dec_int_dir is None:
                return None
            int_ids = int_ids.to(self.dec_int_dir.device).long()
            out = self.monotone_level_vec(int_ids, self.dec_int_dir,
                                          self.dec_int_null,
                                          self.n_intensity_levels)
        else:
            out = self.dec_int_emb(
                int_ids.to(self.dec_int_emb.weight.device).long())
        return out

    @staticmethod
    def _same_hints(a, b) -> bool:
        """True when two hint dicts are the same state (bitwise per channel)."""
        if a is None or b is None:
            return a is b
        if a is OMIT_HINTS or b is OMIT_HINTS:
            return a is b
        for key in ("hint_int", "hint_reg", "hint_pitch", "hint_slot"):
            va, vb = a.get(key), b.get(key)
            if va is None:
                continue
            if vb is None or tuple(va.shape) != tuple(vb.shape):
                return False
            if not torch.equal(va, vb):
                return False
        return True

    def _guidance_setup(self, hints, phone_ids, phone_lengths, sid,
                        hint_guidance=None, hint_guidance_books=None):
        """``(per-book weights, uncond memory, mask, uncond dec hint)``.

        Classifier-free guidance over the hint channels:
        ``logits = uncond + w * (cond - uncond)`` with ``uncond`` the trained
        null state (every channel absent) and ``cond`` the hints actually
        passed in.  It exists because the hint channels are *style* inputs: the
        teacher-forced CE gradient they get is tiny (measured 0.0000-0.011
        nats), so a plain conditional model gives a weak knob even with a
        working injection site, while the
        ``cond - uncond`` direction is exactly what the channel changes.

        Returns all ``None`` when it is off, when the conditional hints are
        already the null state (the direction is zero -- no second pass is
        paid for), or when every weight is 1.0 (the exact identity).

        Args:
            hint_guidance: scalar weight for every codebook; ``None`` -> the
                model's ``transformer.hint_guidance``.
            hint_guidance_books: per-codebook weights (overrides the scalar).
        """
        g = self.hint_guidance if hint_guidance is None else float(hint_guidance)
        books = (self.hint_guidance_books if hint_guidance_books is None
                 else [float(w) for w in hint_guidance_books])
        if not self.hints or (g is None and books is None):
            return None, None, None, None
        K = self.codebook_num
        weights = books if books is not None else [float(g)] * K
        if len(weights) != K:
            raise ValueError(
                f"hint_guidance_books has {len(weights)} entries but the model "
                f"has {K} codebooks")
        if all(abs(w - 1.0) < 1e-9 for w in weights):
            return None, None, None, None

        null = self.null_hints(phone_ids.shape[0], phone_ids.shape[1],
                               phone_ids.device)
        if self._same_hints(hints, null):
            return None, None, None, None
        memory, mask = self.encode(phone_ids, phone_lengths, sid, hints=null)
        return weights, memory, mask, self._dec_hint(null)

    @staticmethod
    def _combine_guidance(logits, logits_uncond, weights):
        """``uncond + w_k * (cond - uncond)`` per codebook (list in, list out)."""
        return [lu + w * (lc - lu)
                for w, lc, lu in zip(weights, logits, logits_uncond)]

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
        hints: Optional[dict] = None,
        word_progress: Optional[torch.Tensor] = None,  # (B, T) float, (0,1)
        word_progress_mask: Optional[torch.Tensor] = None,  # (B, T) bool
    ) -> dict:
        """Delayed next-token loss over all codebooks, plus word progress.

        With ``transformer.eos_token`` enabled, an EOS frame is appended to each
        sample's streams *before* the delay pattern is built, so "emit EOS" is
        just another next-token target and the delayed alignment is unchanged.
        Only samples flagged ``complete`` get one.

        Per-codebook cross-entropies are combined as
        ``loss = sum_k w_k * CE_k / sum(w)`` with ``w = loss_weights(epoch)``.
        When ``transformer.word_progress`` is on and the batch carries
        per-frame word targets, a small head predicts the normalised word
        position from the decoder hidden state and a masked regression +
        monotonicity term is added.  ``code_ce`` keeps the pure code CE for
        comparability with runs from before this auxiliary loss.
        """
        B, K, T = codes.shape
        if not isinstance(hints, dict):
            hints = self.resolve_hints(hints, B, memory.shape[1], memory.device)
        if code_lengths is None:
            code_lengths = torch.full((B,), T, dtype=torch.long,
                                      device=codes.device)
        if self.eos_token:
            codes, code_lengths = self._append_eos(codes, code_lengths, complete)
        delayed = self.build_delay_pattern(codes, code_lengths)
        L = delayed.shape[2]
        start = self._pad_column(B, codes.device)

        seq = torch.cat([start, delayed], dim=2)
        inputs = seq[..., :L]
        targets = seq[..., 1:]

        if self.word_progress_head is not None and word_progress is not None:
            logits, decoder_hidden = self.decoder.forward_with_hidden(
                inputs, memory, memory_mask, sid=sid,
                hint_vec=self._dec_hint(hints))
        else:
            decoder_hidden = None
            logits = self.decoder(inputs, memory, memory_mask, sid=sid,
                                  hint_vec=self._dec_hint(hints))

        losses, accs, entropies = [], [], []
        for k in range(K):
            lg = logits[k].float()
            flat_logits = lg.reshape(-1, lg.size(-1))
            flat_targets = targets[:, k].reshape(-1)
            losses.append(F.cross_entropy(
                flat_logits, flat_targets, ignore_index=self.pad_ids[k]))
            if not diagnostics:
                continue
            mask = (flat_targets != self.pad_ids[k]).unsqueeze(-1).to(lg.dtype)
            n_valid = mask.sum().clamp_min(1.0)
            correct = (flat_logits.argmax(-1) == flat_targets)
            accs.append((correct.to(lg.dtype).unsqueeze(-1) * mask).sum()
                        / n_valid)
            probs = (F.softmax(flat_logits, dim=-1) * mask).sum(0) / n_valid
            ent = -(probs.clamp_min(1e-9).log() * probs).sum()
            entropies.append(ent / math.log(max(probs.size(-1) - 1, 2)))

        losses = torch.stack(losses)
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
            accs = torch.stack(accs)
            entropies = torch.stack(entropies)
            out.update({
                "acc": accs.mean(),
                "spec_acc": accs[:n_spec].mean(),
                "pitch_acc": accs[n_spec:].mean(),
                "code_accs": accs,
                "code_usage_entropy": entropies,
            })

        if (decoder_hidden is not None and word_progress is not None
                and word_progress_mask is not None
                and self.word_progress_head is not None):
            out["code_ce"] = out["loss"]
            Tp = min(int(word_progress.shape[1]), int(T),
                     int(decoder_hidden.shape[1]))
            if Tp > 0:
                target = word_progress[:, :Tp].float()
                mask = word_progress_mask[:, :Tp].bool()
                pred = torch.sigmoid(
                    self.word_progress_head(decoder_hidden[:, :Tp])
                ).squeeze(-1).float()
                m = mask.to(pred.dtype)
                denom = m.sum().clamp_min(1.0)
                diff = pred - target
                reg = (diff.square() * m).sum() / denom
                mae = (diff.abs() * m).sum() / denom
                if Tp > 1:
                    pair = mask[:, :-1] & mask[:, 1:]
                    backwards = torch.relu(
                        pred[:, :-1] - pred[:, 1:] + self.word_progress_margin)
                    mono = ((backwards * pair.to(backwards.dtype)).sum()
                            / pair.to(backwards.dtype).sum().clamp_min(1.0))
                else:
                    mono = pred.new_zeros(())
                wp_loss = reg + self.word_progress_mono_weight * mono
                out["word_progress_loss"] = wp_loss
                out["word_progress_mse"] = reg
                out["word_progress_mae"] = mae
                out["word_progress_mono"] = mono
                out["word_progress_frac"] = mask.to(pred.dtype).mean()
                out["loss"] = out["loss"] + self.word_progress_weight * wp_loss
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
        hints: Optional[dict] = None,
        word_progress: Optional[torch.Tensor] = None,
        word_progress_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Training forward pass: phonemes + speaker -> spec/pitch code loss.

        ``epoch`` only selects the codebook weighting schedule (see
        :meth:`loss_weights`); it does not affect the model's computation.
        ``complete`` (``(B,)`` bool) marks the samples whose segment reaches the
        end of the clip and therefore get an EOS target (no-op unless
        ``transformer.eos_token`` is set).  ``diagnostics`` is forwarded to
        :meth:`code_loss`.

        ``hints`` may be a collated batch *or* one of the ``resolve_hints``
        strings (``"null"``/``"omit"``), which is what makes
        :meth:`hint_channel_deltas` able to score the same batch in a different
        state; it is resolved here, once, so ``encode`` and ``_dec_hint`` cannot
        disagree.
        """
        hints = self.resolve_hints(hints, phone_ids.shape[0],
                                   phone_ids.shape[1], phone_ids.device)
        memory, memory_mask = self.encode(phone_ids, phone_lengths, sid,
                                          hints=hints)
        codes = torch.cat([spec_codes, pitch_codes], dim=1).long()
        return self.code_loss(
            memory, memory_mask, codes, code_lengths, sid=sid, epoch=epoch,
            complete=complete, diagnostics=diagnostics, hints=hints,
            word_progress=word_progress,
            word_progress_mask=word_progress_mask)

    #: Sentinel for "no explicit hints argument" (see ``compute_batch_loss``).
    _USE_BATCH_HINTS = object()

    def compute_batch_loss(self, batch: dict,
                           epoch: Optional[int] = None,
                           diagnostics: bool = True,
                           hints=_USE_BATCH_HINTS) -> dict:
        """Convenience wrapper used by the Lightning module.

        Expects ``phone_ids``, ``phone_lengths``, ``sid``, ``spec_codes``,
        ``pitch_codes`` and optional ``code_lengths`` / ``complete`` entries.
        ``epoch`` drives the codebook weight curriculum; ``diagnostics``
        enables the per-codebook accuracy/usage-entropy extras (see
        :meth:`code_loss`).

        ``hints`` defaults to the batch itself; pass ``"omit"`` (or ``None``)
        to score the same batch in a different hint state -- that is what
        :meth:`hint_channel_deltas` does.
        """
        if hints is self._USE_BATCH_HINTS:
            hints = batch if self.hints else None
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
            hints=hints,
            word_progress=batch.get("word_progress"),
            word_progress_mask=batch.get("word_progress_mask"),
        )

    #: Per-channel keys a probe can drop, in the order they are logged.
    HINT_CHANNEL_KEYS = {
        "int": ("hint_int",),
        "reg": ("hint_reg",),
        # Both pitch representations are dropped together: state 0 is the
        # trained "absent" path for the anchors, level 0 for the v1 table.
        "pitch": ("hint_pitch", "hint_pitch_val", "hint_pitch_pos",
                  "hint_pitch_n", "hint_pitch_state"),
        "slot": ("hint_slot",),
        # V2.1 word states: the mask is the absent state, so both move together.
        "word_sem": ("hint_word", "hint_word_valid"),
    }

    @torch.no_grad()
    def hint_channel_deltas(self, batch: dict, epoch: Optional[int] = None,
                            channels: Optional[Sequence[str]] = None) -> dict:
        """Per-channel ``CE(channel dropped) - CE(all hints)``, in nats.

        The question this answers is *"is this channel load-bearing?"*, which
        neither ``val_loss`` nor ``val_hint_gap`` can: the latter drops every
        channel at once, so one live channel hides four dead ones.  Each arm
        re-scores the *same* batch with exactly one channel at its trained
        "not provided" state (every channel absent), plus:

        * ``all``  - every channel dropped (the trained null state, i.e. what
          generation runs in by default);
        * ``omit`` - no hint vectors at all, the pre-Phase-0 inference path.
          The difference between ``all`` and ``omit`` is that bug's cost.

        Cost: ``len(channels) + 1`` teacher-forced forwards.  Run it on one or
        two validation batches per epoch, not on the whole split.
        """
        if not self.hints or batch.get("hint_int") is None:
            return {}
        names = (list(channels) if channels
                 else list(self.HINT_CHANNEL_KEYS) + ["all", "omit"])
        base = float(self.compute_batch_loss(
            batch, epoch=epoch, diagnostics=False)["loss"])
        out = {"base": base}
        all_keys = tuple(k for keys in self.HINT_CHANNEL_KEYS.values()
                         for k in keys)
        for name in names:
            if name == "omit":
                loss = self.compute_batch_loss(batch, epoch=epoch,
                                               diagnostics=False, hints="omit")
                out[name] = float(loss["loss"]) - base
                continue
            if name not in self.HINT_CHANNEL_KEYS and name != "all":
                raise ValueError(
                    f"unknown hint channel {name!r}; expected one of "
                    f"{sorted(self.HINT_CHANNEL_KEYS)} or 'all'/'omit'")
            keys = all_keys if name == "all" else self.HINT_CHANNEL_KEYS[name]
            off = dict(batch)
            for key in keys:
                if off.get(key) is not None:
                    off[key] = torch.zeros_like(off[key])
            loss = self.compute_batch_loss(off, epoch=epoch, diagnostics=False)
            out[name] = float(loss["loss"]) - base
        return out


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
        hint_guidance: Optional[float] = None,
        hint_guidance_books: Optional[Sequence[float]] = None,
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
            hints: optional collated batch / hint dict.  ``None`` means the
                **trained null state** (every channel absent) -- not "no
                hints"; pass ``"omit"`` for
                the historical behaviour of dropping the vectors entirely (see
                :meth:`resolve_hints`).
            hint_guidance, hint_guidance_books: classifier-free guidance over
                the hint channels, ``logits = uncond + w*(cond - uncond)`` with
                ``uncond`` = the null state.  ``None`` -> the model's configured
                ``transformer.hint_guidance`` (off by default).  Costs a second
                decoder pass per step, and is a no-op when ``hints`` already
                *is* the null state.  See :meth:`_guidance_setup`.

        Returns:
            ``(B, K, n)`` codes, or ``(codes, lengths)`` if ``return_lengths``.
        """
        was_training = self.training
        self.eval()
        use_eos = self.eos_token if stop_on_eos is None else bool(stop_on_eos)
        use_eos = use_eos and self.eos_token
        try:
            # ``None`` -> the trained null state (every channel absent), which
            # is the only hint-free state the model ever saw.  See
            # ``resolve_hints``; pass "omit" to reproduce the pre-Phase-0 path.
            hints = self.resolve_hints(hints, phone_ids.shape[0],
                                       phone_ids.shape[1], phone_ids.device)
            memory, memory_mask = self.encode(phone_ids, phone_lengths, sid,
                                              hints=hints)
            dec_hint = self._dec_hint(hints)
            g_weights, g_memory, g_mask, g_hint = self._guidance_setup(
                hints, phone_ids, phone_lengths, sid,
                hint_guidance, hint_guidance_books)
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
                if g_weights is not None:
                    logits = self._combine_guidance(
                        logits,
                        self.decoder(seq, g_memory, g_mask, sid=sid,
                                     hint_vec=g_hint),
                        g_weights)
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
        if self.hints:
            pitch_desc = (f"{self.hint_pitch_anchors} anchors"
                          if self.hint_pitch_mode == "anchors"
                          else f"{self.n_pitch_levels} pitch")
            hints = (f", hints={self.n_intensity_levels} int/{pitch_desc}"
                     f"/{self.n_register_levels} reg/{self.n_slot_buckets}"
                     f" slot"
                     + (f"+word{self.hint_word_dim}" if self.hint_word_sem
                        else "")
                     + (' (decoder too)' if self.decoder_hints
                        else ' (encoder only)'))
            if self.hint_monotone:
                hints += " monotone(int,reg)"
            if self.hint_film_memory or self.hint_decoder_proj:
                where = "+".join([n for n, on in
                                  (("film", self.hint_film_memory),
                                   ("dec_proj", self.hint_decoder_proj)) if on])
                hints += f" inject={where}"
            if self.hint_guidance is not None or self.hint_guidance_books:
                w = (self.hint_guidance_books if self.hint_guidance_books
                     else round(float(self.hint_guidance), 3))
                hints += f" cfg={w}"
        else:
            hints = ", hints=off"
        return (
            f"CodeTransformer(hidden={self.hidden_size}, codebooks="
            f"{self.spec_codebook_num} spec + {self.pitch_codebook_num} pitch, "
            f"speaker={spk}, memory_norm={self.memory_norm}, "
            f"residual_scale={self.residual_scale}, eos={self.eos_token}, "
            + (f"rope(decoder, base={self.rope_base:g})" if self.rope
               else "pos=sinusoidal")
            + (f", input_dropout={self.input_dropout:g}"
               if self.input_dropout > 0 else "")
            + (f", word_progress(w={self.word_progress_weight:g}, "
               f"mono={self.word_progress_mono_weight:g})"
               if self.word_progress else "")
            + hints
            + (f", finetune=lora(r{self.lora_config['rank']})"
               if self.lora_enabled else "")
            + f", params={n / 1e6:.2f}M)"
        )


def adapt_state_dict_special(model: "CodeTransformer", state_dict: dict,
                             prefix: str = "net.") -> dict:
    """Match a checkpoint's codebook special-class rows to this model.

    A stage-2 checkpoint may predate EOS (narrower) or contain the retired FIM
    SUF/MID classes (wider).  ``load_state_dict_mismatch`` skips any tensor
    whose shape differs, which would silently re-initialise every codebook
    embedding/head.  This helper zero-pads a narrower checkpoint and slices a
    wider one down to ``model.decoder.num_classes[k]``.

    FIM is no longer part of the model, so the retired tables are also dropped:
    ``decoder.region_emb``, ``decoder.len_emb`` and ``fim_layout_marker``.

    Args:
        model: the model being loaded into.
        state_dict: checkpoint state dict, with or without ``prefix``.
        prefix: key prefix used by the checkpoint (``"net."`` for Lightning
            stage-2 checkpoints, ``""`` for a bare ``CodeTransformer``).

    Returns:
        A new dict (or the input when nothing changed).
    """
    out = dict(state_dict)
    use_prefix = any(k.startswith(prefix) for k in out) if prefix else False
    out_prefix = prefix if use_prefix else ""
    num_classes = model.decoder.num_classes
    changed = False
    for k in range(len(model.vocab_sizes)):
        target = int(num_classes[k])
        names = (f"embeds.{k}.weight", f"heads.{k}.weight", f"heads.{k}.bias")
        for name in names:
            for key in (prefix + f"decoder.{name}", f"decoder.{name}"):
                src = out.get(key)
                if src is None:
                    continue
                if src.shape[0] == target:
                    continue
                if src.shape[0] > target:
                    out[key] = src[:target].clone()
                else:
                    grown = src.new_zeros((target, *src.shape[1:]))
                    grown[: src.shape[0]] = src
                    out[key] = grown
                changed = True
    # Fold the retired AR-region / AR-length embeddings into one constant
    # decoder bias.  Every plain AR column used to receive
    # ``region_emb[0] + len_emb[0]``; dropping the tables without this would
    # shift every decoder input and make a warm start look broken.
    bias = None
    for name in ("decoder.region_emb.weight", "decoder.len_emb.weight"):
        for key in (prefix + name, name):
            if key in out:
                row = out.pop(key)[0].to(torch.float32)
                bias = row if bias is None else bias + row
                changed = True
    for name in ("fim_layout_marker",):
        for key in (prefix + name, name):
            if key in out:
                del out[key]
                changed = True
    if bias is not None:
        out[out_prefix + "decoder.input_bias"] = bias
    return out if changed else state_dict


# Backwards-compatible name (existing training/eval imports use it).
expand_state_dict_with_special = adapt_state_dict_special
expand_state_dict_with_eos = adapt_state_dict_special



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
