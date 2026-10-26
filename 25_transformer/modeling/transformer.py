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
    """Encodes phoneme tokens + a speaker id into a cross-attention memory."""

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
    ):
        super().__init__()
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
        self.encoder = nn.TransformerEncoder(layer, n_layers, norm=nn.LayerNorm(hidden_size))

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
        return x, valid


class DelayCodebookDecoder(nn.Module):
    """Causal transformer decoder that predicts every codebook per column.

    Each codebook has its own embedding table and output head because the
    spectrogram codebooks and pitch codebooks have different vocabulary
    sizes.  Every codebook gets one extra "pad/start" index equal to its
    vocabulary size (mirroring MusicGen's ``bos == pad`` token).
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
    ):
        super().__init__()
        self.vocab_sizes = list(vocab_sizes)
        self.num_codebooks = len(self.vocab_sizes)
        self.hidden_size = hidden_size
        self.speaker_cond = bool(speaker_cond)

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
            [nn.Embedding(v + 1, hidden_size) for v in self.vocab_sizes]
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
            [nn.Linear(hidden_size, v + 1) for v in self.vocab_sizes]
        )

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
        self.hidden_size = hidden_size
        self.n_speakers = n_speakers
        self.decoder_speaker_cond = decoder_speaker_cond

        self.encoder = PhonemeEncoder(
            phone_vocab_size=phone_vocab,
            n_speakers=n_speakers,
            hidden_size=hidden_size,
            n_layers=enc_layers,
            n_heads=n_heads,
            dim_feedforward=ffn,
            dropout=dropout,
            max_positions=max_positions,
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
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, config) -> "CodeTransformer":
        return cls(config)

    def _pad_column(self, batch_size: int, device, dtype=torch.long) -> torch.Tensor:
        """(B, K, 1) column of per-codebook pad/start ids."""
        col = torch.tensor(self.pad_ids, device=device, dtype=dtype).view(1, -1, 1)
        return col.expand(batch_size, -1, 1).contiguous()

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
    ) -> dict:
        """Delayed next-token loss over all codebooks.

        Returns a dict with the total loss plus per-group (spec/pitch) losses
        and diagnostics that make "is it learning?" answerable from
        TensorBoard alone:

        ``loss``/``spec_loss``/``pitch_loss``
            mean cross-entropy over all / spec / pitch codebooks.
        ``acc``/``spec_acc``/``pitch_acc``
            teacher-forced top-1 accuracy over non-pad target positions.
        ``code_losses`` (K,), ``code_accs`` (K,), ``code_usage_entropy`` (K,)
            the same per codebook.  ``code_usage_entropy`` is the entropy of
            the batch-averaged predicted distribution, normalised to
            ``[0, 1]`` by ``log(vocab - 1)``; values near 0 mean the head has
            collapsed onto a handful of codes even if the loss still looks
            healthy.
        """
        B, K, T = codes.shape
        delayed = self.build_delay_pattern(codes, code_lengths)  # (B, K, L)
        L = delayed.shape[2]
        start = self._pad_column(B, codes.device)

        seq = torch.cat([start, delayed], dim=2)  # (B, K, L + 1)
        inputs = seq[..., :L]
        targets = seq[..., 1:]

        # list of (B, L, v+1)
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
                # target, so the attainable maximum is log(vocab - 1).
                entropies.append(
                    ent / math.log(max(probs.size(-1) - 1, 2)))
            else:  # degenerate batch (everything masked)
                accs.append(lg.new_zeros(()))
                entropies.append(lg.new_zeros(()))

        losses = torch.stack(losses)  # (K,)
        accs = torch.stack(accs)  # (K,)
        entropies = torch.stack(entropies)  # (K,)
        n_spec = self.spec_codebook_num
        return {
            "loss": losses.mean(),
            "spec_loss": losses[:n_spec].mean(),
            "pitch_loss": losses[n_spec:].mean(),
            "acc": accs.mean(),
            "spec_acc": accs[:n_spec].mean(),
            "pitch_acc": accs[n_spec:].mean(),
            "code_losses": losses,
            "code_accs": accs,
            "code_usage_entropy": entropies,
        }

    def forward(
        self,
        phone_ids: torch.Tensor,
        phone_lengths: Optional[torch.Tensor],
        sid: torch.Tensor,
        spec_codes: torch.Tensor,
        pitch_codes: torch.Tensor,
        code_lengths: Optional[torch.Tensor] = None,
    ) -> dict:
        """Training forward pass: phonemes + speaker -> spec/pitch code loss."""
        memory, memory_mask = self.encode(phone_ids, phone_lengths, sid)
        codes = torch.cat([spec_codes, pitch_codes], dim=1).long()
        return self.code_loss(
            memory, memory_mask, codes, code_lengths, sid=sid)

    def compute_batch_loss(self, batch: dict) -> dict:
        """Convenience wrapper used by the Lightning module.

        Expects ``phone_ids``, ``phone_lengths``, ``sid``, ``spec_codes``,
        ``pitch_codes`` and an optional ``code_lengths`` entry.
        """
        return self.forward(
            phone_ids=batch["phone_ids"],
            phone_lengths=batch.get("phone_lengths"),
            sid=batch["sid"],
            spec_codes=batch["spec_codes"],
            pitch_codes=batch["pitch_codes"],
            code_lengths=batch.get("code_lengths"),
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
    ) -> torch.Tensor:
        """Autoregressively sample ``(B, K, max_frames)`` delayed codes.

        The returned tensor has the spec codebooks first (``spec_codebook_num``
        rows) followed by the pitch codebooks, matching the layout used for
        training.  Uses the same delay pattern as :meth:`code_loss`.
        """
        was_training = self.training
        self.eval()
        try:
            memory, memory_mask = self.encode(phone_ids, phone_lengths, sid)
            B = phone_ids.shape[0]
            K = self.codebook_num
            L = max_frames + K - 1
            device = phone_ids.device

            seq = self._pad_column(B, device)
            cols = []
            for i in range(L):
                logits = self.decoder(seq, memory, memory_mask, sid=sid)
                col = []
                for k in range(K):
                    lg = logits[k][:, -1] / max(temperature, 1e-6)
                    # Never emit the pad/start class as a real token.
                    lg = lg.clone()
                    lg[..., self.pad_ids[k]] = float("-inf")
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

                seq = torch.cat([seq, col.unsqueeze(-1)], dim=2)
                cols.append(col)

            cols = torch.stack(cols, dim=2)  # (B, K, L)
            out = cols.new_zeros((B, K, max_frames))
            for k in range(K):
                out[:, k, :] = cols[:, k, k : k + max_frames]
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
            f"speaker={spk}, params={n / 1e6:.2f}M)"
        )


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
