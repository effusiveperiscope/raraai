"""Non-autoregressive plan renderer: phones + plan -> stage-1 RVQ codes.

    phone ids ──encoder──┐
                          expand at 100 Hz (weighted phone mixture)
    phone durations ──────┘                    │
    per-frame plan features (pitch, level, …) ─┼─→ dilated conv decoder
    speaker id ────────────────────────────────┘        │
                                                       ├─ 8 parallel spec heads (64-way)
                                                       ├─ 2 depth-AR pitch heads (128-way)
                                                       └─ optional f0 regression head

No time-autoregression and no EOS: the output length **is** ``T``, the plan's
code-frame count.  The phone expansion runs at 100 Hz (four subframes per code
frame) and is pooled 4:1, so a 40 ms word with three phones still contributes
all three phones instead of yielding zero-length frames.

The codebook head split is measured, not assumed (see AGENTS.md §3): spec books
are close to independent within a frame (~0.4 bits of conditional saving each),
so they are parallel; the two pitch books save 1.23 of ~6 bits, so book 1 sees
book 0 through a small teacher-forced/sampled step.

This module is *model only*.  The training script owns the optimiser, schedule
and checkpointing; ``renderer_loss`` is shared so the val loop cannot drift.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Matches the corpus: 8 spec books of 64, 2 pitch books of 128.
N_SPEC_BOOKS = 8
SPEC_SIZE = 64
N_PITCH_BOOKS = 2
PITCH_SIZE = 128
#: Per-code-frame plan feature vector (order matches ``plan.FRAME_FEATURES``).
N_FRAME_FEATS = 6


class ConvBlock(nn.Module):
    """Residual dilated conv + LayerNorm; mask is applied after the residual."""

    def __init__(self, channels: int, kernel: int = 5, dilation: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        pad = dilation * (kernel // 2)
        self.conv = nn.Conv1d(channels, channels, kernel, padding=pad,
                              dilation=dilation)
        self.norm = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        y = self.norm(x + self.drop(F.gelu(y)))
        return y * mask.unsqueeze(-1).to(y.dtype)


def _sinusoidal(length: int, dim: int, device) -> torch.Tensor:
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32)
                    * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def sample_codes(logits: torch.Tensor, temperature: float = 1.0,
                 top_k: int = 0) -> torch.Tensor:
    """``(..., V)`` logits -> ``(...)`` codes; temperature 0 = argmax."""
    if temperature <= 0:
        return logits.argmax(-1)
    x = logits / float(temperature)
    if top_k and top_k < x.shape[-1]:
        kth = torch.topk(x, top_k, dim=-1).values[..., -1, None]
        x = x.masked_fill(x < kth, float("-inf"))
    flat = x.reshape(-1, x.shape[-1])
    return torch.multinomial(F.softmax(flat, -1), 1).view(x.shape[:-1])


class GuidedRenderer(nn.Module):
    """Phones + plan -> ``(8, T)`` spec codes, ``(2, T)`` pitch codes, f0.

    Args:
        vocab: phoneme vocabulary size (g2pen has 1024 ids, 0 = padding).
        num_speakers: rows in the stage-2 speaker embedding table.
        width: hidden channels (384-512 is ~10-18 M params with the default
            depths; 768 with 8+8 blocks is ~40 M).
        phone_layers / frame_layers: conv block counts.
        depth_ar_pitch: book 1 of the pitch codec sees book 0.  Set False for
            the parallel ablation.
        f0_aux: add the (cheap) f0 regression head, supervised to the true
            contour; it steers the hidden for guidance without changing the
            code output path.
    """

    #: The g2pen tokenizer has 1024 ids (max id 1023).
    def __init__(self, vocab: int = 1024, num_speakers: int = 8192,
                 width: int = 512, phone_layers: int = 4,
                 frame_layers: int = 6, kernel: int = 5,
                 dropout: float = 0.1, depth_ar_pitch: bool = True,
                 f0_aux: bool = True):
        super().__init__()
        d = int(width)
        self.depth_ar_pitch = bool(depth_ar_pitch)
        self.f0_aux = bool(f0_aux)

        # -- phones -------------------------------------------------------- #
        self.phone_emb = nn.Embedding(vocab, d, padding_idx=0)
        self.phone_blocks = nn.ModuleList([
            ConvBlock(d, kernel, dilation=1 << (i % 3), dropout=dropout)
            for i in range(phone_layers)])
        self.phone_norm = nn.LayerNorm(d)
        #: frames with no sounding phone (lead/pause)
        self.silence = nn.Parameter(torch.zeros(d))

        # -- frame path ---------------------------------------------------- #
        self.feat_proj = nn.Sequential(
            nn.Linear(N_FRAME_FEATS, d), nn.GELU(), nn.Linear(d, d))
        self.speaker = nn.Embedding(num_speakers, d)
        self.in_norm = nn.LayerNorm(d)
        self.frame_blocks = nn.ModuleList([
            ConvBlock(d, kernel, dilation=1 << (i % 4), dropout=dropout)
            for i in range(frame_layers)])
        self.out_norm = nn.LayerNorm(d)

        # -- heads --------------------------------------------------------- #
        self.spec_head = nn.Linear(d, N_SPEC_BOOKS * SPEC_SIZE)
        # pitch: book 0 from the hidden; book k>0 also sees book k-1's chosen
        # code embedding.  The output length is unchanged (T in, T out).
        self.pitch_heads = nn.ModuleList([
            nn.Linear(d + (d if (depth_ar_pitch and k) else 0), PITCH_SIZE)
            for k in range(N_PITCH_BOOKS)])
        self.pitch_emb = nn.ModuleList([
            nn.Embedding(PITCH_SIZE, d) for _ in range(N_PITCH_BOOKS)])
        self.f0_head = nn.Linear(d, 1) if f0_aux else None

    # ------------------------------------------------------------------ #

    def expanded_phones(self, phone_h: torch.Tensor,
                        phone_index: torch.Tensor) -> torch.Tensor:
        """``(B, 4T)`` index (-1 = silence) -> ``(B, T, D)`` weighted mixture."""
        B, P = phone_index.shape
        safe = phone_index.clamp_min(0)
        emb = phone_h.gather(1, safe.unsqueeze(-1).expand(-1, -1, phone_h.shape[-1]))
        sil = self.silence.view(1, 1, -1).expand_as(emb)
        emb = torch.where((phone_index >= 0).unsqueeze(-1), emb, sil)
        return emb.reshape(B, P // 4, 4, phone_h.shape[-1]).mean(dim=2)

    def pitch_logits(self, h: torch.Tensor,
                     targets: torch.Tensor | None = None,
                     temperature: float = 1.0, top_k: int = 0
                     ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Depth-AR over the 2 pitch books; all frames in parallel.

        Returns ``(logits_per_book, chosen_codes_per_book)``.  With ``targets``
        the codes are the teacher-forcing targets; without them they are the
        sampled codes, which the eval path needs to build the code tensor.
        """
        out, codes, prev = [], [], None
        for k, head in enumerate(self.pitch_heads):
            use_prev = self.depth_ar_pitch and k > 0
            inp = torch.cat([h, prev], dim=-1) if use_prev else h
            logits = head(inp)
            out.append(logits)
            # targets are (B, n_pitch_books, T), like the stored pitch codes
            code = targets[:, k] if targets is not None else \
                sample_codes(logits, temperature, top_k)
            codes.append(code)
            if self.depth_ar_pitch:
                prev = self.pitch_emb[k](code)
        return out, codes

    def forward(self, *, phone_ids: torch.Tensor, phone_mask: torch.Tensor,
                phone_index: torch.Tensor, frame_feats: torch.Tensor,
                sid: torch.Tensor, frame_mask: torch.Tensor,
                pitch_targets: torch.Tensor | None = None,
                temperature: float = 1.0,
                top_k: int = 0) -> dict:
        B, T = frame_mask.shape
        ph = self.phone_emb(phone_ids) * phone_mask.unsqueeze(-1).to(
            self.phone_emb.weight.dtype)
        for blk in self.phone_blocks:
            ph = blk(ph, phone_mask)
        ph = self.phone_norm(ph) * phone_mask.unsqueeze(-1)

        h = self.expanded_phones(ph, phone_index)
        h = h + self.feat_proj(frame_feats)
        h = h + self.speaker(sid).unsqueeze(1)
        h = h + _sinusoidal(T, h.shape[-1], h.device).unsqueeze(0).to(h.dtype)
        h = self.in_norm(h)
        h = h * frame_mask.unsqueeze(-1).to(h.dtype)
        for blk in self.frame_blocks:
            h = blk(h, frame_mask)
        h = self.out_norm(h) * frame_mask.unsqueeze(-1).to(h.dtype)

        spec = self.spec_head(h).view(B, T, N_SPEC_BOOKS, SPEC_SIZE)
        pitch, pitch_codes = self.pitch_logits(
            h, pitch_targets, temperature, top_k)
        out = {"spec_logits": spec, "pitch_logits": pitch, "hidden": h}
        if pitch_targets is None:
            out["pitch_codes"] = torch.stack(pitch_codes, dim=1)  # (B,2,T)
        if self.f0_head is not None:
            out["f0_pred"] = self.f0_head(h).squeeze(-1)
        return out


# --------------------------------------------------------------------------- #
# loss (shared by train/val)
# --------------------------------------------------------------------------- #

def _masked_ce(logits: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """``logits (B,T,V)``, ``target (B,T)``, ``mask (B,T)`` -> scalar CE."""
    return F.cross_entropy(logits[mask], target[mask])


def renderer_loss(out: dict, batch: dict, *,
                  spec_weight: float = 1.0,
                  pitch_weight: float = 1.0,
                  f0_weight: float = 0.05) -> dict:
    """Masked code CE + optional f0 L1.

    ``spec_weight``/``pitch_weight`` are per-codebook means, not sums, so the
    two terms have comparable gradients across 8 vs 2 books.
    """
    mask = batch["frame_mask"]
    B, T = mask.shape
    spec = out["spec_logits"].reshape(B, T, N_SPEC_BOOKS, SPEC_SIZE)
    target = batch["spec_codes"].permute(0, 2, 1)          # (B,T,8)
    per = F.cross_entropy(spec.reshape(-1, SPEC_SIZE),
                          target.reshape(-1), reduction="none")
    spec_ce = per.view(B, T, N_SPEC_BOOKS)[mask].mean()

    pitch_ce = spec.new_zeros(())
    for k, logits in enumerate(out["pitch_logits"]):
        pitch_ce = pitch_ce + _masked_ce(logits, batch["pitch_codes"][:, k], mask)
    pitch_ce = pitch_ce / N_PITCH_BOOKS

    losses = {"spec_ce": spec_ce, "pitch_ce": pitch_ce,
              "loss": spec_weight * spec_ce + pitch_weight * pitch_ce}
    if "f0_pred" in out and batch.get("f0_valid") is not None:
        f0 = batch["f0"].reshape(B, T, 4)
        ok = batch["f0_valid"].reshape(B, T, 4) & mask.unsqueeze(-1)
        st = 12.0 * torch.log2(f0.clamp_min(1e-3) / 100.0)
        valid = ok.any(-1)
        if valid.any():
            target_st = (st * ok).sum(-1) / ok.sum(-1).clamp_min(1)
            l1 = (out["f0_pred"] - target_st).abs()[valid].mean()
            losses["f0_l1"] = l1
            losses["loss"] = losses["loss"] + f0_weight * l1
    return losses


__all__ = [
    "GuidedRenderer", "renderer_loss", "sample_codes", "ConvBlock",
    "N_SPEC_BOOKS", "SPEC_SIZE", "N_PITCH_BOOKS", "PITCH_SIZE",
]
