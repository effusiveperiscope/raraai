#!/usr/bin/env python3
"""How well does the decoder align frames to phones? (diagnostic, no ASR)

``val_loss`` and ``val_text_delta`` cannot answer this.  ``text_delta`` only
says "changing the phonemes changes the output" -- a model that reads the
phoneme memory as one *pooled* vector passes it easily.  What matters for
pacing (dropped final words, drawn-out prosody) is whether frame ``t`` attends
to the phone that should be sounding at ``t``.

This walks a few val utterances through the teacher-forced path, captures the
decoder's cross-attention weights, and reports per layer:

  ``entropy/logN``  the attention entropy normalised by the maximum. **1.0 is
                    perfectly uniform** (the layer is not looking at any
                    particular phone); a healthy alignment is well below 0.5
                    for the frames inside the speech.
  ``|pos-oracle|``  mean absolute error between the attention's expected phone
                    position and the oracle schedule from the ``.cond``
                    alignment, both normalised to [0, 1].
  ``pacing err``    the same difference at the tail of the utterance (last fifth
                    of the valid frames).  Only meaningful once the entropy is
                    low: with uniform attention the expected position is ~0.5
                    whatever the oracle is, so the number is just ``0.5 -
                    oracle(tail)``.  Once aligned, positive means the attention
                    is ahead of the audio (the decoder has read past the text),
                    i.e. it will wrap up early.

Measured on ``transformer_hints_v2.7_spk256`` (the current best) and on
``transformer_duration_v1``: **every layer sits at 0.96-0.99**, i.e. uniform
attention to a perfectly healthy memory (phone vectors pairwise cosine ~0).
So the decoder reproduces codes from its own prefix + speaker and uses the text
only as a pooled summary -- which is why the utterance length, the tail words
and the word identity are not under text control.

Usage::

    python scratch/check_alignment.py --checkpoint checkpoints_stage2/<exp>/last.ckpt
    python scratch/check_alignment.py --checkpoint ... --config configs/transformer.yaml

Needs the same config the checkpoint was trained with (the hint layout and
RoPE flag change the parameter set).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.WARNING)

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from commons import load_submodule_prefix  # noqa: E402
from modeling.transformer import adapt_state_dict_special, CodeTransformer  # noqa: E402
from stage2_wer import build_wer_dataset  # noqa: E402


def capture_attention(net):
    """Enable attention capture on every decoder cross-attention.

    Layers built with ``transformer.align_supervision`` already have
    ``CapturingCrossAttention`` modules (just flip the flag); the rest are stock
    ``nn.MultiheadAttention`` and get a wrapper that asks for the weights.
    Returns ``(own, spies)``: ``own`` is ``[(layer_index, module)]`` for the
    native ones, ``spies`` the ``{index: weights}`` dict the wrappers fill.
    """
    own, spies = [], {}
    for li, layer in enumerate(net.decoder.decoder.layers):
        module = layer.multihead_attn
        if hasattr(module, "capture"):
            module.capture = True
            own.append((li, module))
            continue
        orig = module.forward

        def make(orig, li):
            def spy(query, key, value, **kw):
                kw["need_weights"] = True
                out, w = orig(query, key, value, **kw)
                spies[li] = w.detach()
                return out, w
            return spy

        module.forward = make(orig, li)
    return own, spies


def captured(own, spies):
    out = {li: m.last_attn for li, m in own if m.last_attn is not None}
    out.update(spies)
    return out


def release_attention(own):
    for _, module in own:
        module.capture = False
        module.last_attn = None


@torch.no_grad()
def layer_stats(net, own, spies, item, device):
    """Per-frame attention statistics for one item (teacher-forced)."""
    phones = item["phone_ids"].unsqueeze(0).to(device)
    n_ph = int(phones.shape[1])
    plen = torch.tensor([n_ph], device=device)
    sid = torch.tensor([int(item["sid"])], device=device)
    codes = torch.cat([item["spec_codes"], item["pitch_codes"]], 0)
    codes = codes.unsqueeze(0).long().to(device)
    T = int(codes.shape[2])
    batch = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v)
             for k, v in item.items() if k.startswith("hint")}
    for k in ("hint_int", "hint_reg"):
        if k in item:
            batch[k] = torch.tensor([int(item[k])], device=device)
    spies.clear()
    for _, module in own:
        module.last_attn = None
    memory, mask = net.encode(phones, plen, sid, hints=batch)
    net.code_loss(memory, mask, codes, torch.tensor([T], device=device), sid=sid,
                  epoch=None, complete=torch.tensor([True], device=device),
                  diagnostics=False, hints=batch)

    oracle = item.get("dur_prog")
    if oracle is None:
        return None
    oracle = oracle[:T].numpy()
    valid = item["dur_prog_mask"][:T].numpy()
    if not valid.any():
        return None
    out = {}
    for li, w in captured(own, spies).items():
        # Native capture keeps the per-head axis (B, H, L, S); the fallback
        # wrapper returns the head average (B, L, S).  Average either way.
        if w.dim() == 4:
            w = w.mean(1)
        p = w[0][:T, :n_ph].clamp_min(1e-9)
        p = p / p.sum(-1, keepdim=True)
        ent = float((-(p.log() * p).sum(-1)).mean() / np.log(max(n_ph, 2)))
        pos = ((p * torch.arange(n_ph, device=p.device)).sum(-1)
               / max(n_ph - 1, 1)).cpu().numpy()
        err = float(np.abs(pos - oracle)[valid].mean())
        n_tail = max(1, int(valid.sum()) // 5)
        tail = float((pos - oracle)[valid][-n_tail:].mean())
        out[li] = (ent, err, tail)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/transformer.yaml")
    ap.add_argument("--items", type=int, default=4)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or
                          ("cuda:0" if torch.cuda.is_available() else "cpu"))
    cfg = OmegaConf.load(args.config)
    net = CodeTransformer(cfg)
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False,
                    mmap=True)["state_dict"]
    sd = adapt_state_dict_special(net, sd)
    prefix = "net." if any(k.startswith("net.") for k in sd) else ""
    load_submodule_prefix(net, prefix, sd, quiet=True)
    net.eval().to(device)
    torch.backends.mha.set_fastpath_enabled(False)

    # The oracle schedule is a dataset-side artefact, so ask for it even on a
    # checkpoint that does not consume it: the alignment columns are still
    # measurable, and that is how the duration channel was shown to be inert.
    ds_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    ds_cfg.transformer.duration_cond = True
    ds = build_wer_dataset(ds_cfg, hints=bool(cfg.transformer.get("hints", False)),
                           duration=True)
    picks = [i for i in range(len(ds))
             if "dur_prog" in ds[i] and int(ds[i]["dur_prog_mask"].sum()) > 0
             and ds[i]["phone_ids"].numel() >= 12][:args.items]
    if not picks:
        raise SystemExit("no val item has a .cond word alignment (dur_prog); "
                         "hints/build.py --words is required")

    own, spies = capture_attention(net)
    per_layer = {}
    n = 0
    for i in picks:
        stats = layer_stats(net, own, spies, ds[i], device)
        if stats is None:
            continue
        n += 1
        for li, v in stats.items():
            per_layer.setdefault(li, []).append(v)
    if not n:
        raise SystemExit("no usable item")
    release_attention(own)
    print(f"{os.path.basename(os.path.dirname(args.checkpoint))}"
          f"/{os.path.basename(args.checkpoint)}  ({n} clips, "
          f"from {args.config})")
    print(f"{'layer':>5} {'entropy/logN':>13} {'|pos-oracle|':>13} "
          f"{'pacing err':>11}")
    for li in sorted(per_layer):
        e = np.mean([x[0] for x in per_layer[li]])
        d = np.mean([x[1] for x in per_layer[li]])
        t = np.mean([x[2] for x in per_layer[li]])
        print(f"{li:>5} {e:>13.3f} {d:>13.3f} {t:>+11.3f}")
    print("\nentropy/logN 1.0 = uniform attention to the phone memory; "
          "a working alignment is < 0.5 for the voiced frames.  Read the "
          "position columns only after that has moved:\nwith uniform "
          "attention they are pinned at ~0.5 - oracle.")


if __name__ == "__main__":
    main()
