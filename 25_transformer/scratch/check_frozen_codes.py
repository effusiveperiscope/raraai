"""Verify the stage-1 frozen-code ("acoustic fine-tuning") invariant.

``1_training.py --acoustic_only`` freezes everything that *produces* RVQ codes
(:data:`modeling.generator.CODE_DECODE_PREFIXES` is the trainable allow-list, its
complement is frozen, and both quantisers get ``freeze()`` so their EMA cannot
move the codebook). The point of the mode is that the codes a checkpoint already
produced stay valid, so ``spec_codes``/``pitch_codes`` never need re-extracting
for an existing corpus.

This script checks that claim from both ends:

    # 1. graph level, no checkpoint needed (CPU is fine, ~1 min):
    #    freeze -> 3 optimizer steps -> codes must be bit-identical, and no
    #    frozen weight or codebook buffer may have changed.
    python scratch/check_frozen_codes.py

    # 2. checkpoint level, the practical question: "did my fine-tune really
    #    keep the codes?"  Compares two stage-1 checkpoints on real val items.
    python scratch/check_frozen_codes.py \
        --checkpoint checkpoints_stage1/base_v0.6a/last.ckpt \
        --compare   checkpoints_stage1/base_v0.7_spk2/last.ckpt

Exits non-zero if anything moved (mode 1: any code difference, frozen-weight
change or unexpected trainable tensor; mode 2: any code difference).
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from omegaconf import OmegaConf

from commons import load_submodule_prefix
from modeling.generator import CODE_DECODE_PREFIXES, Generator


def build_generator(config_path: str, checkpoint: str | None = None) -> Generator:
    hp = OmegaConf.load(config_path)
    net = Generator(hp)
    if checkpoint:
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        load_submodule_prefix(net, 'net_g.', state['state_dict'])
    net.eval()
    return net


def report_match(name: str, ref: torch.Tensor, cur: torch.Tensor) -> bool:
    """Print the per-codebook match fraction; True when everything matches."""
    if ref.shape != cur.shape:
        print(f'  {name}: SHAPE MISMATCH {tuple(ref.shape)} vs {tuple(cur.shape)}')
        return False
    ok = True
    for q in range(ref.shape[0]):
        frac = float((ref[q] == cur[q]).float().mean())
        ok = ok and frac == 1.0
        print(f'  {name} book {q}: {frac * 100:9.5f} % identical'
              f'{"  <-- MOVED" if frac < 1.0 else ""}')
    n = int((ref != cur).sum())
    print(f'  {name}: {n} differing codes out of {ref.numel()}')
    return ok


def graph_check(config_path: str) -> int:
    torch.manual_seed(0)
    net = build_generator(config_path)
    hp = net.hp
    b, t = 2, 100
    spec = torch.rand(b, hp.data.filter_length // 2 + 1, t)
    spec_l = torch.tensor([t, t - 12])
    f0 = torch.rand(b, t) * 200.0
    sid = torch.tensor([0, 1])

    net.eval()
    ref_spec, ref_pitch = net.encode_codes(spec, spec_l, f0, sid)

    net.freeze_code_encoders()
    trainable = [n for n, p in net.named_parameters() if p.requires_grad]
    stray = [n for n in trainable if not n.startswith(CODE_DECODE_PREFIXES)]
    frozen = {n: p.detach().clone()
              for n, p in net.named_parameters() if not p.requires_grad}
    buffers = {n: x.detach().clone() for n, x in net.named_buffers()}
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    n_froz = sum(p.numel() for p in net.parameters() if not p.requires_grad)
    print(f'  trainable {n_train / 1e6:.2f} M / frozen {n_froz / 1e6:.2f} M')
    if stray:
        print('  ERROR: trainable tensors outside CODE_DECODE_PREFIXES:')
        for name in stray[:10]:
            print(f'    {name}')
        return 1

    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                            lr=1e-3)
    net.train()
    for _ in range(3):
        (audio, _, yq_ds, x_ds, vqloss, _, f0_hat, _, pq_ds, px_ds,
         p_vqloss, _, _) = net(spec, spec_l, f0, sid)
        if vqloss.requires_grad or p_vqloss.requires_grad:
            print('  ERROR: vqloss is not gradient-free with the codecs frozen')
            return 1
        if x_ds.requires_grad or px_ds.requires_grad:
            print('  ERROR: an encoder latent still requires grad')
            return 1
        loss = (audio.pow(2).mean() + f0_hat.pow(2).mean()
                + yq_ds.pow(2).mean() + pq_ds.pow(2).mean())
        opt.zero_grad()
        loss.backward()
        opt.step()

    moved = [n for n, p in net.named_parameters()
             if n in frozen and not torch.equal(p.detach(), frozen[n])]
    moved += [n for n, x in net.named_buffers() if not torch.equal(x, buffers[n])]
    if moved:
        print('  ERROR: frozen tensors changed:', moved[:10])
        return 1
    print('  frozen weights + codebook buffers: unchanged')

    net.eval()
    new_spec, new_pitch = net.encode_codes(spec, spec_l, f0, sid)
    ok = report_match('spec', ref_spec, new_spec)
    ok = report_match('pitch', ref_pitch, new_pitch) and ok
    return 0 if ok else 1


def checkpoint_check(config_path: str, checkpoint: str, compare: str,
                     filelist: str, max_items: int) -> int:
    """Compare the codes two stage-1 checkpoints produce on the same items."""
    items = []
    with open(filelist) as fh:
        for line in fh:
            parts = line.strip().split('|')
            if not parts or not parts[0]:
                continue
            base = os.path.splitext(parts[0])[0]
            try:
                spec = torch.load(base + '.spec', weights_only=True)
                f0 = torch.load(base + '.f0', weights_only=True)
            except FileNotFoundError:
                print(f'  skipping {base}: tensors not found')
                continue
            t = min(800, spec.shape[0], f0.shape[0])
            items.append((spec[:t].T.unsqueeze(0).float(),
                          f0[:t].unsqueeze(0).float()))
            if len(items) >= max_items:
                break
    if not items:
        print('no usable items in', filelist)
        return 1
    print(f'  comparing on {len(items)} item(s) from {filelist}')

    codes = {}
    for tag, path in (('ref', checkpoint), ('cmp', compare)):
        net = build_generator(config_path, path)
        out = []
        with torch.no_grad():
            for spec, f0 in items:
                out.append(net.encode_codes(
                    spec, torch.tensor([spec.shape[-1]]), f0,
                    torch.zeros(1, dtype=torch.long)))
        codes[tag] = out
        del net

    ok = True
    for i, ((rs, rp), (cs, cp)) in enumerate(zip(codes['ref'], codes['cmp'])):
        print(f'item {i}:')
        ok = report_match('spec', rs, cs) and ok
        ok = report_match('pitch', rp, cp) and ok
    print('\nCODES IDENTICAL' if ok else '\nCODES MOVED - stage-2 codes must '
          'be re-extracted for the fine-tuned checkpoint')
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='configs/base.yaml')
    parser.add_argument('--checkpoint', default=None,
                        help='Base stage-1 checkpoint (enables the checkpoint '
                             'comparison; requires --compare).')
    parser.add_argument('--compare', default=None,
                        help='Fine-tuned stage-1 checkpoint to compare against.')
    parser.add_argument('--filelist', default='data/base/val.txt',
                        help='Stage-1 filelist used to pick items; the .spec/'
                             '.f0 tensors are read next to the listed .wave.')
    parser.add_argument('--max_items', type=int, default=4)
    args = parser.parse_args(argv)

    if args.checkpoint or args.compare:
        if not (args.checkpoint and args.compare):
            parser.error('pass both --checkpoint and --compare')
        print('=== checkpoint comparison: codes must be identical ===')
        return checkpoint_check(args.config, args.checkpoint, args.compare,
                                args.filelist, args.max_items)

    print('=== graph check: freeze -> 3 steps -> codes must be identical ===')
    return graph_check(args.config)


if __name__ == '__main__':
    sys.exit(main())
