#!/usr/bin/env python3
"""Merge the pitch codec weights of one stage-1 checkpoint into another.

Stage-1 ``--pitch_only`` fine-tuning (``1_training.py --pitch_only``) updates
*only* the pitch path of the ``Generator`` (see ``Generator.freeze_non_pitch``
/ :data:`modeling.generator.PITCH_PATH_PREFIXES`) and leaves everything else —
posterior encoder, spectrogram codec, resampling layers, NSF decoder, speaker
table, discriminators — untouched.  This script splices that pitch path into a
*different* stage-1 run, which is how e.g. a long pitch-codec fine-tune is
combined with the main run that owns the rest of the model:

    net_g.pitch_downsample.*     <- --pitch_checkpoint
    net_g.pitch_codec.*          <- --pitch_checkpoint
    net_g.pitch_upsample.*       <- --pitch_checkpoint
    net_g.pitch_out_proj.*       <- --pitch_checkpoint
    net_g.pitch_uv_proj.*        <- --pitch_checkpoint
    everything else              <- --base_checkpoint

The two checkpoints must have identical key sets and identical shapes (i.e. the
same architecture, normally the same ``configs/base.yaml``); that is checked
before anything is written.  The output is a normal Lightning checkpoint: the
base checkpoint's metadata (``epoch``, ``global_step``, ``loops``,
``callbacks``) is kept so ``2_preprocess.py --checkpoint``,
``scratch/test_stage1_codes.py`` and ``train.generator_checkpoint`` consume it as
usual.

Optimizer state is **dropped by default**: the base run's Adam moments do not
belong to the spliced-in pitch weights and Lightning matches them by position,
so keeping them would silently apply the wrong moments to the pitch path.
Continue a stage-1 run from the merged file with ``--transfer_from`` (weights
only), not ``--resume_from``.  ``--keep_optimizer_state`` keeps the base run's
state for the case where that is what you want.

Usage:
    python merge_pitch_codec.py \
        --pitch_checkpoint pretrain/stage1_pitch_last.ckpt \
        --base_checkpoint pretrain/5090_stage1_last.ckpt \
        --output pretrain/stage1_merged_last.ckpt
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import torch

# Single source of truth: the same tuple 1_training.py --pitch_only unfreezes.
from modeling.generator import PITCH_PATH_PREFIXES

DEFAULT_MODULE_PREFIX = 'net_g.'
PITCH_DISC_PREFIX = 'net_dp.'


def load_checkpoint(path: str) -> dict:
    """Load a Lightning checkpoint, returning the raw dict."""
    if not os.path.isfile(path):
        raise SystemExit(f'checkpoint not found: {path}')
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict):
        raise SystemExit(f'{path} is not a Lightning checkpoint (got '
                         f'{type(ckpt).__name__})')
    if 'state_dict' not in ckpt:
        raise SystemExit(f'{path} has no "state_dict" entry; expected a '
                         f'Lightning checkpoint')
    return ckpt


def select_pitch_keys(state_dict: Dict[str, torch.Tensor],
                      prefix: str = DEFAULT_MODULE_PREFIX,
                      include_pitch_disc: bool = False) -> list[str]:
    """Return the state-dict keys that make up the pitch codec path.

    Args:
        state_dict: the checkpoint's ``state_dict``.
        prefix: module prefix the parameters are stored under (``net_g.`` for
            the ``Generator`` in ``1_training.py`` checkpoints).
        include_pitch_disc: also take the pitch discriminator (``net_dp.*``)
            from the pitch checkpoint. It is trained by ``--pitch_only`` too,
            but it is not part of the pitch *codec*, so this is off by default.

    Returns:
        The matching keys, sorted — exactly the ones
        ``Generator.freeze_non_pitch`` keeps trainable, plus ``net_dp.*`` when
        requested.
    """
    prefixes = tuple(prefix + p for p in PITCH_PATH_PREFIXES)
    keys = [k for k in state_dict if k.startswith(prefixes)]
    if include_pitch_disc:
        keys += [k for k in state_dict if k.startswith(PITCH_DISC_PREFIX)]
    keys.sort()
    if not keys:
        raise SystemExit(
            f'no pitch codec keys found under {prefix!r} — is this a stage-1 '
            f'Generator checkpoint? Expected keys starting with '
            f'{[prefix + p for p in PITCH_PATH_PREFIXES]}')
    return keys


def check_compatible(base_state: Dict[str, torch.Tensor],
                     pitch_state: Dict[str, torch.Tensor],
                     pitch_keys: list[str]) -> None:
    """Fail loudly unless both checkpoints describe the same architecture."""
    base_only = sorted(set(base_state) - set(pitch_state))
    pitch_only = sorted(set(pitch_state) - set(base_state))
    if base_only or pitch_only:
        msg = ['the two checkpoints have different key sets:']
        if base_only:
            msg.append(f'  only in --base_checkpoint ({len(base_only)}): '
                       f'{base_only[:5]}')
        if pitch_only:
            msg.append(f'  only in --pitch_checkpoint ({len(pitch_only)}): '
                       f'{pitch_only[:5]}')
        msg.append('Both must have been trained with the same architecture '
                   '(normally the same configs/base.yaml).')
        raise SystemExit('\n'.join(msg))

    shape_bad = [(k, tuple(base_state[k].shape), tuple(pitch_state[k].shape))
                 for k in pitch_keys
                 if base_state[k].shape != pitch_state[k].shape]
    dtype_bad = [(k, base_state[k].dtype, pitch_state[k].dtype)
                 for k in pitch_keys
                 if base_state[k].dtype != pitch_state[k].dtype]
    for title, bad in (('shape', shape_bad), ('dtype', dtype_bad)):
        if bad:
            lines = [f'{title} mismatch between the two checkpoints:']
            lines += [f'  {k}: base {b} vs pitch {p}' for k, b, p in bad[:10]]
            raise SystemExit('\n'.join(lines))


def merge_state_dicts(base_state: Dict[str, torch.Tensor],
                      pitch_state: Dict[str, torch.Tensor],
                      pitch_keys: list[str]) -> tuple[dict, list[str], list[str]]:
    """Return a new state dict with the pitch keys taken from ``pitch_state``.

    Returns:
        ``(merged, changed, unchanged)`` where ``changed``/``unchanged`` split
        the overridden keys by whether they already agreed (a fully frozen path
        is bit-identical, so that split is the quickest sanity check that the
        pitch checkpoint really is a fine-tune of the base).
    """
    merged = dict(base_state)
    changed, unchanged = [], []
    for key in pitch_keys:
        merged[key] = pitch_state[key]
        (unchanged if torch.equal(base_state[key], pitch_state[key])
         else changed).append(key)
    return merged, changed, unchanged


def verify_merged(merged_state: Dict[str, torch.Tensor],
                  base_state: Dict[str, torch.Tensor],
                  pitch_state: Dict[str, torch.Tensor],
                  pitch_keys: list[str]) -> None:
    """Assert every merged tensor still equals the intended source."""
    pitch_set = set(pitch_keys)
    if set(merged_state) != set(base_state):
        raise SystemExit('merge changed the key set')
    for key, tensor in merged_state.items():
        source = pitch_state[key] if key in pitch_set else base_state[key]
        if not torch.equal(tensor, source):
            raise SystemExit(f'merge verification failed for {key}')


def verify_with_model(merged_state: Dict[str, torch.Tensor],
                      config_path: str,
                      prefix: str) -> None:
    """Build a Generator from ``config_path`` and load the merged weights.

    ``load_state_dict_mismatch`` silently skips tensors whose shape disagrees,
    so a config/checkpoint mismatch can go unnoticed; a strict
    ``load_state_dict`` is the cheap proof that the merge produced a usable
    Generator.
    """
    from omegaconf import OmegaConf
    from modeling.generator import Generator

    config = OmegaConf.load(config_path)
    net_g = Generator(hp=config)
    state = {k[len(prefix):]: v for k, v in merged_state.items()
             if k.startswith(prefix)}
    try:
        net_g.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise SystemExit(
            f'merged {prefix!r} tensors do not match the Generator built from '
            f'{config_path}: {exc}')
    print(f'Strict load into Generator({config_path}) OK: {len(state)} tensors, '
          f'{sum(p.numel() for p in net_g.parameters())/1e6:.2f} M params')


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pitch_checkpoint', required=True,
                        help='stage-1 checkpoint to take the pitch codec path '
                             'from (e.g. a base_v0.6_pitchonly run)')
    parser.add_argument('--base_checkpoint', required=True,
                        help='stage-1 checkpoint to take everything else from')
    parser.add_argument('--output', required=True,
                        help='path of the merged checkpoint to write')
    parser.add_argument('--module_prefix', default=DEFAULT_MODULE_PREFIX,
                        help='state-dict prefix of the Generator parameters '
                             f'(default {DEFAULT_MODULE_PREFIX!r})')
    parser.add_argument('--include_pitch_disc', action='store_true',
                        help='also take the pitch discriminator (net_dp.*) '
                             'from --pitch_checkpoint')
    parser.add_argument('--keep_optimizer_state', action='store_true',
                        help="keep the base checkpoint's optimizer_states / "
                             'lr_schedulers in the merged checkpoint (off by '
                             'default: they were not trained on the spliced-in '
                             'pitch weights)')
    parser.add_argument('--config', default='configs/base.yaml',
                        help='config used to verify the merged Generator loads '
                             'strictly (default configs/base.yaml)')
    parser.add_argument('--no_verify', action='store_true',
                        help='skip building the Generator for the strict load '
                             'check (the tensor-level checks still run)')
    args = parser.parse_args()

    out_abs = os.path.abspath(args.output)
    if out_abs in (os.path.abspath(args.base_checkpoint),
                   os.path.abspath(args.pitch_checkpoint)):
        raise SystemExit('--output must not overwrite either input checkpoint')
    if os.path.exists(out_abs):
        raise SystemExit(f'refusing to overwrite existing file {args.output}')

    print(f'Base  checkpoint: {args.base_checkpoint}')
    print(f'Pitch checkpoint: {args.pitch_checkpoint}')
    base_ckpt = load_checkpoint(args.base_checkpoint)
    pitch_ckpt = load_checkpoint(args.pitch_checkpoint)
    base_state = base_ckpt['state_dict']
    pitch_state = pitch_ckpt['state_dict']
    for name, ckpt in (('base', base_ckpt), ('pitch', pitch_ckpt)):
        print(f'  {name:<5s} epoch={ckpt.get("epoch")} '
              f'global_step={ckpt.get("global_step")} '
              f'({len(ckpt["state_dict"])} tensors)')

    pitch_keys = select_pitch_keys(base_state, args.module_prefix,
                                   args.include_pitch_disc)
    check_compatible(base_state, pitch_state, pitch_keys)
    merged_state, changed, unchanged = merge_state_dicts(
        base_state, pitch_state, pitch_keys)
    verify_merged(merged_state, base_state, pitch_state, pitch_keys)

    n_params = sum(merged_state[k].numel() for k in pitch_keys)
    print(f'\nTook {len(pitch_keys)} tensors ({n_params/1e6:.2f} M params) '
          f'from the pitch checkpoint:')
    for p in PITCH_PATH_PREFIXES:
        n = sum(1 for k in pitch_keys if k.startswith(args.module_prefix + p))
        print(f'  {args.module_prefix}{p:<20s} {n:3d} tensors')
    if args.include_pitch_disc:
        n = sum(1 for k in pitch_keys if k.startswith(PITCH_DISC_PREFIX))
        print(f'  {PITCH_DISC_PREFIX:<25s} {n:3d} tensors')
    print(f'  differing from the base: {len(changed)}')
    print(f'  identical to the base:   {len(unchanged)} '
          f'(frozen in the pitch-only run)')

    merged = {
        'epoch': base_ckpt.get('epoch'),
        'global_step': base_ckpt.get('global_step'),
        'pytorch-lightning_version': base_ckpt.get(
            'pytorch-lightning_version'),
        'state_dict': merged_state,
        'loops': base_ckpt.get('loops', {}),
        'callbacks': base_ckpt.get('callbacks', {}),
    }
    if args.keep_optimizer_state:
        for key in ('optimizer_states', 'lr_schedulers'):
            if key in base_ckpt:
                merged[key] = base_ckpt[key]
        print('\nKeeping the base checkpoint optimizer state: --resume_from '
              'reuses Adam moments that were never trained on the merged pitch '
              'weights.')
    else:
        print('\nDropped optimizer_states/lr_schedulers (they belong to the '
              'base run, not to the spliced-in pitch weights). Continue a '
              'stage-1 run from this file with --transfer_from.')

    merged['merge_provenance'] = {
        'base_checkpoint': os.path.abspath(args.base_checkpoint),
        'base_epoch': base_ckpt.get('epoch'),
        'base_global_step': base_ckpt.get('global_step'),
        'pitch_checkpoint': os.path.abspath(args.pitch_checkpoint),
        'pitch_epoch': pitch_ckpt.get('epoch'),
        'pitch_global_step': pitch_ckpt.get('global_step'),
        'pitch_keys': pitch_keys,
    }

    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    torch.save(merged, args.output)
    print(f'\nWrote {args.output} ({os.path.getsize(args.output)/1e6:.1f} MB)')

    if not args.no_verify:
        verify_with_model(merged_state, args.config, args.module_prefix)


if __name__ == '__main__':
    main()
