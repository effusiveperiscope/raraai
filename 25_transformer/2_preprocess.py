import argparse
import itertools
import math
import os
import random

import librosa
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

import speaker_map as speaker_map_lib
from commons import load_state_dict_mismatch, load_submodule_prefix
from features import FeatureExtractor
from modeling.generator import Generator
from utils import win_longpath

import logging
# Something above this is setting DEBUG but I don't know what
logging.getLogger().setLevel(logging.WARNING)

# Tensors written per source file, in the order they appear in the filelist.
FEATURE_KEYS = ['phone_ids', 'spec_codes', 'pitch_codes']

# Window used by the silence filter. 20 ms spans a couple of periods even for
# a low male voice (~60 Hz -> 16.7 ms), so a voiced frame scores close to its
# true level instead of being split across two windows.
SILENCE_FRAME_MS = 20.0

# Sentinel for "no threshold": every real level compares greater than -inf
# (digital silence is clamped to ~-200 dBFS, see `audio_level_stats`).
NO_DB_FLOOR = float('-inf')

# Default decision level. Speech is normally far above it (loudest 20 ms
# frames of a normalised corpus sit around -20..-6 dBFS), so this only drops
# clips that are effectively empty, not merely quiet recordings.
DEFAULT_MIN_RMS_DB = -50.0


def feature_paths(output_dir, base_line, keys=FEATURE_KEYS):
    """Return the on-disk tensor paths for one source file.

    Tensors are named by *basename* only (see AGENTS.md footgun 11).
    """
    return [win_longpath(os.path.join(output_dir,
        os.path.basename(base_line) + '.' + key)) for key in keys]


def audio_level_stats(wav, sr, frame_ms=SILENCE_FRAME_MS,
                      active_db=-50.0):
    """Level statistics used to detect silent / near-silent clips.

    Args:
        wav: 1-D mono float waveform in [-1, 1].
        sr: Sample rate of `wav`.
        frame_ms: Analysis window length in milliseconds.
        active_db: A frame counts as *active* when its RMS reaches this level
            (dBFS).

    Returns:
        ``(rms_db, peak_db, active_frac)`` where `rms_db` is the RMS level of
        the *loudest* frame in dBFS (a clip's usable level -- unlike the RMS
        of the whole clip, it does not shrink just because the clip has
        long silences), `peak_db` is the absolute peak sample level in dBFS
        and `active_frac` is the fraction of frames at or above `active_db`.
        Levels are clamped at -200 dBFS so digital silence is a large finite
        number rather than ``-inf``/NaN.
    """
    wav = np.asarray(wav, dtype=np.float64).reshape(-1)
    if wav.size == 0:
        return NO_DB_FLOOR, NO_DB_FLOOR, 0.0

    frame_len = max(1, int(round(sr * frame_ms / 1000.0)))
    # `max(1, ...)` keeps clips shorter than one window usable: the whole clip
    # then forms a single frame.
    n_frames = max(1, wav.size // frame_len)
    frames = wav[:n_frames * frame_len].reshape(n_frames, -1)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
    frame_db = 20.0 * np.log10(np.maximum(frame_rms, 1e-10))

    peak = float(np.max(np.abs(wav)))
    peak_db = 20.0 * math.log10(max(peak, 1e-10))
    active_frac = float(np.mean(frame_db >= active_db))
    return float(np.max(frame_db)), peak_db, active_frac


def is_silent(wav, sr, min_rms_db=DEFAULT_MIN_RMS_DB, min_speech_frac=0.0,
              frame_ms=SILENCE_FRAME_MS):
    """Decide whether a clip is silent or near-silent.

    A clip is dropped when either

    * its loudest `frame_ms` window is below `min_rms_db` dBFS, i.e. the whole
      recording is a dead or nearly dead channel, or
    * fewer than `min_speech_frac` of its frames reach `min_rms_db`, i.e. the
      clip is almost entirely silence with a stray blip in it.

    Args:
        wav: 1-D mono float waveform in [-1, 1].
        sr: Sample rate of `wav`.
        min_rms_db: Level (dBFS) a frame must reach to count as audio. Pass
            `NO_DB_FLOOR` (``-inf``) to disable the filter entirely.
        min_speech_frac: Minimum fraction of active frames; `0` disables the
            "mostly silence" half of the check.
        frame_ms: Analysis window length in milliseconds.

    Returns:
        ``(silent, stats)`` where `stats` holds the measured ``rms_db``,
        ``peak_db`` and ``active_frac`` for logging. It is empty when the
        filter is disabled.
    """
    if min_rms_db is None or not min_rms_db > NO_DB_FLOOR:
        return False, {}

    rms_db, peak_db, active_frac = audio_level_stats(
        wav, sr, frame_ms=frame_ms, active_db=min_rms_db)
    silent = (rms_db < min_rms_db
              or (min_speech_frac > 0 and active_frac < min_speech_frac))
    return silent, {
        'rms_db': rms_db,
        'peak_db': peak_db,
        'active_frac': active_frac,
    }


def db_floor(value):
    """argparse type for a dBFS threshold; ``none``/``off`` disables it.

    argparse rejects a bare ``-inf`` argument (it looks like an option), so
    accept a readable word for the same thing.
    """
    text = str(value).strip().lower()
    if text in ('none', 'off', 'disable', 'disabled', '-inf', 'inf'):
        return NO_DB_FLOOR
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f'expected a level in dBFS or "none", got {value!r}')


def load_generator(checkpoint_path, config, device):
    net_g = Generator(hp=config)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    if any(k.startswith('net_g.') for k in state.keys()):
        load_submodule_prefix(net_g, 'net_g.', state)
    else:
        load_state_dict_mismatch(net_g, state)
    net_g.to(device)
    net_g.eval()
    return net_g

def process_filelist(filelist_path,
    val_fraction=0.05, output_dir='data/preprocessed', shuffle_seed=42,
    skip_if_one_exists=False, checkpoint_path=None, config_path='configs/base.yaml',
    device=None, speaker_map_path=None, min_rms_db=DEFAULT_MIN_RMS_DB,
    min_speech_frac=0.0):
    if checkpoint_path is None:
        raise ValueError('--checkpoint is required: path to a 1_training checkpoint')

    with open(filelist_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]

    os.makedirs(output_dir, exist_ok=True)

    random.seed(shuffle_seed)
    random.shuffle(lines)

    hp = OmegaConf.load(config_path)
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    fe = FeatureExtractor(hp=hp)
    net_g = load_generator(checkpoint_path, hp, device)

    # Optional canonicalisation of speaker ids into the stage-1 space so a
    # single id works for both the stage-2 transformer and the frozen
    # Generator's speaker embedding (see speaker_map.py).
    smap = None
    if speaker_map_path is not None:
        smap = speaker_map_lib.load_speaker_map(speaker_map_path)
        print(f'Using speaker map {speaker_map_path}')

    new_lines = []
    n_skipped_speaker = 0
    n_skipped_silent = 0
    n_reused = 0

    # `--min_rms_db none` (NO_DB_FLOOR) turns the silence filter off.
    silence_filter = min_rms_db is not None and min_rms_db > NO_DB_FLOOR
    if silence_filter:
        print(f'Silence filter: dropping clips whose loudest '
              f'{SILENCE_FRAME_MS:.0f} ms frame is below {min_rms_db:.1f} dBFS'
              + (f' or that have less than {min_speech_frac:.1%} active frames'
                 if min_speech_frac > 0 else ''))
    else:
        print('Silence filter disabled (--min_rms_db none)')

    for line in tqdm(lines, total=len(lines), desc='Preprocessing'):
        if not line:
            continue
        if 'longform' in line:
            continue
        split = line.split('|')

        if len(split) != 3: # skip malformed lines
            continue

        line = split[0]
        sid = split[1]
        transcr = split[2]

        base_line = line
        line = win_longpath(line)

        if smap is not None:
            resolved = speaker_map_lib.resolve_sid(
                smap, path=base_line, raw_sid=int(sid))
            if resolved is None:
                n_skipped_speaker += 1
                continue
            sid = resolved
        else:
            sid = int(sid)

        savepaths = feature_paths(output_dir, base_line)
        reuse = (skip_if_one_exists
                 and all(os.path.exists(p) for p in savepaths))
        # A reused clip needs no audio unless the silence filter has to
        # re-check it, which is what makes `--skip_existing` usable for
        # retroactively dropping silent clips from an existing corpus.
        if reuse and not silence_filter:
            n_reused += 1
            new_lines.append('|'.join(savepaths) + f'|{sid}')
            continue

        if not os.path.exists(line):
            print(f'File not found: {line}')
            continue

        # 48 kHz is the stage-1 feature rate, and the once-per-clip level check
        # is done on it too.
        wav_48k, _ = librosa.load(line, sr=48000)

        if silence_filter:
            silent, stats = is_silent(
                wav_48k, 48000, min_rms_db=min_rms_db,
                min_speech_frac=min_speech_frac)
            if silent:
                print(f'Skipping silent/near-silent {base_line}: loudest '
                      f'frame {stats["rms_db"]:.1f} dBFS, peak '
                      f'{stats["peak_db"]:.1f} dBFS, '
                      f'{stats["active_frac"]:.1%} active frames')
                n_skipped_silent += 1
                continue
            if reuse:
                n_reused += 1
                new_lines.append('|'.join(savepaths) + f'|{sid}')
                continue

        wav_16k, _ = librosa.load(line, sr=16000)
        savepaths = []
        try:

            feats, _ = fe.extract(wav_16k, transcription=transcr)
            feats_ac = fe.extract_features_ac(wav_48k)
            phone_ids = list(itertools.chain.from_iterable(
                [f['phone_ids'] for f in feats]))

            # Encode raw spec/f0 into discrete RVQ codes with the 1_training
            # Generator checkpoint (channel-first spec, frame-rate f0).
            spec = feats_ac['spec']  # [T, C]
            f0 = feats_ac['f0'].float()  # [T]
            spec_b = spec.transpose(0, 1).unsqueeze(0).to(device)  # [1, C, T]
            spec_l = torch.tensor([spec.shape[0]], device=device)
            f0_b = f0.unsqueeze(0).to(device)  # [1, T]
            sid_b = torch.tensor([int(sid)], device=device)
            with torch.no_grad():
                spec_codes, pitch_codes = net_g.encode_codes(
                    spec_b, spec_l, f0_b, sid_b)
            # encode_codes returns (Q, B, T_ds); drop batch dim -> (Q, T_ds).
            spec_codes = spec_codes.squeeze(1).cpu()
            pitch_codes = pitch_codes.squeeze(1).cpu()

            feats_n = {
                'phone_ids': torch.tensor(phone_ids),
                'spec_codes': spec_codes,
                'pitch_codes': pitch_codes,
            }
            # phone_ids [n_phone]
            # spec_codes [Q_spec, T_codes] (Q_spec = codec.codebook_num)
            # pitch_codes [Q_pitch, T_codes] (Q_pitch = pitch_codec.codebook_num)
            for key, value in feats_n.items():
                savepath = os.path.join(output_dir,
                    os.path.basename(base_line) + '.' + key)
                savepath = win_longpath(savepath)
                torch.save(value, savepath)
                savepaths.append(savepath)

        except Exception as e:
            print(f'Error processing {line}: {e}')
            raise e
            continue

        newline = '|'.join(savepaths)+f'|{sid}'
        new_lines.append(newline)

    if smap is not None and n_skipped_speaker:
        print(f'Skipped {n_skipped_speaker} file(s) with no speaker-map match')
    print(f'Kept {len(new_lines)} file(s) ({n_reused} reused); '
          f'omitted {n_skipped_silent} silent/near-silent clip(s)'
          + (f' below {min_rms_db:.1f} dBFS' if silence_filter else '')
          + '.')

    # Split the *kept* items, not the raw line count: omitting clips must not
    # shrink train and overflow val.
    if not new_lines:
        raise RuntimeError(
            f'No clips from {filelist_path} survived validation, so there is '
            'nothing to write to train_stage2.txt / val_stage2.txt. Lower '
            '--min_rms_db (or pass "--min_rms_db none" to disable the silence '
            'filter) and re-run.')
    n_valid = len(new_lines)
    if val_fraction > 0 and n_valid > 1:
        val_size = max(1, int(n_valid * val_fraction))
        val_size = min(val_size, n_valid - 1)
        val_lines = new_lines[-val_size:]
        train_lines = new_lines[:-val_size]
    else:
        val_lines = new_lines[:1] # ensure at least one item in val for checkpointing
        train_lines = new_lines

    with open(os.path.join(output_dir, 'train_stage2.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_lines))
    with open(os.path.join(output_dir, 'val_stage2.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_lines))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', type=str, help='path to filelist', required=True)
    parser.add_argument('--val_fraction', type=float, default=0.05)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--shuffle_seed', type=int, default=42)
    parser.add_argument('--checkpoint', type=str, required=True,
        help='path to 1_training (lightning or Generator) checkpoint')
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--speaker_map', type=str, default=None,
        help='JSON map canonicalising speaker ids to the stage-1 space '
             '(see speaker_map.py)')
    parser.add_argument('--min_rms_db', type=db_floor, default=DEFAULT_MIN_RMS_DB,
        help='drop clips that are silent or near-silent: a clip is skipped '
             'when the RMS of its loudest 20 ms frame is below this level in '
             'dBFS (default %(default)s). Pass "none" to disable the filter; '
             'raise it (e.g. -35) to also drop very quiet recordings. '
             'With --skip_existing the level check is re-run on already '
             'encoded clips (audio only, no re-extraction).')
    parser.add_argument('--min_speech_frac', type=float, default=0.0,
        help='also skip clips where less than this fraction of 20 ms frames '
             'reach --min_rms_db, i.e. clips that are almost entirely '
             'silence (default %(default)s = only the level check is used)')

    args = parser.parse_args()

    process_filelist(
        filelist_path=args.filelist,
        val_fraction=args.val_fraction,
        output_dir=args.output_dir,
        shuffle_seed=args.shuffle_seed,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        speaker_map_path=args.speaker_map,
        min_rms_db=args.min_rms_db,
        min_speech_frac=args.min_speech_frac,
    )
