import argparse
import itertools
import logging
import math
import os
import random

# Claim the root logger's handler slot *before* the heavy imports; see the long
# comment at the top of 2_training.py.  `modeling.generator` (imported below)
# pulls in the vendored `svc_helper.svc.rvc` -> `fairseq`, which runs
# `logging.basicConfig(level=logging.DEBUG)` at import time; that is a no-op only
# if a handler already exists.  The reset below used to be the only line, and it
# ran *after* the imports had already enabled DEBUG.
logging.basicConfig(level=logging.WARNING)

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

logging.getLogger().setLevel(logging.WARNING)

# Tensors written per source file, in the order they appear in the filelist.
FEATURE_KEYS = ['phone_ids', 'spec_codes', 'pitch_codes']

# The three tensors come from two independent stages: the text side
# (`phone_ids`, cheap, CPU) and the acoustic side (both code tensors, produced
# together by one `Generator.encode_codes` call). `--only` exploits that.
PHONE_KEY = 'phone_ids'
CODE_KEYS = ('spec_codes', 'pitch_codes')

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


def tensor_shape(path, key):
    """Header-read a single stage-2 tensor.

    Args:
        path: Tensor file written by this script.
        key: Tensor suffix (`phone_ids` / `spec_codes` / `pitch_codes`), used
            for the error message only.

    Returns:
        ``(shape, reason)``: ``shape`` is a tuple, or ``None`` with a short
        human-readable ``reason`` when the file is missing/unreadable.
        ``mmap=True`` maps the file instead of reading it, so checking a whole
        corpus is cheap; filesystems without mmap fall back to a full load.
    """
    if not os.path.exists(path):
        return None, f'{key} missing'
    try:
        value = torch.load(path, map_location='cpu', weights_only=True,
                           mmap=True)
    except (RuntimeError, OSError, ValueError):
        # Filesystems without mmap support (or an older save format).
        try:
            value = torch.load(path, map_location='cpu', weights_only=True)
        except Exception as e:  # corrupt / truncated file
            return None, f'unreadable {key} ({e})'
    return tuple(value.shape), ''


def key_shape_ok(key, shape):
    """Does one tensor's header satisfy what stage 2 requires?

    The checks mirror ``dataset_codes.py`` / ``Generator.infer_from_codes``.
    Each tensor is validated on its own so ``--only`` can rebuild just the one
    that is broken (the cross-tensor frame-count check lives in
    `code_lengths_agree`).
    """
    if key == 'phone_ids':
        return len(shape) == 1 and shape[0] > 0
    return len(shape) == 2 and shape[0] > 0 and shape[1] > 0


def code_lengths_agree(shapes):
    """``(ok, reason)``: spec_codes and pitch_codes need the same frame count.

    They come out of a single ``encode_codes`` call (AGENTS.md footgun 7), so a
    mismatch means one of them is stale/truncated.
    """
    n_spec = shapes['spec_codes'][1]
    n_pitch = shapes['pitch_codes'][1]
    if n_spec != n_pitch:
        return False, (f'spec_codes/pitch_codes frame counts differ '
                       f'({n_spec} vs {n_pitch})')
    return True, ''


# `--only` shorthands: name -> tensors it selects.
ONLY_ALIASES = {
    'phone': ['phone_ids'],
    'phones': ['phone_ids'],
    'text': ['phone_ids'],
    'code': ['spec_codes', 'pitch_codes'],
    'codes': ['spec_codes', 'pitch_codes'],
    'acoustic': ['spec_codes', 'pitch_codes'],
}


def parse_only(value):
    """argparse type for ``--only``: which tensors to (re)build.

    Accepts ``all`` (the default) or a comma-separated list of
    ``phone_ids``/``spec_codes``/``pitch_codes`` plus the ``phones`` and
    ``codes`` shorthands. The selection is returned in `FEATURE_KEYS` order.

    ``spec_codes`` and ``pitch_codes`` are always selected together: one
    ``encode_codes`` call produces both and they must keep the same frame
    count, so rebuilding only one would leave an inconsistent pair.
    """
    text = str(value).strip().lower()
    if text in ('', 'all', 'everything', 'both'):
        return list(FEATURE_KEYS)
    selected = set()
    for token in text.replace('+', ',').split(','):
        token = token.strip()
        if not token:
            continue
        if token in ONLY_ALIASES:
            selected.update(ONLY_ALIASES[token])
        elif token in FEATURE_KEYS:
            selected.add(token)
        else:
            raise argparse.ArgumentTypeError(
                f'unknown tensor {token!r}; expected one of '
                f'{", ".join(FEATURE_KEYS)} (shorthands: phones, codes, all)')
    if selected & {'spec_codes', 'pitch_codes'}:
        selected |= {'spec_codes', 'pitch_codes'}
    if not selected:
        raise argparse.ArgumentTypeError('--only selected no tensors')
    return [key for key in FEATURE_KEYS if key in selected]


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
    min_speech_frac=0.0, only=None, force=False, phones_from='aligner'):
    """Extract stage-2 (phone_ids, spec_codes, pitch_codes) for a filelist.

    The three tensors are produced by independent stages, and each one is
    written, validated and reused *on its own*, so a change to one stage no
    longer forces the others to be recomputed:

    * ``phone_ids`` — phonemizer / dictionary changes (`--only phones`),
      optionally from the transcript alone (`--phones_from transcript`: no
      audio, no ASR, no GPU).
    * ``spec_codes`` + ``pitch_codes`` — codec / Generator changes
      (`--only codes`), which skip the ASR/aligner weights entirely.

    Args:
        filelist_path: ``audio_path|sid|transcription`` lines.
        val_fraction: Fraction of the *kept* clips written to
            `val_stage2.txt`; the rest goes to `train_stage2.txt`.
        output_dir: Where tensors and both filelists are written.
        shuffle_seed: Seed for the pre-split shuffle.
        skip_if_one_exists: Never overwrite a usable tensor: `selected` keys
            that already exist are reused instead of recomputed (combine with
            `force` to redo one of them). Tensors are header-checked one by
            one (`key_shape_ok` / `code_lengths_agree`), so a corrupt file is
            rebuilt instead of silently entering the filelist. The silence
            filter still runs, which loads the audio but performs no feature
            extraction, so silent clips can be dropped from an
            already-encoded corpus.
        checkpoint_path: Stage-1 checkpoint used (frozen) as the codec. Only
            needed when code tensors are (re)built; a phones-only run may omit
            it.
        config_path: Stage-1 config matching `checkpoint_path`.
        device: Torch device for the Generator; defaults to cuda if present.
        speaker_map_path: JSON map canonicalising ids into the stage-1 space
            (see speaker_map.py).
        min_rms_db: Silence filter threshold in dBFS (`NO_DB_FLOOR` disables
            the filter). See `is_silent`.
        min_speech_frac: Also drop clips with fewer active frames than this.
        only: Tensors to (re)build, in `FEATURE_KEYS` order; see `parse_only`.
            ``None`` (default) rebuilds all three. Keys *not* listed are reused
            from `output_dir` and the clip is skipped if one of them is
            missing/unusable.
        force: Rebuild the `only` tensors even when `skip_if_one_exists` would
            reuse them.
        phones_from: ``aligner`` (default) derives the words from the forced
            alignment, exactly as a full run does. ``transcript`` phonemizes
            the filelist transcript directly (`FeatureExtractor.
            phonemize_text`), which needs no audio/ASR at all, but it cannot
            tell that a clip has no speakable text.  Both paths tokenize the
            words as one sentence and therefore produce the same `phone_ids`
            for the same text -- the aligner path only differs in that it knows
            the word timings and that it drops a clip whose alignment covers no
            words at all.
            Punctuation is carried into `phone_ids` either way: the aligner
            strips it from its tokens, so `FeatureExtractor.align` restores it
            from the transcript / ASR text (see `features.attach_punctuation`).

    Clips whose ASR/forced alignment produce an empty phoneme sequence (very
    short interjections, usually) are dropped when `phone_ids` is rebuilt: the
    codes are valid but there is no text to condition on, so the clip could
    only train the transformer as an unconditional sample. Codes are then not
    written either, so a later `--skip_existing` run does not pick them up.
    """
    if phones_from not in ('aligner', 'transcript'):
        raise ValueError("phones_from must be 'aligner' or 'transcript', "
                         f'got {phones_from!r}')
    if isinstance(only, str):
        only = parse_only(only)
    selected = [key for key in FEATURE_KEYS
                if key in set(only if only is not None else FEATURE_KEYS)]
    if not selected:
        raise ValueError('only=... selected no tensors')
    if any(key in CODE_KEYS for key in selected) and checkpoint_path is None:
        raise ValueError(
            '--checkpoint is required to (re)build spec/pitch codes; a '
            'phones-only run (--only phones) does not need it')

    with open(filelist_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]

    os.makedirs(output_dir, exist_ok=True)

    random.seed(shuffle_seed)
    random.shuffle(lines)

    hp = OmegaConf.load(config_path)
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # `FeatureExtractor` pulls in Qwen3-ASR/forced aligner + RMVPE and the
    # Generator needs a checkpoint load, so both are built lazily: only when a
    # clip actually has to be extracted, and only as heavy as needed
    # (`need_qwen` below: a codes-only or transcript-only run skips the ASR
    # weights, and RMVPE is built on first pitch use). A `--skip_existing` run
    # that rebuilds nothing with the silence filter off therefore never touches
    # either.
    fe = None
    net_g = None

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
    n_skipped_phone = 0
    n_skipped_missing = 0
    n_reused = 0
    n_rebuilt = 0
    n_invalid = 0

    reused_keys = [key for key in FEATURE_KEYS if key not in selected]
    # Only the alignment needs Qwen; `phonemize_text` and the codec do not.
    need_qwen = PHONE_KEY in selected and phones_from == 'aligner'
    print('Tensors: ' + ', '.join(selected)
          + (f' (rebuild/validate) | reuse as-is: {", ".join(reused_keys)}'
             if reused_keys else ' (rebuild/validate)')
          + (f' | phones from {phones_from}' if PHONE_KEY in selected else '')
          + (' | skip usable' if skip_if_one_exists and not force else '')
          + (' | force rebuild' if force else ''))

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
                print(f"Skipped line {base_line} due to no sid resolution")
                n_skipped_speaker += 1
                continue
            sid = resolved
        else:
            sid = int(sid)

        paths = dict(zip(FEATURE_KEYS, feature_paths(output_dir, base_line)))

        # Decide per tensor what to do. Selected tensors are rebuilt unless
        # --skip_existing finds them usable (--force overrides that); the
        # others are reused as-is and must already be usable on disk.
        shapes = {}
        rebuild = []
        unusable = []
        for key in FEATURE_KEYS:
            shape, reason = tensor_shape(paths[key], key)
            ok = shape is not None and key_shape_ok(key, shape)
            if key not in selected:
                if ok:
                    shapes[key] = shape
                else:
                    unusable.append(key)
                continue
            if ok and skip_if_one_exists and not force:
                shapes[key] = shape
                continue
            if shape is not None and not ok:
                print(f'Rebuilding {base_line}: {reason}')
                n_invalid += 1
            rebuild.append(key)

        if unusable:
            print(f'Skipping {base_line}: {", ".join(unusable)} missing or '
                  'unusable on disk (rebuild with --only '
                  f'{",".join(unusable)})')
            n_skipped_missing += 1
            continue

        # Reused code tensors must still agree (footgun 7); a mismatch can only
        # be fixed by re-encoding, which `--only codes` does.
        if all(key in shapes for key in CODE_KEYS):
            agree, reason = code_lengths_agree(shapes)
            if not agree:
                print(f'Skipping {base_line}: {reason} '
                      '(rebuild with --only codes)')
                n_skipped_missing += 1
                continue

        # Audio is needed for the silence check, for the codes and for the
        # alignment. A transcript-only phone rebuild needs none of it.
        needs_audio = (silence_filter
                       or any(key in CODE_KEYS for key in rebuild)
                       or (PHONE_KEY in rebuild and phones_from == 'aligner'))
        if needs_audio and not os.path.exists(line):
            if rebuild:
                print(f'File not found: {line}')
                n_skipped_missing += 1
                continue
            # The tensors are usable but the source audio is gone, so the level
            # check cannot be re-run; keep the clip rather than dropping an
            # entry that is already encoded.
            print(f'Audio not found for {base_line}; '
                  'keeping existing tensors')
            n_reused += 1
            new_lines.append('|'.join(paths[key] for key in FEATURE_KEYS)
                             + f'|{sid}')
            continue

        wav_48k = None
        if needs_audio:
            # 48 kHz is the stage-1 feature rate, and the once-per-clip level
            # check is done on it too.
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

        if not rebuild:
            # Nothing to recompute: everything on disk is reused. The audio may
            # have been loaded above purely for the level check.
            n_reused += 1
            new_lines.append('|'.join(paths[key] for key in FEATURE_KEYS)
                             + f'|{sid}')
            continue
        try:
            if fe is None:
                fe = FeatureExtractor(hp=hp, load_qwen=need_qwen)

            if PHONE_KEY in rebuild:
                if phones_from == 'transcript':
                    # Cheapest path: no audio, no ASR, no GPU.
                    phone_ids = (fe.phonemize_text(transcr)
                                 if str(transcr).strip() else [])
                else:
                    wav_16k, _ = librosa.load(line, sr=16000)
                    items = fe.phonemize_items(
                        fe.align(wav_16k, transcription=transcr),
                        transcript=transcr)
                    phone_ids = list(itertools.chain.from_iterable(
                        [it['phone_ids'] for it in items]))
                if not phone_ids:
                    # ASR/forced alignment produced nothing (typically a very
                    # short interjection).  The codes would be fine, but with no
                    # phonemes the clip can only train the transformer as an
                    # *unconditional* sample -- it teaches the model to predict
                    # codes from nothing.  Drop it instead of writing an
                    # unusable tensor (and don't write the codes either, if they
                    # were part of this run), so a later --skip_existing run
                    # does not pick them up.
                    print(f'Skipping {base_line}: no phonemes')
                    n_skipped_phone += 1
                    continue
                # phone_ids [n_phone]
                torch.save(torch.tensor(phone_ids), paths[PHONE_KEY])

            if any(key in CODE_KEYS for key in rebuild):
                if net_g is None:
                    net_g = load_generator(checkpoint_path, hp, device)
                feats_ac = fe.extract_features_ac(wav_48k)

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
                # encode_codes returns (Q, B, T_ds); drop batch dim -> (Q, T_ds):
                # spec_codes [Q_spec, T_codes] (Q_spec = codec.codebook_num)
                # pitch_codes [Q_pitch, T_codes] (Q_pitch = pitch_codec.codebook_num)
                torch.save(spec_codes.squeeze(1).cpu(), paths['spec_codes'])
                torch.save(pitch_codes.squeeze(1).cpu(), paths['pitch_codes'])

        except Exception as e:
            print(f'Error processing {line}: {e}')
            continue

        n_rebuilt += 1
        new_lines.append('|'.join(paths[key] for key in FEATURE_KEYS) + f'|{sid}')

    if smap is not None and n_skipped_speaker:
        print(f'Skipped {n_skipped_speaker} file(s) with no speaker-map match')
    print(f'Kept {len(new_lines)} file(s) '
          f'({n_reused} reused, {n_rebuilt} updated); '
          f'omitted {n_skipped_silent} silent/near-silent clip(s)'
          + (f' below {min_rms_db:.1f} dBFS' if silence_filter else '')
          + (f', {n_skipped_phone} clip(s) with no phonemes'
             if n_skipped_phone else '')
          + (f', {n_skipped_missing} clip(s) with missing/unusable tensors'
             if n_skipped_missing else '')
          + (f'; rebuilt {n_invalid} unusable tensor(s)'
             if n_invalid else '')
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
    parser.add_argument('--checkpoint', type=str, default=None,
        help='path to 1_training (lightning or Generator) checkpoint; only '
             'needed when spec_codes/pitch_codes are (re)built, i.e. required '
             'unless --only excludes them')
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--only', type=parse_only, default=list(FEATURE_KEYS),
        help='tensors to (re)build: "all" (default), a comma-separated subset '
             'of phone_ids,spec_codes,pitch_codes, or the shorthands '
             '"phones" (= phone_ids) and "codes" (= spec_codes,pitch_codes, '
             'always rebuilt together). Tensors not listed are reused as-is '
             'from --output_dir. Examples: "--only phones" after a phonemizer '
             'or dictionary change (no audio/ASR/codec), "--only codes" after '
             'a new stage-1 checkpoint (no ASR/aligner weights).')
    parser.add_argument('--force', action='store_true',
        help='rebuild the --only tensors even if --skip_existing would reuse '
             'them (with --skip_existing alone nothing is ever overwritten)')
    parser.add_argument('--phones_from', type=str, default='aligner',
        choices=['aligner', 'transcript'],
        help='how phone_ids are produced: "aligner" (default) phonemizes the '
             'forced-aligned words, exactly like a full run; "transcript" '
             'phonemizes the filelist transcript directly, which needs no '
             'audio, ASR or GPU but keeps clips the aligner would have left '
             'empty and drops word boundaries. Both keep the punctuation the '
             'transcript has (the aligner strips it; see features.py)')
    parser.add_argument('--speaker_map', type=str, default=None,
        help='JSON map canonicalising speaker ids to the stage-1 space '
             '(see speaker_map.py)')
    parser.add_argument('--skip_existing', action='store_true',
        help='never overwrite a usable tensor: the --only tensors are reused '
             'if they already exist (add --force to redo them). Every tensor '
             'is shape-checked on its own, so missing/corrupt files are '
             'rebuilt. The silence check still runs (audio only, no '
             'extraction).')
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
        skip_if_one_exists=args.skip_existing,
        min_rms_db=args.min_rms_db,
        min_speech_frac=args.min_speech_frac,
        only=args.only,
        force=args.force,
        phones_from=args.phones_from,
    )
