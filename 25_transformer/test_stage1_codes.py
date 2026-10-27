"""Ad-hoc stage-1 codec round-trip test for arbitrary audio files.

For each audio file given on the command line this runs exactly the code path
used by ``2_preprocess.py`` (encode) and stage-2 validation (decode):

    wav --(FeatureExtractor)--> spec (T, 769) + f0 (T,)
        --(Generator.encode_codes)--> spec_codes (Q_spec, T/down_factor) +
                                      pitch_codes (Q_pitch, T/down_factor)
        --(Generator.infer_from_codes)--> waveform + f0 decoded from the codes

and writes ``<output_dir>/<name>_recon.wav`` next to a per-file metric
summary (log-mel L1 in dB, f0 error in cents, voicing agreement, per-codebook
code usage). ``--roundtrip`` also re-encodes the synthesized audio and reports
how many codes survive the code -> audio -> code loop, which is the harshest
single-number check of codec fidelity. ``--save_tensors`` dumps the codes and
both f0 contours for offline inspection.

Usage:
    python test_stage1_codes.py test.wav \
        --checkpoint checkpoints/base_v0.4.6/last.ckpt --sid 0

Notes:
    * The Generator is built from ``--config`` (default ``configs/base.yaml``);
      the architecture is never inferred from the checkpoint, and
      ``load_state_dict_mismatch`` only reports keys whose shapes disagree, so
      a config that does not match the checkpoint can silently load partially
      (see AGENTS.md). Use the config the checkpoint was trained with.
    * ``FeatureExtractor`` is created with ``load_qwen=False``: only the
      spectrogram + RMVPE pitch path is needed here, so the heavy ASR /
      forced-aligner weights are skipped.
    * Audio goes through the model in one piece, so memory scales with file
      length. Use ``--max_seconds`` to truncate long inputs.
    * Both codecs are speaker-conditioned, so ``--sid`` must be a stage-1 id
      that exists in the checkpoint's speaker table: the "right" voice gives
      the lowest metrics, an unknown id still synthesises but shifts f0 /
      timbre towards whatever that embedding learned (see AGENTS.md §6.1).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import librosa
import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf

from commons import load_state_dict_mismatch, load_submodule_prefix
from features import FeatureExtractor
from modeling.generator import Generator
from utils import count_parameters, win_longpath


# ---------------------------------------------------------------------- #
# Checkpoint / audio loading
# ---------------------------------------------------------------------- #
def load_generator(checkpoint_path: str, config, device: str) -> Generator:
    """Build a ``Generator`` from ``config`` and load a stage-1 checkpoint.

    Mirrors ``load_generator`` in ``2_preprocess.py`` / ``2_training.py``: the
    checkpoint may be a Lightning file (state nested under ``state_dict``) and
    its weights may or may not carry a ``net_g.`` prefix.
    """
    path = win_longpath(checkpoint_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    net_g = Generator(hp=config)
    state = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    if any(k.startswith('net_g.') for k in state.keys()):
        load_submodule_prefix(net_g, 'net_g.', state)
    else:
        load_state_dict_mismatch(net_g, state)
    net_g.to(device)
    net_g.eval()
    return net_g


def load_audio(path: str, sampling_rate: int,
               max_seconds: float | None = None) -> np.ndarray:
    """Load ``path`` as mono float32 at ``sampling_rate``, like 1_preprocess."""
    resolved = win_longpath(path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f'Audio file not found: {path}')
    wav, _ = librosa.load(resolved, sr=sampling_rate, mono=True)
    if max_seconds is not None:
        wav = wav[:max(1, int(max_seconds * sampling_rate))]
    if wav.size == 0:
        raise ValueError('Audio file is empty after loading')
    return wav.astype(np.float32, copy=False)


# ---------------------------------------------------------------------- #
# Metrics
# ---------------------------------------------------------------------- #
def log_mel_db(audio: np.ndarray, sampling_rate: int, n_fft: int,
               win_length: int, hop_length: int, n_mels: int,
               fmin: float, fmax: float) -> np.ndarray:
    """Absolute log-mel spectrogram in dB (``(n_mels, frames)``)."""
    mel = librosa.feature.melspectrogram(
        y=np.asarray(audio, dtype=np.float32),
        sr=sampling_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
        center=True,
    )
    # ref=1.0 keeps absolute levels so gain errors are not normalised away.
    return librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None)


def mel_l1_db(ref: np.ndarray, gen: np.ndarray, *,
              sampling_rate: int, n_fft: int, win_length: int,
              hop_length: int, n_mels: int, fmin: float,
              fmax: float) -> float:
    """Mean |dB| difference between the log-mels of two waveforms."""
    n = min(ref.size, gen.size)
    if n == 0:
        return float('nan')
    mel_ref = log_mel_db(ref[:n], sampling_rate, n_fft, win_length,
                         hop_length, n_mels, fmin, fmax)
    mel_gen = log_mel_db(gen[:n], sampling_rate, n_fft, win_length,
                         hop_length, n_mels, fmin, fmax)
    frames = min(mel_ref.shape[1], mel_gen.shape[1])
    if frames == 0:
        return float('nan')
    return float(np.abs(mel_ref[:, :frames] - mel_gen[:, :frames]).mean())


def f0_metrics(f0_ref: np.ndarray, f0_gen: np.ndarray) -> dict:
    """Compare two f0 contours in Hz (0 == unvoiced) at the same frame rate.

    All error terms are restricted to frames where *both* contours are voiced,
    so the V/UV decision (which the pitch codec's head makes at synthesis) is
    reported separately as agreement/recall/precision instead of polluting the
    pitch error with zeros.
    """
    n = min(f0_ref.size, f0_gen.size)
    ref = np.asarray(f0_ref[:n], dtype=np.float64)
    gen = np.asarray(f0_gen[:n], dtype=np.float64)
    out = {
        'frames': int(n),
        'f0_voiced_frames_ref': int((ref > 0).sum()),
        'f0_voiced_frames_gen': int((gen > 0).sum()),
        'voicing_agreement': float('nan'),
        'voiced_recall': float('nan'),
        'voiced_precision': float('nan'),
        'f0_mae_cents': float('nan'),
        'f0_rmse_hz': float('nan'),
    }
    if n == 0:
        return out

    ref_v = ref > 0
    gen_v = gen > 0
    out['voicing_agreement'] = float((ref_v == gen_v).mean())
    if ref_v.any():
        out['voiced_recall'] = float((gen_v & ref_v).sum() / ref_v.sum())
    if gen_v.any():
        out['voiced_precision'] = float((gen_v & ref_v).sum() / gen_v.sum())

    both = ref_v & gen_v
    if both.any():
        cents = 1200.0 * np.log2(gen[both] / ref[both])
        out['f0_mae_cents'] = float(np.abs(cents).mean())
        out['f0_rmse_hz'] = float(np.sqrt(((gen[both] - ref[both]) ** 2).mean()))
    return out


def code_stats(codes: np.ndarray, vocab_size: int) -> dict:
    """Per-codebook unique-id count and empirical entropy (in bits).

    ``codes`` is ``(Q, T)``. For a single utterance the entropy is only a
    sanity check ("did this book collapse to one id?"), not a usage metric.
    """
    uniques, entropies = [], []
    for book in codes:
        _, counts = np.unique(book, return_counts=True)
        p = counts / counts.sum()
        uniques.append(int(counts.size))
        entropies.append(float(-(p * np.log2(p)).sum()))
    return {
        'vocab_size': int(vocab_size),
        'unique_per_book': uniques,
        'entropy_bits_per_book': [round(e, 3) for e in entropies],
    }


def code_match_fraction(a: np.ndarray, b: np.ndarray) -> list[float]:
    """Per-codebook fraction of equal ids between ``(Q, T)`` code tensors."""
    frames = min(a.shape[1], b.shape[1])
    if frames == 0:
        return [float('nan')] * int(a.shape[0])
    return [float((a[q, :frames] == b[q, :frames]).mean())
            for q in range(min(a.shape[0], b.shape[0]))]


# ---------------------------------------------------------------------- #
# Per-file pipeline
# ---------------------------------------------------------------------- #
def encode_codes(net_g: Generator, feats: dict, sid: int,
                 device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(spec, f0) -> (spec_codes, pitch_codes), each ``(Q, 1, T_codes)``."""
    spec = feats['spec']
    f0 = feats['f0'].float()
    spec_b = spec.transpose(0, 1).unsqueeze(0).to(device)  # (1, C, T)
    spec_l = torch.tensor([spec.shape[0]], device=device)
    f0_b = f0.unsqueeze(0).to(device)  # (1, T)
    sid_b = torch.tensor([int(sid)], device=device, dtype=torch.long)
    with torch.no_grad():
        spec_codes, pitch_codes = net_g.encode_codes(
            spec_b, spec_l, f0_b, sid_b)
    return spec_codes, pitch_codes


def process_file(path: str, fe: FeatureExtractor, net_g: Generator, hp,
                 args, device: str) -> dict:
    """Reconstruct one audio file through the codec and return its metrics."""
    sr = int(hp.data.sampling_rate)
    wav = load_audio(path, sr, args.max_seconds)
    duration = wav.size / sr
    if args.max_seconds is None and duration > 120.0:
        print(f'  warning: {duration:.1f}s of audio is processed in one piece; '
              f'consider --max_seconds to bound memory')

    feats = fe.extract_features_ac(wav)
    spec = feats['spec']  # (T, C)
    f0_ref = feats['f0'].float().numpy()  # (T,)
    n_frames = int(spec.shape[0])
    if n_frames < 8:
        raise ValueError(
            f'Only {n_frames} spectrogram frames; need at least 8 for the '
            '4x codec downsampling -- use a longer clip')

    spec_codes, pitch_codes = encode_codes(net_g, feats, args.sid, device)
    sid_b = torch.tensor([int(args.sid)], device=device, dtype=torch.long)
    # out_len trims the right-pad the encoder added to reach a multiple of
    # down_factor, so the waveform is exactly spec_frames * hop_length long.
    with torch.no_grad():
        audio, f0_gen = net_g.infer_from_codes(
            spec_codes, pitch_codes, sid_b,
            out_len=n_frames, return_pitch=True)
    audio = audio[0, 0].float().cpu().numpy()
    f0_gen = f0_gen[0].float().cpu().numpy()

    spec_codes_np = spec_codes[:, 0].cpu().numpy()
    pitch_codes_np = pitch_codes[:, 0].cpu().numpy()
    n_codes = int(spec_codes_np.shape[1])

    result = {
        'file': os.path.abspath(path),
        'seconds_in': round(duration, 3),
        'seconds_out': round(audio.size / sr, 3),
        'spec_frames': n_frames,
        'code_frames': n_codes,
        'spec_codes': code_stats(spec_codes_np, int(hp.codec.codebook_size)),
        'pitch_codes': code_stats(pitch_codes_np,
                                  int(hp.pitch_codec.codebook_size)),
        'mel_l1_db': mel_l1_db(
            wav, audio,
            sampling_rate=sr,
            n_fft=int(hp.data.filter_length),
            win_length=int(hp.data.win_length),
            hop_length=int(hp.data.hop_length),
            n_mels=int(hp.data.mel_channels),
            fmin=float(hp.data.mel_fmin),
            fmax=float(hp.data.mel_fmax)),
    }
    result.update(f0_metrics(f0_ref, f0_gen))

    # Optional code -> audio -> code loop: re-encode the synthesis and compare
    # the ids. Fidelity limits (VQ, f0 re-tracking on synthetic audio) mean
    # this is never 1.0; it exposes collapsed/dead codebooks.
    if args.roundtrip:
        recon_feats = fe.extract_features_ac(audio)
        rt_spec_codes, rt_pitch_codes = encode_codes(
            net_g, recon_feats, args.sid, device)
        result['roundtrip_spec_code_match'] = code_match_fraction(
            spec_codes_np, rt_spec_codes[:, 0].cpu().numpy())
        result['roundtrip_pitch_code_match'] = code_match_fraction(
            pitch_codes_np, rt_pitch_codes[:, 0].cpu().numpy())

    # --- write artefacts ---
    stem = os.path.splitext(os.path.basename(path))[0]
    recon_path = os.path.join(args.output_dir, f'{stem}_recon.wav')
    sf.write(recon_path, np.clip(audio, -1.0, 1.0), sr)
    result['recon_path'] = os.path.abspath(recon_path)

    if args.save_tensors:
        tensor_path = os.path.join(args.output_dir, f'{stem}_stage1.pt')
        torch.save({
            'spec_codes': spec_codes[:, 0].cpu(),       # (Q_spec, T_codes)
            'pitch_codes': pitch_codes[:, 0].cpu(),     # (Q_pitch, T_codes)
            'f0_ref': torch.from_numpy(f0_ref),         # (T_frames,) Hz
            'f0_from_codes': torch.from_numpy(f0_gen),  # (T_frames_out,) Hz
            'sid': int(args.sid),
        }, tensor_path)
        result['tensor_path'] = os.path.abspath(tensor_path)

    return result


# ---------------------------------------------------------------------- #
# Reporting
# ---------------------------------------------------------------------- #
def _fmt(value: float, spec: str) -> str:
    """Format a metric, printing 'n/a' for None/NaN/inf."""
    if value is None or not np.isfinite(value):
        return 'n/a'
    return format(value, spec)


def code_line(name: str, stats: dict) -> str:
    uniques = ' '.join(str(u) for u in stats['unique_per_book'])
    return (f'  {name} codes: {len(stats["unique_per_book"])} books, '
            f'unique ids per book [{uniques}] (vocab {stats["vocab_size"]})')


def print_result(index: int, total: int, result: dict) -> None:
    print(f'[{index}/{total}] {result["file"]}')
    print(f'  length: {result["seconds_in"]:.2f}s in -> '
          f'{result["seconds_out"]:.2f}s out | '
          f'{result["spec_frames"]} spec frames -> '
          f'{result["code_frames"]} code frames')
    print(code_line('spec', result['spec_codes']))
    print(code_line('pitch', result['pitch_codes']))
    print(f'  mel L1: {_fmt(result["mel_l1_db"], ".2f")} dB | '
          f'f0 MAE: {_fmt(result["f0_mae_cents"], ".1f")} cents | '
          f'f0 RMSE: {_fmt(result["f0_rmse_hz"], ".1f")} Hz')
    print(f'  voicing: acc {_fmt(result["voicing_agreement"], ".3f")} | '
          f'recall {_fmt(result["voiced_recall"], ".3f")} | '
          f'precision {_fmt(result["voiced_precision"], ".3f")} '
          f'({result["f0_voiced_frames_ref"]} voiced ref frames, '
          f'{result["f0_voiced_frames_gen"]} voiced decoded frames)')
    if 'roundtrip_spec_code_match' in result:
        spec_match = ' '.join(
            _fmt(m, '.2f') for m in result['roundtrip_spec_code_match'])
        pitch_match = ' '.join(
            _fmt(m, '.2f') for m in result['roundtrip_pitch_code_match'])
        print(f'  code round-trip match (code -> audio -> code): '
              f'spec [{spec_match}] | pitch [{pitch_match}]')
    print(f'  wrote {result["recon_path"]}')
    if 'tensor_path' in result:
        print(f'  wrote {result["tensor_path"]}')


def print_summary(results: list[dict],
                  failures: list[tuple[str, str]]) -> None:
    print('=== Summary ===')
    if results:
        mel = np.array([r['mel_l1_db'] for r in results], dtype=np.float64)
        cents = np.array([r['f0_mae_cents'] for r in results],
                         dtype=np.float64)
        voicing = np.array([r['voicing_agreement'] for r in results],
                           dtype=np.float64)
        print(f'  files: {len(results)} | '
              f'mean mel L1 {_fmt(float(np.nanmean(mel)), ".2f")} dB | '
              f'mean f0 MAE {_fmt(float(np.nanmean(cents)), ".1f")} cents | '
              f'mean voicing acc {_fmt(float(np.nanmean(voicing)), ".3f")}')
    if failures:
        print(f'  failed: {len(failures)}')
        for path, err in failures:
            print(f'    {path}: {err}')
    else:
        print('  failed: 0')


# ---------------------------------------------------------------------- #
# Entry point
# ---------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Stage-1 checkpoint codec round-trip test: encode audio '
                    'files to RVQ codes and decode them back to audio.')
    parser.add_argument('audio_files', nargs='+',
                        help='audio file(s) to reconstruct')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='stage-1 checkpoint (1_training Lightning .ckpt '
                             'or a bare Generator state dict)')
    parser.add_argument('--config', type=str, default='configs/base.yaml',
                        help='stage-1 config matching the checkpoint')
    parser.add_argument('--sid', type=int, default=0,
                        help='stage-1 speaker id used for both encoding and '
                             'decoding (default: 0)')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join('test_out', 'stage1_codes'),
                        help='directory for reconstructed wavs / metrics')
    parser.add_argument('--device', type=str, default=None,
                        help="torch device, e.g. 'cuda:0' or 'cpu' "
                             '(default: cuda:0 if available)')
    parser.add_argument('--max_seconds', type=float, default=None,
                        help='truncate each input to this many seconds')
    parser.add_argument('--roundtrip', action='store_true',
                        help='re-encode the synthesized audio and report '
                             'code -> audio -> code match fractions')
    parser.add_argument('--save_tensors', action='store_true',
                        help='also save codes and both f0 contours as .pt')
    args = parser.parse_args(argv)

    if args.max_seconds is not None and args.max_seconds <= 0:
        parser.error('--max_seconds must be > 0')

    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    hp = OmegaConf.load(args.config)
    max_spk = int(hp.vits.max_spk_count)
    if not 0 <= args.sid < max_spk:
        parser.error(f'--sid must be in [0, {max_spk}), got {args.sid}')

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Config:     {args.config}')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Device:     {device} | sid: {args.sid}')
    net_g = load_generator(args.checkpoint, hp, device)
    print(f'Generator:  {count_parameters(net_g)}')
    # Only spec + f0 are needed, so skip the (large) ASR / aligner weights.
    fe = FeatureExtractor(hp=hp, device=device, load_qwen=False)

    results, failures = [], []
    for i, path in enumerate(args.audio_files, start=1):
        try:
            result = process_file(path, fe, net_g, hp, args, device)
        except Exception as e:  # keep going: one bad file shouldn't abort a run
            print(f'[{i}/{len(args.audio_files)}] {path}')
            print(f'  ERROR: {type(e).__name__}: {e}')
            if not isinstance(e, FileNotFoundError):
                traceback.print_exc()
            failures.append((path, f'{type(e).__name__}: {e}'))
            continue
        results.append(result)
        print_result(i, len(args.audio_files), result)

    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({
            'checkpoint': os.path.abspath(args.checkpoint),
            'config': os.path.abspath(args.config),
            'sid': args.sid,
            'roundtrip': bool(args.roundtrip),
            'files': results,
        }, f, indent=2)
    print(f'Metrics: {os.path.abspath(metrics_path)}')

    print_summary(results, failures)
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
