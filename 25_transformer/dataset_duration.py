"""Audio-duration statistics for the filelists used by both stages.

The repo uses three line formats, and this script auto-detects the format of
*each* filelist from the first field of its first non-empty line:

``stage2`` — output of ``2_preprocess.py``::

    <base>.phone_ids|<base>.spec_codes|<base>.pitch_codes|sid

  Duration comes from the number of code frames (``spec_codes.shape[-1]``)
  times the code frame period.  The code rate is
  ``sampling_rate / (hop_length * down_factor)``, i.e. 25 Hz (40 ms) with the
  shipped configs, so no audio has to be decoded and nothing is loaded into
  memory (tensors are opened with ``mmap=True`` and only their shape is read).

``stage1`` — output of ``1_preprocess.py``::

    <base>.wave|<base>.spec|<base>.f0|sid

  Duration comes from the ``.wave`` sample count (falls back to the ``.spec``
  frame count if a ``.wave`` file is missing).

``audio`` — a raw source filelist such as ``filelists_2/*.txt``::

    <audio path>|sid|<text>          (or a bare ``<audio path>``)

  Duration is read from the container header with ``soundfile.info``, so the
  audio is never decoded.

Lines containing the substring ``longform`` are skipped by both preprocessing
scripts, so they are skipped (and counted) here too unless
``--include-longform`` is given.

Examples
--------
Total (and per-speaker) duration of the stage-2 split actually used for
training::

    python dataset_duration.py data/base2/train_stage2.txt \
        data/base2/val_stage2.txt --per-speaker

How much raw audio the source filelists hold, before the 5% val split::

    python dataset_duration.py filelists_2/25_stage2_base1_linux.txt --hist

Machine-readable output (also useful to compare runs)::

    python dataset_duration.py data/base/train.txt data/base/val.txt --json dur.json

``--sampling_rate`` / ``--hop_length`` / ``--down_factor`` only affect how
tensor shapes are converted to seconds; they must match the config the tensors
were produced with (``configs/base.yaml``, ``Generator(down_factor=...)``).
``--config`` reads the two ``data`` keys automatically.  For raw audio the
file header is authoritative.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

import speaker_map as speaker_map_lib
from utils import win_longpath

# Suffixes used by ``1_preprocess.py`` / ``2_preprocess.py`` to name the
# per-file tensors.  The *first* field of a line decides the format.
STAGE2_SUFFIXES = (".phone_ids", ".spec_codes", ".pitch_codes")
STAGE1_SUFFIXES = (".wave", ".spec", ".f0")


# ---------------------------------------------------------------------- #
# Frame-rate bookkeeping
# ---------------------------------------------------------------------- #
@dataclass
class Rates:
    """How tensor shapes convert to seconds."""

    sampling_rate: int = 48000
    hop_length: int = 480
    down_factor: int = 4

    @property
    def mel_hz(self) -> float:
        """Decoder/mel frame rate (``sampling_rate / hop_length``)."""
        return self.sampling_rate / self.hop_length

    @property
    def code_hz(self) -> float:
        """RVQ code rate (the Generator downsamples by ``down_factor``)."""
        return self.sampling_rate / (self.hop_length * self.down_factor)


# ---------------------------------------------------------------------- #
# Per-item measurement
# ---------------------------------------------------------------------- #
def _load_shape(path: str) -> Tuple[int, ...]:
    """Shape of a ``torch.save``d tensor without materialising its storage.

    ``mmap=True`` makes ``torch.load`` map the file instead of reading/copying
    the whole tensor, so measuring a 30k-item filelist stays I/O bound on the
    file header rather than the payload.  Falls back to a plain load for
    tensors saved in the legacy (non-zip) format.
    """
    try:
        t = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except (RuntimeError, ValueError):
        t = torch.load(path, map_location="cpu", weights_only=True)
    return tuple(t.shape)


def _audio_seconds(path: str) -> float:
    """Duration of an audio file from its header (no decoding)."""
    import soundfile as sf

    return float(sf.info(path).duration)


def detect_format(first_field: str) -> str:
    """Classify a filelist line from its first field."""
    if first_field.endswith(STAGE2_SUFFIXES):
        return "stage2"
    if first_field.endswith(STAGE1_SUFFIXES):
        return "stage1"
    return "audio"


#: Field index of each stage-2 tensor inside a ``phone|spec|pitch|sid`` line.
_STAGE2_FIELD = {"phone_ids": 0, "spec_codes": 1, "pitch_codes": 2}


def measure_item(
    parts: List[str],
    fmt: str,
    rates: Rates,
    code_stream: str = "spec_codes",
) -> Tuple[Optional[float], Optional[int]]:
    """Return ``(seconds, frames)`` for one filelist line.

    ``frames`` is the number of *decoder* frames for ``stage1`` items and the
    number of *code* frames for ``stage2`` items (``None`` for raw audio).
    Exceptions are left to the caller so a single unreadable file cannot abort
    a 30k-item report.

    Note that preprocessed filelists store *full* tensor paths in every field
    (``.../x.flac|.../x.flac.spec_codes|...`` is not the format — each field
    already carries the ``data/`` prefix and the tensor suffix).
    """
    if fmt == "stage2":
        # All three tensors exist per item; pick one stream and read only its
        # shape.  spec/pitch code lengths are asserted equal by
        # ``Generator.infer_from_codes``, so either is a valid clock.
        idx = _STAGE2_FIELD[code_stream]
        code_path = parts[idx] if len(parts) > idx else parts[0]
        t = _load_shape(win_longpath(code_path))
        frames = int(t[-1])
        return frames / rates.code_hz, frames

    if fmt == "stage1":
        # Field 0 is one of the ``.wave`` / ``.spec`` / ``.f0`` tensors.
        path = win_longpath(parts[0])
        if path.endswith(".wave"):
            try:
                samples = int(_load_shape(path)[-1])
                return samples / rates.sampling_rate, samples
            except FileNotFoundError:
                # Fall back to the mel frames (same clock, / hop_length).
                path = path[: -len(".wave")] + ".spec"
        frames = int(_load_shape(path)[-1])
        return frames / rates.mel_hz, frames

    return _audio_seconds(win_longpath(parts[0])), None


# ---------------------------------------------------------------------- #
# Filelist parsing / reporting
# ---------------------------------------------------------------------- #
@dataclass
class Item:
    path: str                 # first field, as written in the filelist
    sid: Optional[int]        # resolved (or raw) speaker id
    seconds: float
    frames: Optional[int]


@dataclass
class Report:
    filelist: str
    fmt: str = "?"
    lines: int = 0
    skipped_empty: int = 0
    skipped_longform: int = 0
    skipped_unmapped: int = 0
    skipped_missing: int = 0
    skipped_error: int = 0
    items: List[Item] = field(default_factory=list)
    errors: List[Tuple[str, str]] = field(default_factory=list)

    # -- aggregates ---------------------------------------------------- #
    @property
    def count(self) -> int:
        return len(self.items)

    @property
    def seconds(self) -> float:
        return sum(it.seconds for it in self.items)

    @property
    def frames(self) -> int:
        return sum(it.frames or 0 for it in self.items)

    @property
    def skipped(self) -> int:
        return (self.skipped_empty + self.skipped_longform
                + self.skipped_unmapped + self.skipped_missing
                + self.skipped_error)

    def durations(self) -> List[float]:
        return sorted(it.seconds for it in self.items)

    def by_speaker(self) -> Dict[Optional[int], List[Item]]:
        groups: Dict[Optional[int], List[Item]] = {}
        for it in self.items:
            groups.setdefault(it.sid, []).append(it)
        return groups

    def identities(self) -> List[str]:
        """Item identity used for duplicate / cross-filelist overlap checks.

        Preprocessed filelists append a tensor suffix to the shared base, so
        strip it and keep the *full* path: ``1_preprocess.py`` names outputs by
        basename only, so two different source folders can share a basename and
        a basename-keyed check would report phantom duplicates.
        """
        out = []
        for it in self.items:
            path = it.path
            for suffix in STAGE2_SUFFIXES + STAGE1_SUFFIXES:
                if path.endswith(suffix):
                    path = path[: -len(suffix)]
                    break
            out.append(os.path.normpath(path))
        return out


def parse_sid(parts: List[str], fmt: str) -> Optional[int]:
    """Speaker id for a raw line.

    Preprocessed formats (``stage1``/``stage2``) always end with ``sid``;
    raw source filelists are ``path|sid|text``.
    """
    candidates = [parts[-1]] if fmt != "audio" else parts[1:2]
    for cand in candidates:
        try:
            return int(cand.strip())
        except (ValueError, IndexError):
            continue
    return None


def read_filelist(
    filelist: str,
    rates: Rates,
    fmt: Optional[str] = None,
    speaker_map=None,
    include_longform: bool = False,
    code_stream: str = "spec_codes",
    workers: int = 8,
    progress: bool = True,
) -> Report:
    """Measure every item of ``filelist`` (in parallel)."""
    report = Report(filelist=filelist)

    lines: List[List[str]] = []
    with open(filelist, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            report.lines += 1
            if not line:
                report.skipped_empty += 1
                continue
            if not include_longform and "longform" in line:
                report.skipped_longform += 1
                continue
            lines.append(line.split("|"))

    if not lines:
        return report

    # The format is a property of the filelist, not of a line: detect it from
    # the first usable line unless the caller forced one.
    if fmt is None:
        fmt = detect_format(lines[0][0])
    report.fmt = fmt

    work: List[Tuple[List[str], Optional[int]]] = []
    for parts in lines:
        path = parts[0]
        sid = parse_sid(parts, fmt)
        if speaker_map is not None:
            # Same semantics as dataset_codes.py: path rules are only usable
            # when the audio path is still present (stage1/audio filelists).
            resolved = speaker_map_lib.resolve_sid(
                speaker_map,
                path=path if fmt in ("stage1", "audio") else None,
                raw_sid=sid,
            )
            if resolved is None:
                report.skipped_unmapped += 1
                continue
            sid = resolved
        work.append((parts, sid))

    def _run(entry: Tuple[List[str], Optional[int]]):
        parts, sid = entry
        try:
            seconds, frames = measure_item(parts, fmt, rates, code_stream)
        except (FileNotFoundError, OSError):
            return entry, None, None, "missing"
        except Exception as exc:  # noqa: BLE001 - report, don't abort
            return entry, None, None, f"{type(exc).__name__}: {exc}"
        return entry, seconds, frames, None

    iterator = work
    if progress:
        iterator = tqdm(work, desc=os.path.basename(filelist), unit="item",
                        leave=False)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for (parts, sid), seconds, frames, err in pool.map(_run, iterator):
            if err == "missing":
                report.skipped_missing += 1
                report.errors.append((parts[0], "missing"))
            elif err is not None:
                report.skipped_error += 1
                report.errors.append((parts[0], err))
            else:
                report.items.append(Item(path=parts[0], sid=sid,
                                         seconds=float(seconds),
                                         frames=frames))
    return report



# ---------------------------------------------------------------------- #
# Formatting
# ---------------------------------------------------------------------- #
def fmt_hms(seconds: float) -> str:
    """``3725.4`` -> ``01:02:05.400``."""
    ms = int(round(max(seconds, 0.0) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of an already sorted sequence."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def print_histogram(sorted_vals: Sequence[float], bins: int = 10,
                    width: int = 40) -> None:
    """ASCII histogram of item durations (helps pick ``max_frames``)."""
    if not sorted_vals:
        return
    lo, hi = sorted_vals[0], sorted_vals[-1]
    if hi <= lo:
        return
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in sorted_vals:
        counts[min(int((v - lo) / step), bins - 1)] += 1
    peak = max(counts) or 1
    print(f"  duration histogram ({bins} bins x {step:.2f} s):")
    for i, c in enumerate(counts):
        bar = "#" * int(round(width * c / peak))
        print(f"    {lo + i * step:7.2f}s | {c:6d} {bar}")


def print_report(report: Report, rates: Rates, per_speaker: bool = False,
                 hist: bool = False, max_errors: int = 3) -> None:
    fmt_desc = {
        "stage2": f"stage-2 codes (code frames @ {rates.code_hz:g} Hz)",
        "stage1": f"stage-1 features (wave samples @ {rates.sampling_rate} Hz)",
        "audio": "raw audio (header duration)",
    }.get(report.fmt, report.fmt)

    print(f"=== {report.filelist} ===")
    print(f"  format         {fmt_desc}")
    print(f"  lines          {report.lines}")
    print(f"  counted        {report.count}")
    if report.skipped:
        print(f"  skipped        {report.skipped}  "
              f"(empty {report.skipped_empty}, longform {report.skipped_longform}, "
              f"unmapped speaker {report.skipped_unmapped}, "
              f"missing {report.skipped_missing}, error {report.skipped_error})")
    if report.errors:
        print(f"  problems       {len(report.errors)} "
              f"(showing first {min(max_errors, len(report.errors))}):")
        for path, err in report.errors[:max_errors]:
            print(f"    - {path}: {err}")
    if not report.items:
        print()
        return

    vals = report.durations()
    total = report.seconds
    print(f"  total          {fmt_hms(total)}  =  {total / 3600:.3f} h  "
          f"=  {total:.1f} s")
    if report.fmt == "stage2":
        unit, frames = "code frames", report.frames
    elif report.fmt == "stage1":
        unit, frames = "wave samples", report.frames
    else:
        unit, frames = "frames", "n/a"
    print(f"  {unit:<14s} {frames}")
    print(f"  per item       mean {total / report.count:7.3f} s   "
          f"median {_percentile(vals, 0.5):7.3f} s   "
          f"min {vals[0]:6.3f} s   max {vals[-1]:6.3f} s")
    print(f"  percentiles    p05 {_percentile(vals, 0.05):6.3f}  "
          f"p25 {_percentile(vals, 0.25):6.3f}  "
          f"p50 {_percentile(vals, 0.50):6.3f}  "
          f"p75 {_percentile(vals, 0.75):6.3f}  "
          f"p95 {_percentile(vals, 0.95):6.3f}  (s)")

    groups = report.by_speaker()
    print(f"  speakers       {len(groups)}")
    dups = sum(c - 1 for c in collections.Counter(report.identities()).values())
    if dups:
        print(f"  duplicates     {dups} extra line(s) for an item already "
              f"counted (seen more than once per epoch)")

    if per_speaker:
        rows = [(sum(it.seconds for it in items), sid, len(items))
                for sid, items in groups.items()]
        rows.sort(reverse=True)
        print(f"    {'sid':>5s}  {'items':>7s}  {'hours':>8s}  {'share':>7s}  "
              f"{'mean s':>7s}")
        for secs, sid, n in rows:
            print(f"    {str(sid):>5s}  {n:>7d}  {secs / 3600:>8.3f}  "
                  f"{100 * secs / total:>6.1f}%  {secs / n:>7.3f}")
        if len(rows) > 1:
            top4 = sum(r[0] for r in rows[:4]) / total
            print(f"    (top-4 speakers = {100 * top4:.1f}% of the audio)")

    if hist:
        print_histogram(vals)

    print()



# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("filelists", nargs="+",
                        help="one or more filelists (stage-1, stage-2 or raw "
                             "audio format; auto-detected per file)")
    parser.add_argument("--format",
                        choices=("auto", "stage1", "stage2", "audio"),
                        default="auto",
                        help="force a line format for every filelist "
                             "(default: auto-detect per file)")
    parser.add_argument("--config", default=None,
                        help="optional config yaml to read data.sampling_rate / "
                             "data.hop_length from (e.g. configs/transformer.yaml)")
    parser.add_argument("--sampling_rate", type=int, default=48000)
    parser.add_argument("--hop_length", type=int, default=480)
    parser.add_argument("--down_factor", type=int, default=4,
                        help="Generator down_factor: decoder frames per RVQ "
                             "code frame (4 -> 25 Hz code rate)")
    parser.add_argument("--code_stream",
                        choices=("spec_codes", "pitch_codes"),
                        default="spec_codes",
                        help="which stage-2 tensor to count frames from "
                             "(lengths are equal; pitch_codes is 4x smaller)")
    parser.add_argument("--per-speaker", action="store_true",
                        help="print a per-speaker duration table")
    parser.add_argument("--speaker_map", default=None,
                        help="optional speaker_map.json to canonicalise ids "
                             "(same sid_map/unmapped semantics as training)")
    parser.add_argument("--hist", action="store_true",
                        help="print a duration histogram per filelist")
    parser.add_argument("--include-longform", action="store_true",
                        help="include lines containing 'longform' "
                             "(skipped by both preprocessing scripts)")
    parser.add_argument("--workers", type=int, default=8,
                        help="I/O threads used to load file headers/tensors")
    parser.add_argument("--json", default=None,
                        help="write the full report (per item included) as JSON")
    return parser.parse_args(argv)


def _rates_from_args(args: argparse.Namespace) -> Rates:
    rates = Rates(sampling_rate=args.sampling_rate,
                  hop_length=args.hop_length,
                  down_factor=args.down_factor)
    if args.config:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(args.config)
        data = cfg.get("data", None)
        if data is not None:
            rates.sampling_rate = int(data.get("sampling_rate",
                                               rates.sampling_rate))
            rates.hop_length = int(data.get("hop_length", rates.hop_length))
    if min(rates.sampling_rate, rates.hop_length, rates.down_factor) <= 0:
        raise SystemExit("sampling_rate, hop_length and down_factor must be > 0")
    return rates



def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    rates = _rates_from_args(args)

    speaker_map = None
    if args.speaker_map:
        speaker_map = speaker_map_lib.load_speaker_map(args.speaker_map)

    fmt = None if args.format == "auto" else args.format
    reports: List[Report] = []
    for filelist in args.filelists:
        report = read_filelist(
            filelist, rates, fmt=fmt, speaker_map=speaker_map,
            include_longform=args.include_longform,
            code_stream=args.code_stream, workers=args.workers,
            progress=len(args.filelists) > 1,
        )
        print_report(report, rates, per_speaker=args.per_speaker,
                     hist=args.hist)
        reports.append(report)

    total = sum(r.seconds for r in reports)
    print("=== grand total ===")
    print(f"  filelists      {len(reports)}")
    print(f"  items          {sum(r.count for r in reports)}")
    print(f"  duration       {fmt_hms(total)}  =  {total / 3600:.3f} h")

    # Duplicate lines inflate the epoch, and a shared item between train and
    # val is a leak.  Both are easy to introduce: 1_preprocess names outputs
    # by *basename*, so same-named files from different source folders collapse
    # onto one tensor.
    within = [sum(c - 1 for c in collections.Counter(r.identities()).values())
              for r in reports]
    all_ids: Dict[str, int] = {}
    for r in reports:
        for ident in set(r.identities()):
            all_ids[ident] = all_ids.get(ident, 0) + 1
    shared = sum(1 for c in all_ids.values() if c > 1)
    print(f"  unique items   {len(all_ids)}  "
          f"(duplicate lines within a filelist: {sum(within)}, "
          f"items shared between filelists: {shared})")
    if sum(within):
        print("  warning        duplicate lines point at the same tensors, so "
              "they are seen twice per epoch")
    if shared:
        print("  warning        items shared between filelists leak audio "
              "across splits (val is not truly held out)")
    print()


    if args.json:
        payload = {
            "rates": {
                "sampling_rate": rates.sampling_rate,
                "hop_length": rates.hop_length,
                "down_factor": rates.down_factor,
                "mel_hz": rates.mel_hz,
                "code_hz": rates.code_hz,
            },
            "grand_total_seconds": total,
            "filelists": [
                {
                    "filelist": r.filelist,
                    "format": r.fmt,
                    "lines": r.lines,
                    "counted": r.count,
                    "skipped": {
                        "empty": r.skipped_empty,
                        "longform": r.skipped_longform,
                        "unmapped_speaker": r.skipped_unmapped,
                        "missing": r.skipped_missing,
                        "error": r.skipped_error,
                    },
                    "total_seconds": r.seconds,
                    "total_hours": r.seconds / 3600,
                    "frames": r.frames,
                    "speakers": len(r.by_speaker()),
                    "per_speaker_seconds": {
                        str(sid): sum(it.seconds for it in items)
                        for sid, items in r.by_speaker().items()
                    },
                    "items": [
                        {"path": it.path, "sid": it.sid,
                         "seconds": it.seconds, "frames": it.frames}
                        for it in r.items
                    ],
                    "problems": [{"path": p, "error": e} for p, e in r.errors],
                }
                for r in reports
            ],
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote JSON report to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

