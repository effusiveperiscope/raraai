"""Build stage-2 *input* filelists from external corpora.

``2_preprocess.py`` consumes ``audio_path|sid|transcription`` lines (exactly
three fields, no ``|`` inside the text).  This script turns the on-disk
layouts of the extra corpora into that format, next to a ``*_speakers.json``
sidecar recording which raw sid belongs to which speaker::

    python 2_gather_corpora.py --corpus all

    filelists_2/vctk_linux.txt           path|sid|text   (sid = 0..N-1)
    filelists_2/vctk_speakers.json
    filelists_2/expresso_read_linux.txt  path|sid|text   (sid = 0..3)
    filelists_2/expresso_read_speakers.json
    filelists_2/mlp_linux_speakers.json  sidecar only -- ``mlp_linux.txt``
                                         already exists and is not rewritten

**Speaker ids are local to each corpus** (0-based, sorted by speaker name).
That is deliberate: they are *raw* ids in a namespace of their own, and
``2_combine_corpora.py`` lifts them into the global stage-2 space on top of the
base corpus (option B: every new speaker gets a fresh, unique id in the unused
range of ``transformer.n_speakers``).  Nothing downstream needs the raw ids to
mean anything.

Supported corpora
-----------------
``mlp``
    Already shipped as ``filelists_2/mlp_linux.txt`` -- so there is nothing to
    gather.  It is validated (3 fields per line, audio present) and its speaker
    names are recovered from the sliced-dialogue filenames
    (``00_08_51_Spike_Neutral_...`` -> ``Spike``) into
    ``mlp_linux_speakers.json``, purely so ``2_combine_corpora.py`` can label
    the new ids it assigns.  The filelist itself is never rewritten.

``vctk``
    VCTK-Corpus-0.92: ``wav48_silence_trimmed/<spk>/<spk>_<utt>_mic{1,2}.flac``
    + ``txt/<spk>/<spk>_<utt>.txt``.  Only *one* microphone is used (``mic1``
    by default): mic2 is the same utterance through a second mic, so keeping
    both would duplicate every transcript (doubling the corpus weight) while
    teaching the model nothing new.  Speakers with audio but no transcript
    directory (VCTK 0.92 ships mic audio for ``p315`` but no ``txt/p315``) are
    dropped.

``expresso``
    Read speech only: ``audio_48khz/read/<spk>/<style>/base/<name>.wav`` +
    ``read_transcriptions.txt`` (``name<TAB>text``).  ``longform`` is skipped
    (the ``longform/`` corpus directories *and* any ``_longform`` filename):
    those are minute-long recordings whose alignment is precisely what a
    segment-level model cannot use.  ``singing`` is skipped by default as well
    (sung, out of domain for a TTS voice); pass
    ``--expresso_keep_style singing`` to keep it.

    The ``conversational`` half of Expresso is **not supported**: it is stereo
    (one speaker per channel) and the release ships no transcriptions for it,
    only per-channel VAD segments.  It would need channel separation +
    diarisation + ASR first, and ``2_preprocess.py`` aligns against the
    transcript it is handed.

Next step, per corpus (VCTK shown)::

    CKPT=checkpoints_stage1/base_v0.7/last.ckpt
    python 2_preprocess.py --filelist filelists_2/vctk_linux.txt \\
        --output_dir data/vctk_stage2 --checkpoint $CKPT

That is the normal full run (``--only all``: codes + phones).  Splitting it is
possible -- the codes need the GPU, the phones are cheap -- but the **code
tensors must come first**: a ``--only phones`` run reuses the code tensors
already on disk and skips every clip that does not have them yet::

    python 2_preprocess.py ... --only codes  --checkpoint $CKPT
    python 2_preprocess.py ... --only phones --phones_from transcript

then ``python 2_combine_corpora.py --extra data/vctk_stage2 ...`` to fold it
into the base (Genshin) corpus.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple


def clean_text(text: str) -> str:
    """Make a transcript safe for the ``path|sid|text`` line format.

    ``2_preprocess.py`` splits the line on ``|`` and drops anything whose field
    count is not exactly 3, so a pipe inside a transcript would silently lose
    the clip.  Newlines/tabs are collapsed for the same reason.
    """
    return " ".join(str(text).split()).replace("|", "/").strip()


# ---------------------------------------------------------------------- #
# VCTK-Corpus-0.92
# ---------------------------------------------------------------------- #
def parse_vctk_speaker_info(path: str) -> Dict[str, dict]:
    """``speaker-info.txt`` -> ``{speaker: {...}}`` (best effort).

    The file is whitespace-aligned with a variable column count (some rows omit
    REGION/COMMENTS), so it is split on runs of whitespace and indexed
    defensively.  ``GENDER`` is ``F``/``M`` in VCTK 0.92.
    """
    info: Dict[str, dict] = {}
    if not os.path.exists(path):
        return info
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            toks = line.split()
            if not toks or not toks[0].startswith("p") or toks[0] == "ID":
                continue
            gender = toks[2].upper() if len(toks) > 2 else ""
            info[toks[0]] = {
                "gender": {"F": "female", "M": "male"}.get(gender),
                "age": toks[1] if len(toks) > 1 else None,
                "accent": toks[3] if len(toks) > 3 else None,
                "region": " ".join(toks[4:]) if len(toks) > 4 else None,
            }
    return info



def gather_vctk(root: str, mic: str = "mic1") -> Tuple[List[str], dict, dict]:
    """Rows for VCTK.  Returns ``(rows, speakers, stats)``."""
    wav_root = os.path.join(root, "wav48_silence_trimmed")
    txt_root = os.path.join(root, "txt")
    if not os.path.isdir(wav_root) or not os.path.isdir(txt_root):
        raise SystemExit(f"not a VCTK-Corpus-0.92 tree: {root}")
    info = parse_vctk_speaker_info(os.path.join(root, "speaker-info.txt"))

    audio_speakers = {d for d in os.listdir(wav_root)
                      if d.startswith("p")
                      and os.path.isdir(os.path.join(wav_root, d))}
    txt_speakers = {d for d in os.listdir(txt_root)
                    if d.startswith("p")
                    and os.path.isdir(os.path.join(txt_root, d))}
    stats = {"no_audio": 0, "empty_text": 0,
             "audio_without_transcripts": sorted(audio_speakers - txt_speakers),
             "transcripts_without_audio": sorted(txt_speakers - audio_speakers),
             # e.g. ``txt/s5``: the shared "s5" sentence set has no speaker of
             # its own, so it is not a corpus of voices and is skipped.
             "ignored_dirs": sorted(
                 d for d in os.listdir(txt_root)
                 if not d.startswith("p")
                 and os.path.isdir(os.path.join(txt_root, d)))}

    rows: List[str] = []
    meta: Dict[str, dict] = {}
    for spk in sorted(txt_speakers & audio_speakers):
        sid = len(meta)  # this speaker's raw id; only committed if it has rows
        spk_rows: List[str] = []
        for txt_path in sorted(glob.glob(os.path.join(txt_root, spk, "*.txt"))):
            stem = os.path.basename(txt_path)[: -len(".txt")]
            wav = os.path.join(wav_root, spk, f"{stem}_{mic}.flac")
            if not os.path.exists(wav):
                stats["no_audio"] += 1
                continue
            with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                text = clean_text(f.read())
            if not text:
                stats["empty_text"] += 1
                continue
            spk_rows.append(f"{wav}|{sid}|{text}")
        if not spk_rows:
            continue
        rows.extend(spk_rows)
        meta[str(sid)] = {"speaker": spk, "lines": len(spk_rows),
                          **info.get(spk, {"gender": None})}
    return rows, meta, stats


# ---------------------------------------------------------------------- #
# Expresso (read speech)
# ---------------------------------------------------------------------- #
def parse_expresso_transcripts(path: str) -> Dict[str, str]:
    """``read_transcriptions.txt`` (``key<TAB>text``) -> ``{key: text}``."""
    table: Dict[str, str] = {}
    if not os.path.exists(path):
        return table
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            key, text = line.split("\t", 1)
            table[key.strip()] = clean_text(text)
    return table


# Genders are NOT in the Expresso release (it ships no speaker metadata file);
# these come from the Expresso paper (4 speakers: 2 female, 2 male).  They are
# only used to suggest a matching stage-1 proxy voice for decoding -- check by
# ear and edit the emitted *_speakers.json if a label is wrong.
EXPRESSO_GENDERS = {"ex01": "male", "ex02": "female",
                    "ex03": "male", "ex04": "female"}


def gather_expresso(root: str, skip_styles=("longform", "singing")
                    ) -> Tuple[List[str], dict, dict]:
    """Rows for Expresso *read* speech.  Returns ``(rows, speakers, stats)``."""
    read_root = os.path.join(root, "audio_48khz", "read")
    if not os.path.isdir(read_root):
        raise SystemExit(
            f"not an Expresso tree (missing audio_48khz/read): {root}")
    transcripts = parse_expresso_transcripts(
        os.path.join(root, "read_transcriptions.txt"))
    skip = set(skip_styles)
    used = set()
    rows: List[str] = []
    meta: Dict[str, dict] = {}
    stats = {"no_transcript": 0, "longform_skipped": 0, "style_skipped": {},
             "unused_transcripts": 0}

    for spk in sorted(d for d in os.listdir(read_root)
                      if os.path.isdir(os.path.join(read_root, d))):
        sid = len(meta)  # only committed if this speaker yields any clip
        spk_rows: List[str] = []
        for style in sorted(os.listdir(os.path.join(read_root, spk))):
            style_dir = os.path.join(read_root, spk, style)
            if not os.path.isdir(style_dir):
                continue
            if style in skip:
                n = len(glob.glob(os.path.join(style_dir, "**", "*.wav"),
                                  recursive=True))
                stats["style_skipped"][style] = (
                    stats["style_skipped"].get(style, 0) + n)
                continue
            # Only the short-sentence corpus: the glob never enters
            # `<style>/longform/`, which is what we want.
            for wav in sorted(glob.glob(os.path.join(style_dir, "base", "*.wav"))):
                name = os.path.basename(wav)
                if "longform" in name:  # e.g. default_longform_00001.wav
                    stats["longform_skipped"] += 1
                    continue
                key = os.path.splitext(name)[0]
                if key in used:
                    continue
                text = transcripts.get(key, "")
                if not text:
                    stats["no_transcript"] += 1
                    continue
                used.add(key)
                spk_rows.append(f"{wav}|{sid}|{text}")
        if not spk_rows:
            continue
        rows.extend(spk_rows)
        meta[str(sid)] = {"speaker": spk, "lines": len(spk_rows),
                          "gender": EXPRESSO_GENDERS.get(spk)}
    # Count the longform/ corpus directories too (they sit next to base/).
    stats["longform_skipped"] += len(
        glob.glob(os.path.join(read_root, "*", "*", "longform", "*.wav")))
    stats["unused_transcripts"] = len(set(transcripts) - used)
    return rows, meta, stats


# ---------------------------------------------------------------------- #
# MLP (already a stage-2 input filelist)
# ---------------------------------------------------------------------- #
def gather_mlp(filelist: str, check_audio: bool = True
               ) -> Tuple[List[str], dict, dict]:
    """Validate an existing stage-2 input filelist and name its speakers.

    The MLP corpus already ships as ``filelists_2/mlp_linux.txt``
    (``audio_path|sid|transcription``), so there is nothing to gather -- but its
    raw sids carry no name and ``2_combine_corpora.py`` writes a much more
    useful manifest when it can say "2643 = Twilight" instead of "null".  The
    character name is parseable from the sliced-dialogue filename
    (``speaker_map.speaker_name_from_path``), so it is recovered here instead of
    being retyped.

    The filelist itself is never rewritten.  Returns ``(rows, speakers, stats)``
    where ``rows`` is the filelist as read.
    """
    import collections

    import speaker_map as speaker_map_lib

    rows = read_rows(filelist)
    stats = {"bad_lines": 0, "missing_audio": 0, "unparsed_names": 0,
             "multiple_names_per_sid": {}}
    by_sid: Dict[int, "collections.Counter"] = collections.defaultdict(
        collections.Counter)
    for line in rows:
        parts = line.split("|")
        if len(parts) != 3:
            stats["bad_lines"] += 1
            continue
        path, raw_sid, _ = parts
        name = speaker_map_lib.speaker_name_from_path(path)
        if name is None:
            stats["unparsed_names"] += 1
        if check_audio and not os.path.exists(path):
            stats["missing_audio"] += 1
        by_sid[int(raw_sid)][name] += 1

    speakers = {}
    for raw_sid in sorted(by_sid):
        names = by_sid[raw_sid]
        top = names.most_common(1)[0][0]
        others = {k: v for k, v in names.items() if k != top}
        if others:
            stats["multiple_names_per_sid"][str(raw_sid)] = {
                str(top): names[top], **{str(k): v for k, v in others.items()}}
        speakers[str(raw_sid)] = {"speaker": top, "gender": None,
                                  "lines": sum(names.values())}
    return rows, speakers, stats


# ---------------------------------------------------------------------- #
# Driver
# ---------------------------------------------------------------------- #
def read_rows(filelist: str) -> List[str]:
    """Read a stage-2 input filelist, dropping blank lines."""
    with open(filelist, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def write_corpus(name: str, rows: List[str], meta: dict, out_dir: str,
                 extra: dict) -> str:
    """Write ``<name>.txt`` + ``<name>_speakers.json``; return the filelist."""
    os.makedirs(out_dir, exist_ok=True)
    list_path = os.path.join(out_dir, f"{name}.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    side = {"corpus": name, "n_lines": len(rows), "n_speakers": len(meta),
            "speakers": meta, **extra}
    with open(os.path.join(out_dir, f"{name}_speakers.json"), "w",
              encoding="utf-8") as f:
        json.dump(side, f, indent=2)
    print(f"{list_path}: {len(rows)} clips, {len(meta)} speakers")
    return list_path


def write_sidecar_only(name: str, rows: List[str], meta: dict, out_dir: str,
                       extra: dict) -> str:
    """Write only ``<name>_speakers.json`` (the filelist already exists)."""
    os.makedirs(out_dir, exist_ok=True)
    side = {"corpus": name, "n_lines": len(rows), "n_speakers": len(meta),
            "speakers": meta, **extra}
    path = os.path.join(out_dir, f"{name}_speakers.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(side, f, indent=2)
    print(f"{path}: {len(meta)} speakers named, {len(rows)} clips")
    return path


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Build stage-2 input filelists from external corpora")
    p.add_argument("--corpus",
                   choices=["all", "mlp", "vctk", "expresso"], default="all")
    p.add_argument("--out_dir", default="filelists_2",
                   help="where the filelists + speaker sidecars are written")
    p.add_argument("--mlp_filelist", default="filelists_2/mlp_linux.txt",
                   help="existing stage-2 input filelist for the MLP corpus; it "
                        "is validated and its speakers are named, never rewritten")
    p.add_argument("--vctk_root",
                   default="/mnt/data/DataAugmentation/VCTK-Corpus-0.92")
    p.add_argument("--vctk_mic", default="mic1", choices=["mic1", "mic2"],
                   help="VCTK has two mic recordings of every utterance; using "
                        "both would duplicate every transcript")
    p.add_argument("--expresso_root",
                   default="/mnt/data/DataAugmentation/expresso")
    p.add_argument("--expresso_keep_style", action="append", default=[],
                   help="style skipped by default that should be kept "
                        "(currently only 'singing' is meaningful)")
    p.add_argument("--no_check_audio", action="store_true",
                   help="for --corpus mlp: skip the audio-path existence check")
    args = p.parse_args(argv)

    if args.corpus in ("all", "mlp"):
        rows, meta, stats = gather_mlp(args.mlp_filelist,
                                       check_audio=not args.no_check_audio)
        write_sidecar_only("mlp_linux", rows, meta, args.out_dir,
                           {"filelist": args.mlp_filelist, "stats": stats})
        print(f"  {len(meta)} sids named from the filenames  "
              f"bad_lines={stats['bad_lines']}  "
              f"missing_audio={stats['missing_audio']}  "
              f"unparsed_names={stats['unparsed_names']}")
        if stats["multiple_names_per_sid"]:
            print(f"  WARNING: sids with more than one character name: "
                  f"{stats['multiple_names_per_sid']}")

    if args.corpus in ("all", "vctk"):
        rows, meta, stats = gather_vctk(args.vctk_root, mic=args.vctk_mic)
        write_corpus("vctk_linux", rows, meta, args.out_dir,
                     {"root": args.vctk_root, "mic": args.vctk_mic,
                      "stats": stats})
        print(f"  mic={args.vctk_mic}  no_audio={stats['no_audio']}  "
              f"empty_text={stats['empty_text']}  "
              f"audio_without_transcripts="
              f"{stats['audio_without_transcripts']}  "
              f"ignored_dirs={stats['ignored_dirs']}")

    if args.corpus in ("all", "expresso"):
        skip = [s for s in ("longform", "singing")
                if s not in set(args.expresso_keep_style)]
        rows, meta, stats = gather_expresso(args.expresso_root, skip_styles=skip)
        write_corpus("expresso_read_linux", rows, meta, args.out_dir,
                     {"root": args.expresso_root, "skipped_styles": sorted(skip),
                      "stats": stats})
        print(f"  read-only  no_transcript={stats['no_transcript']}  "
              f"longform_skipped={stats['longform_skipped']}  "
              f"style_skipped={stats['style_skipped']}  "
              f"unused_transcripts={stats['unused_transcripts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

