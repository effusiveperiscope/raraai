#!/usr/bin/env python3
"""fdisk-style interactive REPL for stage-2 (``CodeTransformer``) inference.

Loads a stage-2 checkpoint plus, optionally, the frozen stage-1 ``Generator``
that turns sampled codes back into audio, and then lets you poke at the model
from a single-keystroke menu: phonemize a sentence, pick a speaker, sample
codes, listen to / save the result, or compare a free-running sample against a
validation item's ground truth.

Fill-in-the-middle is wired in as ``n``: it drops a random span from the *last*
sample and re-generates it with :meth:`CodeTransformer.infill`, keeping the
untouched prefix and suffix, and prints an ASCII map of where in the clip the
span sat.  The sample itself is left alone -- the filled clip is a separate
result, so ``n`` can be repeated on the same generation -- and only a FIM-trained
checkpoint can do it (right now that means
``checkpoints_stage2/transformer_fim.ckpt``): a pre-FIM run has untrained
SUF/MID sentinel classes, so ``n`` refuses with a note instead of sampling
garbage.  ``generate`` (plain AR) is the other sampling path.

Usage
-----
    # interactive, shipped config + its checkpoint (single keys, fdisk style)
    python scratch/stage2_repl.py

    # a specific run / checkpoint, no stage-1 audio decode
    python scratch/stage2_repl.py \\
        --checkpoint checkpoints_stage2/transformer_v0.5a/last.ckpt --no_audio

    # bf16 weights + sampling (transformer only); press z for the memory report
    python scratch/stage2_repl.py --precision bf16

    # one-shot: phonemize, sample, decode, write a wav, exit
    python scratch/stage2_repl.py --text "Hello there, traveller." --sid 12 \\
        --frames 200 --out scratch/repl_out/hello.wav

    # line-based mode (automatic when stdin is not a TTY)
    python scratch/stage2_repl.py --line_mode < session.txt

Keys (press ``m`` inside the REPL for the same list)
----------------------------------------------------
    m  help                        p  print current settings
    l  list speakers (val split)   c  summarise the last sampled codes
    t  text        (prompt)        y  say = text + generate (prompt)
    g  generate from current text  v  val item (prompt: index, blank = next)
    i  speaker id  (prompt)        f  max code frames (prompt)
    h  temperature (prompt)        k  top-k (prompt, ``none`` disables)
    a  decode the last codes again w  write the last audio to a wav (prompt)
    x  dump the last codes tensor  q  quit (^C / ^D also work)
    b  play the last audio (pydub; space does the same)
    z  report GPU / CPU memory usage (models, peaks, allocator cache)

    n  fill a random span of the last sample -- re-generate it from its own
       kept prefix/suffix without touching the sample (prompt: frames, or
       start:frames; blank = random span, as training draws it).  FIM
       checkpoints only; a/b/c/w/x then act on the filled clip.

    s  switch stage-2 checkpoint (prompt: run name or path; ``?`` lists runs)

Notes
-----
* Speaker ids are in the **stage-1** id space (``speaker_map.py`` canonicalises
  them); that is the only id the frozen ``Generator`` can voice.
* The phonemizer is ``FeatureExtractor(..., load_qwen=False)`` -- g2p_en + the
  g2pen tokenizer only, no ASR / aligner / RMVPE weights and no audio input.
  Same path as ``2_preprocess.py --phones_from transcript``.
* ``{...}`` in the text is ARPAbet you supply yourself: ``hello {W ER1 L D}``
  phonemizes "hello" normally and splices ``WER1LD`` in its place (spaces
  optional, case-insensitive). One group = one word (its symbols are
  concatenated, exactly like g2p writes a pronunciation); use several groups
  for several words. The whole result is tokenized once, so the ids are the
  same shape a whole-sentence phonemization produces.
* ``transformer.eos_token`` / ``fim`` / ``region_cond`` change the codebook
  embedding/head shapes, so a config that disagrees with its checkpoint would
  silently re-initialise them (``load_state_dict_mismatch`` skips whole
  tensors). The layout is therefore read back **from the checkpoint** and the
  config adjusted before the model is built; the printed notes say when.
* Read-only: this never writes a checkpoint or touches a log tree.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import logging
import os
import random
import re
import shutil
import sys
import threading
import time
import traceback
from typing import Optional, Tuple

try:  # POSIX only; the REPL still works without it (no RSS line)
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore[assignment]

# Claim the root logger's handler slot before anything can import the stage-1
# Generator (`load_generator` below, i.e. the first `g`): `modeling/generator.py`
# pulls in the vendored NSF decoder -> `svc_helper.svc.rvc` -> `fairseq`, whose
# `models/wav2vec/wav2vec2_classification` runs
# `logging.basicConfig(level=logging.DEBUG)` at import time and would turn the
# root logger to DEBUG (numba bytecode dumps + faiss INFO).  `basicConfig` is a
# no-op once a handler exists, so doing it here wins.
# See the long comment at the top of 2_training.py.
logging.basicConfig(level=logging.WARNING)

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from commons import load_state_dict_mismatch  # noqa: E402
from modeling.transformer import CodeTransformer, uses_rope  # noqa: E402


# ---------------------------------------------------------------------- #
# fdisk-style single-key input
# ---------------------------------------------------------------------- #
# Long names accepted in --line_mode (key mode only ever needs the letter).
LINE_ALIASES = {
    "help": "m", "status": "p", "info": "p", "print": "p", "list": "l",
    "text": "t", "say": "y", "gen": "g", "generate": "g", "val": "v",
    "value": "v", "sid": "i", "id": "i", "frames": "f", "temp": "h",
    "temperature": "h", "topk": "k", "audio": "a", "write": "w",
    "save": "w", "dump": "x", "codes": "c", "play": "b", "hear": "b",
    "switch": "s", "load": "s", "open": "s", "ckpt": "s",
    "checkpoint": "s", "checkpoints": "s",
    "fill": "n", "infill": "n", "in_fill": "n", "middle": "n", "fim": "n",
    "intensity": "e", "effort": "e", "register": "r",
    "pitch": "j", "word_pitch": "j",
    "slot": "u", "timing": "u", "duration": "u",
    "word": "d", "words": "d", "word_sem": "d", "word_semantic": "d",
    "quit": "q", "exit": "q",
    "memory": "z", "mem": "z", "vram": "z", "gpu": "z", "usage": "z",
}

# Sampling precision (`--precision`).  Stage 2 trains in 32-true (see
# AGENTS.md §5), so fp32 is the default and the reference; bf16/fp16 wrap the
# transformer's generate/infill in autocast, which halves the activation
# traffic and is what the memory report (`z`) exists to compare.
PRECISION_ALIASES = {
    "fp32": "fp32", "float32": "fp32", "f32": "fp32", "32": "fp32",
    "bf16": "bf16", "bfloat16": "bf16", "b16": "bf16",
    "fp16": "fp16", "float16": "fp16", "half": "fp16", "f16": "fp16",
}
PRECISION_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def resolve_precision(name, device: str) -> str:
    """Normalise ``--precision`` and reject what the device cannot do.

    autocast needs a device type with a narrower fast path; CPU bf16 is a
    pessimisation in practice (and fp16 essentially unimplemented), so a
    non-CUDA device silently stays fp32 rather than surprising on speed.
    """
    raw = str(name or "fp32").strip().lower()
    precision = PRECISION_ALIASES.get(raw)
    if precision is None:
        raise SystemExit(f"--precision must be one of "
                         f"{sorted(PRECISION_DTYPES) + ['fp32']} (got {name!r})")
    if precision != "fp32" and not str(device).startswith("cuda"):
        print(f"note: --precision {precision} needs CUDA; using fp32 on "
              f"{device}")
        return "fp32"
    return precision


# ---------------------------------------------------------------------- #
# Phonemization with manual ARPAbet ({...})
# ---------------------------------------------------------------------- #
# One symbol: consonants (HH, L, ...) or a stressed/unstressed vowel (AH0,
# OW1, ...).  The base is checked against the real ARPAbet inventory, so a typo
# like "{hello}" / "{XXQ}" is caught here instead of reaching the tokenizer
# (which would turn it into [UNK]).
_ARPABET_TOKEN = re.compile(r"^([A-Z]+)([0-2]?)$")
_ARPABET_SYMBOLS = frozenset(
    "AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY "
    "P R S SH T TH UH UW V W Y Z ZH".split())
_BRACE_GROUP = re.compile(r"\{([^{}]*)\}")


def split_arpabet_braces(text: str):
    """Split ``text`` into ``("text", chunk)`` / ``("arpabet", phones)`` parts.

    A ``{...}`` group is ARPAbet the user supplied by hand; everything else is
    ordinary text that still goes through g2p.  The braces are not kept, and
    the group's symbols are concatenated into **one word** exactly like g2p
    writes a pronunciation (``{F AO1 R}`` -> ``FAO1R``, not ``F AO1 R``);
    write one group per word.  This matters: the tokenizer merges phones
    inside a word but not across spaces (measured ``FAO1R`` -> 2 ids vs
    ``F AO1 R`` -> 6).

    Raises:
        ValueError: unbalanced ``{`` / ``}``, an empty group, or a token that
            does not look like an ARPAbet symbol.
    """
    text = str(text)
    segments = []
    pos = 0
    for match in _BRACE_GROUP.finditer(text):
        gap = text[pos:match.start()]
        if "{" in gap or "}" in gap:
            raise ValueError("unbalanced '{' or '}' (write ARPAbet as "
                             "{HH AH0 L OW1})")
        segments.append(("text", gap))
        tokens = match.group(1).split()
        if not tokens:
            raise ValueError("empty {...} group; put ARPAbet in it "
                             "(e.g. {W ER1 L D})")
        for token in tokens:
            match_token = _ARPABET_TOKEN.match(token.upper())
            if (match_token is None
                    or match_token.group(1) not in _ARPABET_SYMBOLS):
                raise ValueError(f"{token!r} is not an ARPAbet symbol "
                                 f"(expected e.g. HH, AH0, L, OW1)")
        segments.append(("arpabet", "".join(t.upper() for t in tokens)))
        pos = match.end()
    tail = text[pos:]
    if "{" in tail or "}" in tail:
        raise ValueError("unbalanced '{' or '}' (write ARPAbet as "
                         "{HH AH0 L OW1})")
    segments.append(("text", tail))
    return segments


def _bert_word(token):
    """The aligner's own token form, for the BERT encoder (``None`` if empty).

    ``hints/build.py --word-bert`` feeds ``features.transcript_words(text)``,
    i.e. the token *reduced* exactly like the forced aligner's ``clean_token``
    (letters, digits and ``'``; ``top-notch`` -> ``topnotch``, punctuation
    stripped).  Inference must see the same string, or the 768-d state has a
    different meaning for the same word.  A token with no kept characters
    (a lone ``--`` / ``...``) has no semantic state -- it goes through the
    trained ``word_null`` rather than embedding punctuation.
    """
    if token is None:
        return None
    from features import _aligner_keeps
    word = "".join(ch for ch in str(token) if _aligner_keeps(ch))
    return word or None


def phonemize_with_arpabet(fe, text: str):
    """``(phones, ids, word_index, word_texts)`` for ``text`` with ``{...}``.

    Ordinary text is phonemized with the same sentence-level POS tagging
    `FeatureExtractor.phonemize_text` uses (so heteronyms still resolve from
    context across the whole sentence), and each ``{...}`` group replaces its
    part with the ARPAbet exactly as written.  The combined phone string is
    then tokenized **once**, which is what keeps the result compatible with
    ids produced by phonemizing the same sentence normally.

    ``word_index`` maps every phone id to its word (via the same
    `FeatureExtractor.tokenize_with_word_spans` the dataset path uses) and
    ``word_texts`` is one entry per word: the source token, or ``None`` for a
    ``{...}`` group.  Those two are what let the REPL attach a per-word BERT
    state (`hint_word_bert`) to an arbitrary sentence.

    Args:
        fe: A `FeatureExtractor` (only ``hphzr`` / ``tokenizer_g2p`` used).
        text: Transcript, optionally containing ``{HH AH0 L OW1}`` groups.

    Returns:
        ``(phones, ids, word_index, word_texts)``; all empty when the text has
        no phonemes.

    Raises:
        ValueError: malformed braces (see `split_arpabet_braces`).
    """
    # Drop `*sigh*`-style stage directions (and unwrap emphasis) before the
    # word list is built, so `word_index` / `word_texts` still describe the
    # phones that `phonemize_text` would produce for this text.
    from features import strip_stage_directions
    text = strip_stage_directions(text)
    segments = split_arpabet_braces(text)
    words = [w for kind, chunk in segments if kind == "text"
             for w in chunk.split()]
    tags = fe.hphzr.tag_words(words)
    phone_words, word_texts = [], []
    cursor = 0
    for kind, chunk in segments:
        if kind == "arpabet":
            phone_words.append(chunk)
            word_texts.append(None)
            continue
        chunk_words = chunk.split()
        for word, tag in zip(chunk_words, tags[cursor:cursor + len(chunk_words)]):
            phone_words.append(fe.hphzr.phonemize(word, tag=tag))
            word_texts.append(word)
        cursor += len(chunk_words)
    phones = " ".join(p for p in phone_words if p)
    if not phones.strip():
        return "", [], [], []
    ids, word_index = fe.tokenize_with_word_spans(phone_words)
    if fe.tokenizer_g2p.unk_token_id in ids:
        print("warning: the tokenizer mapped part of the input to [UNK]; "
              "check the symbols in your {...} groups")
    return phones, ids, word_index, word_texts


class KeyReader:
    """One keystroke per command (cbreak), with a line-based fallback.

    ``tty.setcbreak`` clears both ``ECHO`` and ``ISIG``, so the caller must echo
    the key itself and treat ``\\x03`` / ``\\x04`` (^C / ^D) as quit -- the
    terminal will not turn them into signals while a key is being read.
    """

    def __init__(self, stream=None, force_line_mode: bool = False):
        self.stream = stream or sys.stdin
        self.line_mode = bool(force_line_mode) or not self.stream.isatty()

    def describe(self) -> str:
        return "line" if self.line_mode else "single-key"

    def read_command(self) -> Tuple[Optional[str], str]:
        """Return ``(key, rest)``; ``key`` is ``None`` on EOF / ^C / ^D.

        In line mode the whole line is parsed so the long names work
        (``text hello there`` -> ``("t", "hello there")``).
        """
        if self.line_mode:
            try:
                line = input("stage2: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return None, ""
            if not line:
                return "", ""
            head, _, rest = line.partition(" ")
            head = head.lower()
            return LINE_ALIASES.get(head, head[:1]), rest.strip()

        ch = self._read_byte()
        if ch is None:
            return None, ""
        return ch, ""

    def _read_byte(self) -> Optional[str]:
        import termios
        import tty

        fd = self.stream.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            ch = self.stream.read(1)
            if ch == "\x1b":  # ESC: swallow an arrow-key / function-key escape
                for _ in range(2):
                    self.stream.read(1)
                return ""
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

        if ch in ("\x03", "\x04", ""):  # ^C / ^D
            return None
        if ch in ("\r", "\n"):
            return ""
        if not ch.isprintable() or not ch.isascii():
            return ""
        return ch

    def read_line(self, prompt: str) -> Optional[str]:
        """A normal cooked line (for commands that take a string argument)."""
        try:
            return input(prompt)
        except (EOFError, KeyboardInterrupt):
            print()
            return None


# ---------------------------------------------------------------------- #
# checkpoint helpers
# ---------------------------------------------------------------------- #
def read_state_dict(path: str) -> dict:
    """Load a Lightning checkpoint and strip the ``net.`` module prefix."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(k.startswith("net.") for k in state):
        state = {k[len("net."):]: v for k, v in state.items()
                 if k.startswith("net.")}
    return state


def layout_from_rows(rows: int, vocab: int) -> Tuple[bool, bool]:
    """``(eos_token, fim)`` implied by a codebook's class count.

    ``DelayCodebookDecoder.num_classes = vocab + 1 (pad) + eos + 2*fim``, so the
    four layouts are exactly ``vocab + 1..4`` (see ``modeling/transformer.py``).
    """
    table = {1: (False, False), 2: (True, False),
             3: (False, True), 4: (True, True)}
    extra = int(rows) - int(vocab)
    if extra not in table:
        raise ValueError(
            f"a codebook has {rows} classes, i.e. vocab ({vocab}) + {extra}; "
            f"expected +1 (plain), +2 (EOS), +3 (FIM) or +4 (EOS+FIM)")
    return table[extra]


def load_weights(model, state: dict) -> Tuple[int, int, int, int]:
    """Load ``state`` into ``model`` via the repo helper, with a summary.

    ``load_state_dict_mismatch`` prints each key it skips but says nothing about
    the model-side keys the checkpoint never had, so both counts are reported.
    Returns ``(n_loaded, n_ckpt, n_bad, n_missing)``.
    """
    model_state = model.state_dict()
    loaded = [k for k, v in state.items()
              if k in model_state and model_state[k].shape == v.shape]
    bad = [k for k, v in state.items()
           if k not in model_state or model_state[k].shape != v.shape]
    missing = [k for k in model_state if k not in state]
    load_state_dict_mismatch(model, state)
    if missing:
        shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
        print(f"NOTE: {len(missing)} model tensor(s) absent from the "
              f"checkpoint (left at their initial values): {shown}")
    return len(loaded), len(state), len(bad), len(missing)


def list_checkpoints(current: Optional[str] = None,
                     root: str = "checkpoints_stage2") -> None:
    """Print the stage-2 checkpoints available under ``root``.

    One row per run *file* (``last.ckpt`` / ``best-checkpoint.ckpt``), with size
    and mtime, marking the one currently loaded.  Needs no model, so it works
    before (or instead of) loading anything.
    """
    if not os.path.isdir(root):
        print(f"{root}/ not found (run from the repo root).")
        return
    here = os.path.abspath(current) if current else None
    rows = []
    for name in sorted(os.listdir(root)):
        run = os.path.join(root, name)
        if not os.path.isdir(run):
            continue
        for ckpt in ("last.ckpt", "best-checkpoint.ckpt"):
            path = os.path.join(run, ckpt)
            if not os.path.isfile(path):
                continue
            st = os.stat(path)
            rows.append((name, ckpt, path, st.st_size / 1e9,
                         time.strftime("%Y-%m-%d %H:%M",
                                       time.localtime(st.st_mtime))))
    if not rows:
        print(f"No .ckpt files under {root}/.")
        return
    print(f"Stage-2 checkpoints under {root}/:")
    for name, ckpt, path, gb, mtime in rows:
        mark = "   <- loaded" if here == os.path.abspath(path) else ""
        print(f"  {name:<24} {ckpt:<20} {gb:5.2f} GB  {mtime}{mark}")
    print("  switch at runtime with `s`, then a run name or a path "
          "(a directory resolves to last.ckpt, then best-checkpoint.ckpt)")


# ---------------------------------------------------------------------- #
# Live playback (pydub)
# ---------------------------------------------------------------------- #
def _has_simpleaudio() -> bool:
    return importlib.util.find_spec("simpleaudio") is not None


def playback_backend() -> Optional[str]:
    """Which backend pydub will pick (its order: simpleaudio, pyaudio, ffplay)."""
    if _has_simpleaudio():
        return "simpleaudio"
    if importlib.util.find_spec("pyaudio") is not None:
        return "pyaudio"
    if shutil.which("ffplay"):
        return "ffplay"
    return None


class AudioPlayer:
    """Play a decoded waveform through pydub, without blocking the REPL.

    ``pydub.playback.play`` needs a backend -- ``simpleaudio``, ``pyaudio``, or
    else the ``ffplay`` binary, which is what this env has: it exports a temp wav
    and runs ``ffplay -nodisp -autoexit``, and *that call blocks until the clip is
    over*.  It therefore runs on a daemon thread, and the next ``play()`` waits
    for the previous clip to finish instead of starting a second, overlapping
    ffplay.

    The audio is handed over as raw 16-bit PCM at the stage-1 sample rate, so
    building the segment needs no file round-trip (the ffplay backend writes its
    own temp wav).
    """

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[str] = None

    def describe(self) -> str:
        """One-line summary of the backend, for ``p`` (print settings)."""
        if self._error:
            return f"failed ({self._error})"
        if importlib.util.find_spec("pydub") is None:
            return "off (pydub not installed)"
        backend = playback_backend()
        if backend is None:
            return "off (needs simpleaudio, pyaudio or the ffplay binary)"
        if backend == "ffplay":
            return "pydub -> ffplay (blocking, backgrounded)"
        return f"pydub -> {backend}"

    def wait(self) -> None:
        """Wait for the clip started by the previous ``play()`` to finish."""
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()
        self._thread = None

    def play(self, wave, sampling_rate: int) -> bool:
        """Start playing ``wave`` (a 1-D float waveform); returns success."""
        if self._error:
            print(f"  playback unavailable: {self._error}")
            return False
        try:
            import numpy as np
            from pydub import AudioSegment
        except Exception as exc:  # pydub missing
            self._error = f"pydub not importable ({exc})"
            print(f"  playback unavailable: {self._error}"
                  f"  -- pip install pydub (needs ffplay, pyaudio or simpleaudio)")
            return False

        backend = playback_backend()
        if backend is None:
            self._error = ("no playback backend (install simpleaudio or pyaudio, "
                           "or put ffplay on PATH)")
            print(f"  playback unavailable: {self._error}")
            return False

        self.wait()  # never overlap two clips
        pcm = (np.clip(np.asarray(wave, dtype=np.float32), -1.0, 1.0)
               * 32767.0).astype("<i2")
        seg = AudioSegment(data=pcm.tobytes(), sample_width=2,
                           frame_rate=int(sampling_rate), channels=1)
        self._thread = threading.Thread(target=self._run, args=(seg,),
                                        daemon=True)
        self._thread.start()
        print(f"  playing {seg.duration_seconds:.2f} s @ {sampling_rate} Hz "
              f"through pydub/{backend}")
        return True

    def _run(self, seg) -> None:
        try:
            if playback_backend() == "ffplay":
                self._run_ffplay(seg)
            else:
                # `AudioSegment.play()` does not exist in pydub 0.25.1 (this
                # env); the entry point is the module-level function.
                from pydub.playback import play as pydub_play

                pydub_play(seg)
        except Exception as exc:  # no audio device, broken backend, ...
            self._error = str(exc)
            print(f"  playback failed: {exc}")

    @staticmethod
    def _run_ffplay(seg) -> None:
        """pydub's ffplay backend, minus its per-frame status spam.

        ``pydub.playback.play()`` shells out to ``ffplay -nodisp -autoexit`` and
        lets ffplay write its stats line to *our* stderr, which fights with the
        REPL prompt; it also uses ``subprocess.call`` and throws the exit code
        away, so a missing audio device would look like success.  This is the
        same binary with the same arguments plus ``-loglevel error``, and it
        checks the exit code -- pydub still builds and exports the segment.
        """
        import subprocess
        import tempfile

        player = shutil.which("ffplay") or "ffplay"
        with tempfile.NamedTemporaryFile("w+b", suffix=".wav",
                                         delete=False) as handle:
            path = handle.name
        try:
            seg.export(path, format="wav")
            proc = subprocess.run(
                [player, "-nodisp", "-autoexit", "-hide_banner",
                 "-loglevel", "error", path],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE)
            if proc.returncode != 0:
                lines = proc.stderr.decode("utf-8", "replace").strip().splitlines()
                raise RuntimeError(lines[-1] if lines
                                   else f"ffplay exited {proc.returncode}")
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


def load_generator(checkpoint_path: str, config, device: str):
    """Build the frozen stage-1 ``Generator`` (mirrors 2_training/2_preprocess)."""
    from modeling.generator import Generator

    net_g = Generator(hp=config)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(k.startswith("net_g.") for k in state):
        state = {k[len("net_g."):]: v for k, v in state.items()
                 if k.startswith("net_g.")}
    load_state_dict_mismatch(net_g, state)
    net_g.to(device)
    net_g.eval()
    return net_g


# ---------------------------------------------------------------------- #
# Fill-in-the-middle ASCII map
# ---------------------------------------------------------------------- #
def ascii_fill_map(n: int, start: int, span: int, width: int = 64):
    """Two aligned rows showing *where* a FIM span sits in a clip.

    Row 1 renders the clip as ``width`` columns of code frames: ``#`` for the
    span the model re-generates and ``.`` for the audio that is kept, so the
    hole's position *and* its share of the clip are visible at a glance.  Row 2
    puts a ``^`` under each end of the span (they land on the same column for a
    sub-column span).  ``width`` is capped at ``n``, so a column never covers
    more than one frame unless the clip is longer than the map; the caller
    prints the frame/second numbers, which are the precise answer.

    Args:
        n: total number of code frames in the clip.
        start, span: the re-generated span is ``[start, start + span)``.

    Returns:
        ``[map_row, caret_row]`` -- same columns, no prefix (the caret row is
        right-trimmed, so it is usually shorter).
    """
    n = max(1, int(n))
    start = min(max(int(start), 0), n)
    end = min(max(start, start + int(span)), n)
    width = max(1, min(int(width), n))

    def col(frame: int) -> int:
        """Map a frame index to a column, clamped into the map."""
        return max(0, min(width - 1, int(frame * width / n)))

    cells = []
    for c in range(width):
        lo, hi = c * n / width, (c + 1) * n / width
        # A column is '#' when it overlaps the span at all: a short span must
        # still show up, even when it is thinner than one column.
        cells.append("#" if hi > start and lo < end else ".")
    carets = [" "] * width
    carets[col(start)] = "^"
    carets[col(end)] = "^"
    return ["".join(cells).rstrip(), "".join(carets).rstrip()]


# ---------------------------------------------------------------------- #
# REPL
# ---------------------------------------------------------------------- #
class Stage2REPL:
    """Small state machine around ``CodeTransformer.generate`` (+ stage 1)."""

    def __init__(self, config, checkpoint: str, device: str,
                 generator_checkpoint: Optional[str] = None,
                 audio: bool = True,
                 out_dir: str = "scratch/repl_out",
                 auto_play: bool = False,
                 reader: Optional[KeyReader] = None,
                 hint_mode: Optional[str] = None,
                 precision: str = "fp32",
                 stage1_sid_map: Optional[str] = None):
        self.config = config
        self.checkpoint = checkpoint
        self.device = device
        self.precision = resolve_precision(precision, device)
        self.generator_checkpoint = generator_checkpoint
        from stage2_wer import load_sid_map
        if stage1_sid_map is None:
            stage1_sid_map = config.train.get("stage1_speaker_map", None)
        if stage1_sid_map and config.train.get("speaker_map", None):
            print("WARNING: train.speaker_map is set; ignoring "
                  "stage1_speaker_map for decode.")
            stage1_sid_map = None
        self.stage1_sid_map = load_sid_map(stage1_sid_map)
        if self.stage1_sid_map:
            print(f"stage-1 decode speaker map: {len(self.stage1_sid_map)} id(s)")
        self.audio = bool(audio) and generator_checkpoint is not None
        self._audio_disabled = not audio
        self.out_dir = out_dir
        self.reader = reader or KeyReader()
        # Live playback (`b`) + the `--play` auto-play toggle.
        self.auto_play = bool(auto_play)
        self.player = AudioPlayer()
        # Hint state for generation.  ``cond`` uses the loaded val item's
        # ``.cond`` hints when there is one (arbitrary text has none, so it
        # degrades to the trained null state); ``null`` ignores the item's
        # hints; ``omit`` sends no vectors at all (pre-Phase-0 behaviour, out
        # of distribution for a hint model).
        mode = str(hint_mode or config.train.get("sample_hints", "null")
                   or "null").lower()
        if mode == "zero":
            mode = "null"
        if mode not in ("cond", "null", "omit"):
            raise SystemExit(
                f"--hints must be cond | null | omit (got {mode!r})")
        self.hint_mode = mode

        # ---- per-channel hint controls (M5) -------------------------------- #
        # Applied on top of the base state at every generate/infill:
        #   hint_intensity / hint_register   absolute level (None = base value)
        #   hint_pitch_st                    per-word shift in semitones
        #                                    (2 st per bucket, rounded)
        #   hint_slot_shift                  per-word log2 duration shift
        #                                    (+1 = twice as long)
        #   hint_word_mode                   cond | text | null
        #   hint_word_text                   sentence whose per-word BERT states
        #                                    are injected (the swap test)
        self.hint_intensity: Optional[int] = None
        self.hint_register: Optional[int] = None
        self.hint_pitch_st: float = 0.0
        self.hint_slot_shift: int = 0
        self.hint_word_mode: str = "cond"
        self.hint_word_text: Optional[str] = None
        self._base_hints: Optional[dict] = None   # last val item's levels
        # Per-phone word index and the source token of every word for the
        # *current* text (None = a {...} ARPAbet group).  Set by `set_text`;
        # a val item (`v`) has no text, so they stay empty and the word channel
        # can only come from its own `.cond`.
        self.word_index: Optional[list] = None
        self.word_texts: list = []
        self._word_cache: dict = {}


        # Code frames, not decoder frames: `Generator(down_factor=4)` decimates
        # the 100 Hz mel frames to 25 Hz, which is the rate the transformer
        # actually predicts at.  Getting this wrong made every printed duration
        # 4x too short (a 512-frame "5.1 s" request is really 20.5 s).
        self.frame_rate = (
            int(config.data.sampling_rate)
            // int(config.data.hop_length)
            // int(config.data.get("code_down_factor", 4) or 4))
        # Decoding defaults come from the config's sampling block, so the REPL
        # starts at the same operating point 2_training.py samples with.
        self.text: str = ""
        self.phone_ids: list[int] = []
        self.sid: int = int(config.train.get("ood_sid", 0) or 0)
        self.frames: int = int(config.train.get("sample_frames", 128) or 128)
        self.temperature: float = float(
            config.train.get("sample_temperature", 1.0) or 1.0)
        top_k = config.train.get("sample_top_k", None)
        self.top_k: Optional[int] = int(top_k) if top_k else None
        # Fill (`n`) candidate count: >1 ranks by the suffix-lookahead seam CE
        # (needs a checkpoint trained with transformer.fim_lookahead > 0).
        self.infill_best_of: int = 1

        self.net: Optional[CodeTransformer] = None
        # The eos/fim/region_cond layout is *derived from the checkpoint* (see
        # `_align_special_layout`), so the config is mutated on every load.  The
        # values as configured are kept here so switching checkpoints at runtime
        # (`s`) always starts from the same place instead of compounding.
        self._layout_originals = {
            k: config.transformer.get(k, None)
            for k in ("eos_token", "fim", "region_cond", "length_cond",
                      "fim_lookahead",
                      "hints", "decoder_hints", "hint_monotone",
                      "hint_film_memory", "hint_decoder_proj",
                      "hint_word_sem", "hint_word_dim")}
        self._fe = None            # FeatureExtractor (lazy: g2p + tokenizer)
        self._net_g = None         # stage-1 Generator (lazy)
        self._net_g_failed = False
        self._val_ds = None
        self._val_cursor = 0
        self._writes = 0

        self.last_codes = None     # (K, n) long, on self.device -- the generation
        self.last_sid: Optional[int] = None
        self.last_audio = None     # (T,) float32 CPU waveform
        self.last_f0 = None        # (T,) float32 CPU f0 in Hz
        # The most recent `n` (fill): where the span sat and the spliced codes
        # that replaced it.  `last_codes` stays untouched, so a second `n`
        # fills a span of the same clean generation instead of piling one
        # re-generated span on top of another (`fill_codes` is only what the
        # output commands look at, see `current_codes`).
        self.last_fill: Optional[Tuple[int, int]] = None  # (start, span)
        self.fill_codes = None     # (K, n) long -- generation with that span filled

    @property
    def current_codes(self):
        """What ``a``/``b``/``c``/``w``/``x`` look at: filled clip if any."""
        return self.fill_codes if self.fill_codes is not None else self.last_codes

    def clear_fill(self) -> None:
        """Forget the last fill result (a new generation, or a new model)."""
        self.last_fill = None
        self.fill_codes = None

    @contextlib.contextmanager
    def _autocast(self):
        """Autocast the sampling path according to ``--precision``.

        The transformer's weights are already cast to this dtype at build time
        (see ``_module_dtype``); autocast additionally keeps numerically
        sensitive ops (softmax / layer norm / CE) in fp32, which is what makes
        fp16 weights safe.  The stage-1 vocoder is *not* run under this (see
        ``_decode_audio``): its pitch path deliberately runs in an
        ``autocast(enabled=False)`` fp32 block, so it stays fp32 end to end.
        ``fp32`` is a plain no-op context, so the default is unchanged.
        """
        dtype = PRECISION_DTYPES.get(self.precision)
        if dtype is None:
            yield
            return
        with torch.autocast(device_type=torch.device(self.device).type,
                            dtype=dtype):
            yield

    # ------------------------------------------------------------------ #
    # Models
    # ------------------------------------------------------------------ #
    def _module_dtype(self) -> torch.dtype:
        """dtype the transformer's weights are built in (fp32 by default)."""
        return PRECISION_DTYPES.get(self.precision, torch.float32)

    def load_models(self) -> None:
        """Build the transformer (+ stage-1 Generator) and report the load."""
        print(f"Stage-2 checkpoint: {self.checkpoint}")
        state = read_state_dict(self.checkpoint)
        self._align_special_layout(state)
        # Built directly in `--precision`'s dtype: the checkpoint's fp32 tensors
        # are cast by `load_state_dict`, so the fp32 model never exists on the
        # GPU (and the memory report measures halved weight storage).
        self.net = CodeTransformer(self.config).to(
            device=self.device, dtype=self._module_dtype())
        print(self.net.describe())
        if self.precision != "fp32":
            print(f"Precision: {self.precision} weights "
                  f"({self._module_dtype()}), fp32 autocast for softmax/norm")
        n_loaded, n_ckpt, n_bad, n_missing = load_weights(self.net, state)
        print(f"Loaded {n_loaded}/{n_ckpt} checkpoint tensor(s)"
              + (f"; {n_bad} skipped, {n_missing} missing"
                 if n_bad or n_missing else ""))
        if getattr(self.net, "hints", False):
            print(f"Hints: {self.hint_state()}")
        else:
            print("Hints: n/a (this checkpoint has no hint tables)")
        self.net.eval()

        self.frames = self._clamp_frames(self.frames)
        if self.audio:
            # Eager: a bad stage-1 checkpoint should fail now, not on the first
            # `g`.  Failures are non-fatal -- the codes are still inspectable.
            try:
                self._generator()
            except Exception as exc:  # environment specific (missing weights)
                self._net_g_failed = True
                print(f"WARNING: could not load the stage-1 Generator: {exc}")
                print("         Sampling still works; audio decode is off.")

    def _reset_layout(self) -> None:
        """Restore the configured eos/fim/region_cond before re-deriving them."""
        if not hasattr(self, "_layout_originals"):
            return
        tcfg = self.config.transformer
        for key, value in self._layout_originals.items():
            tcfg[key] = value

    def _align_special_layout(self, state: dict,
                              source: Optional[str] = None) -> None:
        """Match eos/fim/region_cond to the checkpoint's class layout.

        ``num_classes`` grew with every stage-2 feature (EOS in v0.5, SUF/MID in
        v0.6-fim), and ``load_state_dict_mismatch`` *skips* a tensor whose shape
        disagrees -- so a config that is newer than its checkpoint would silently
        re-initialise every codebook embedding and head. Infer the layout from
        the checkpoint instead and say so.

        Called before *every* model build, so it starts by restoring the
        configured values (`_reset_layout`): switching checkpoints at runtime
        must not inherit the previous checkpoint's layout when a key is missing.
        """
        source = source or self.checkpoint
        self._reset_layout()
        tcfg = self.config.transformer
        key = "decoder.embeds.0.weight"
        w = state.get(key)
        if w is None:
            print(f"WARNING: {key} not in {source}; keeping the config's "
                  f"layout (eos_token={tcfg.get('eos_token')}, "
                  f"fim={tcfg.get('fim')}).")
            return
        vocab = int(self.config.codec.codebook_size)
        try:
            eos, fim = layout_from_rows(int(w.shape[0]), vocab)
        except ValueError as exc:
            raise SystemExit(f"ERROR: {source}: {exc}")
        if bool(tcfg.get("eos_token")) != eos or bool(tcfg.get("fim")) != fim:
            print(f"NOTE: {source} uses eos_token={eos}, fim={fim} "
                  f"(config said eos_token={tcfg.get('eos_token')}, "
                  f"fim={tcfg.get('fim')}); using the checkpoint's layout.")
            tcfg.eos_token = eos
            tcfg.fim = fim
        region = "decoder.region_emb.weight" in state
        if bool(tcfg.get("region_cond")) != region:
            print(f"NOTE: region_cond={region} in {source}; using that.")
            tcfg.region_cond = region
        # Span-length conditioning (v0.6-fim2) adds its own table; a config that
        # is newer than its checkpoint would otherwise build a fresh, untrained
        # len_emb and quietly degrade every infill.
        length = "decoder.len_emb.weight" in state
        if bool(tcfg.get("length_cond")) != length:
            print(f"NOTE: length_cond={length} in {source}; using that.")
            tcfg.length_cond = length
        # RoPE is parameter-free, so a checkpoint can only record it with the
        # persistent ``rope_marker`` buffer.  A model built with the wrong flag
        # would load *every* tensor and silently score positions differently.
        rope = uses_rope(state)
        if bool(tcfg.get("rope", False)) != rope:
            print(f"NOTE: rope={rope} in {source}; using that.")
            tcfg.rope = rope
        # FIM suffix lookahead is parameter-free too: it only changes the order
        # the *training* stream was laid out in, so the persistent
        # `fim_layout_marker` is the only trace.  A wrong value loads every
        # tensor and silently disables `infill_seam_score` / `best_of`.
        look = 0
        keys = set(state)
        for key in keys:
            if key == "fim_layout_marker" or key.endswith(".fim_layout_marker"):
                look = int(state[key].reshape(-1)[0].item())
                break
        if int(tcfg.get("fim_lookahead", 0) or 0) != look:
            print(f"NOTE: fim_lookahead={look} in {source}; using that.")
            tcfg.fim_lookahead = look
        # Conditioning-hint layout.  These are *not* shape-compatible: a hint
        # model loaded into a non-hint config silently drops every hint tensor,
        # and a monotone/FiLM checkpoint loaded into the table model loses the
        # trained direction (which reads as "the channel does nothing").
        def has(suffix: str) -> bool:
            return any(k == suffix or k.endswith("." + suffix) for k in keys)

        hints = any(has(k) for k in ("int_emb.weight", "int_dir",
                                    "pitch_emb.weight", "pitch_val_dir",
                                    "word_proj.weight"))
        if bool(tcfg.get("hints", False)) != hints:
            print(f"NOTE: hints={hints} in {source}; using that.")
            tcfg.hints = hints
        if not hints:
            # A hints-less checkpoint has no `word_proj`; leaving the config's
            # flag on would make `_word_hint` project through a `None`.
            tcfg.hint_word_sem = False
        if hints:
            dec_hints = has("dec_int_emb.weight") or has("dec_int_dir")
            if bool(tcfg.get("decoder_hints", False)) != dec_hints:
                print(f"NOTE: decoder_hints={dec_hints} in {source}; using that.")
                tcfg.decoder_hints = dec_hints
            monotone = has("int_dir")
            if bool(tcfg.get("hint_monotone", False)) != monotone:
                print(f"NOTE: hint_monotone={monotone} in {source}; using that.")
                tcfg.hint_monotone = monotone
            film = has("film_gamma.weight")
            if bool(tcfg.get("hint_film_memory", False)) != film:
                print(f"NOTE: hint_film_memory={film} in {source}; using that.")
                tcfg.hint_film_memory = film
            proj = has("hint_out.weight")
            if bool(tcfg.get("hint_decoder_proj", False)) != proj:
                print(f"NOTE: hint_decoder_proj={proj} in {source}; using that.")
                tcfg.hint_decoder_proj = proj
            # V2.1 per-word BERT state.  Its width is a parameter dimension and
            # its *presence* is a trained projection, so both come from the
            # checkpoint -- a config that disagrees would silently drop the
            # trained table (hint_word_sem=false) or build a zero one.
            word = has("word_proj.weight")
            if bool(tcfg.get("hint_word_sem", False)) != word:
                print(f"NOTE: hint_word_sem={word} in {source}; using that.")
                tcfg.hint_word_sem = word
            if word:
                for k in keys:
                    if k == "word_proj.weight" or k.endswith(".word_proj.weight"):
                        dim = int(state[k].shape[1])
                        if int(tcfg.get("hint_word_dim", 768) or 768) != dim:
                            print(f"NOTE: hint_word_dim={dim} in {source}; "
                                  f"using that.")
                            tcfg.hint_word_dim = dim
                        break
            # Hints V2 pitch representation.  The anchor count is a parameter
            # dimension, so a config that disagrees would fail to load; take it
            # from the checkpoint like everything else here.
            anchors = has("pitch_val_dir")
            if (str(tcfg.get("hint_pitch_mode", "table") or "table")
                    != ("anchors" if anchors else "table")):
                print(f"NOTE: hint_pitch_mode="
                      f"{'anchors' if anchors else 'table'} in {source}; using "
                      f"that.")
                tcfg.hint_pitch_mode = "anchors" if anchors else "table"
            if anchors:
                for k in keys:
                    if k == "pitch_val_dir" or k.endswith(".pitch_val_dir"):
                        count = int(state[k].shape[0])
                        if int(tcfg.get("hint_pitch_anchors", 3) or 3) != count:
                            print(f"NOTE: hint_pitch_anchors={count} in "
                                  f"{source}; using that.")
                            tcfg.hint_pitch_anchors = count
                        break

    def _generator(self):
        """The frozen stage-1 Generator, or ``None`` when audio is off."""
        if not self.audio or self._net_g_failed:
            return None
        if self._net_g is None:
            print(f"Loading the stage-1 Generator from "
                  f"{self.generator_checkpoint} ...")
            self._net_g = load_generator(self.generator_checkpoint, self.config,
                                         self.device)
        return self._net_g

    def _extractor(self):
        """The phonemizer (CPU, no Qwen weights: text -> ids only)."""
        if self._fe is None:
            from features import FeatureExtractor

            print("Loading the phonemizer (g2p_en + g2pen tokenizer, CPU) ...")
            self._fe = FeatureExtractor(self.config, device="cpu",
                                        load_qwen=False)
        return self._fe

    # ------------------------------------------------------------------ #
    # Runtime checkpoint switching
    # ------------------------------------------------------------------ #
    def _resolve_switch_path(self, raw: str) -> Optional[str]:
        """Turn a run name / directory / file into a checkpoint path.

        ``?``/``list`` prints the candidates instead.  Returns ``None`` when
        nothing usable was named (so the caller can keep the current model).
        """
        raw = os.path.expanduser(str(raw or "").strip())
        if not raw:
            return None
        if raw.lower() in ("?", "list", "ls"):
            list_checkpoints(self.checkpoint)
            return None
        if os.path.isfile(raw):
            return raw
        candidates = []
        if os.path.isdir(raw):
            candidates = [os.path.join(raw, n)
                          for n in ("last.ckpt", "best-checkpoint.ckpt")]
        else:
            # A bare run name (or a partial path) inside the stage-2 tree.
            run = os.path.join("checkpoints_stage2", raw)
            if os.path.isdir(run):
                candidates = [os.path.join(run, n)
                              for n in ("last.ckpt", "best-checkpoint.ckpt")]
        for path in candidates:
            if os.path.isfile(path):
                return path
        if os.path.isdir(raw) and any(
                os.path.isdir(os.path.join(raw, d)) for d in os.listdir(raw)):
            # A parent directory (e.g. `checkpoints_stage2`): list its runs
            # instead of guessing which one was meant.
            print(f"{raw} holds several runs -- pick one:")
            list_checkpoints(self.checkpoint, root=raw)
            return None
        print(f"No checkpoint for {raw!r} -- press s again and enter `?` "
              f"(or `list`) to see the available runs.")
        return None

    def switch_checkpoint(self, raw: str) -> bool:
        """(Re)load a different stage-2 checkpoint without restarting.

        The model is rebuilt from scratch: the eos/fim/region_cond layout is
        re-derived from the new checkpoint, so a run with a different special
        layout loads correctly in the same session.  The build happens *before*
        the swap, so a bad path or an unloadable file leaves the current model
        (and the session) intact.  Stage-1 (``audio``) is untouched -- it is
        keyed to the code space, which all stage-2 runs share.
        """
        if raw is None or not str(raw).strip():
            print("Checkpoint unchanged.")
            return False
        target = self._resolve_switch_path(str(raw))
        if target is None:
            return False
        if os.path.abspath(target) == os.path.abspath(self.checkpoint):
            print(f"Already using {target}.")
            return False
        print(f"Switching stage-2 checkpoint -> {target}")
        try:
            state = read_state_dict(target)
        except Exception as exc:
            print(f"Could not read {target}: {exc}")
            return False

        # Re-derive eos/fim/region_cond for *this* checkpoint (the config is
        # reset to its configured values first, see `_align_special_layout`).
        self._align_special_layout(state, source=target)
        net = CodeTransformer(self.config).to(
            device=self.device, dtype=self._module_dtype())
        print(net.describe())
        n_loaded, n_ckpt, n_bad, n_missing = load_weights(net, state)
        print(f"Loaded {n_loaded}/{n_ckpt} checkpoint tensor(s)"
              + (f"; {n_bad} skipped, {n_missing} missing"
                 if n_bad or n_missing else ""))
        net.eval()

        old, self.net = self.net, net
        self.checkpoint = target
        del old
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
        # Codes/audio in hand came from the previous weights: they are not
        # "the last output of the current model" any more, so drop them rather
        # than letting `w`/`a`/`c` silently serve stale results.
        self.last_codes = self.last_sid = None
        self.last_audio = self.last_f0 = None
        self._base_hints = None
        # A new checkpoint may have been trained with a different BERT encoder
        # (the id is part of its contract), so cached word states no longer
        # mean the same thing.
        self._word_cache = {}
        self.clear_fill()
        self.frames = self._clamp_frames(self.frames)
        print("Last codes/audio cleared (they came from the old checkpoint).")
        return True

    def list_checkpoints(self) -> None:
        """Available stage-2 checkpoints under ``checkpoints_stage2/``."""
        list_checkpoints(self.checkpoint)


    # ------------------------------------------------------------------ #
    # Text / sampling
    # ------------------------------------------------------------------ #
    def set_text(self, text: str) -> bool:
        """Phonemize ``text`` and make it the current conditioning."""
        text = str(text or "").strip()
        if not text:
            print("No text given.")
            return False
        fe = self._extractor()
        try:
            phones, ids, word_index, word_texts = phonemize_with_arpabet(fe, text)
        except Exception as exc:
            print(f"Phonemization failed: {exc}")
            return False
        ids = [int(i) for i in ids]
        if not ids:
            print(f"{text!r} produced no phonemes; text left unchanged.")
            return False
        self.text, self.phone_ids = text, ids
        self.word_index, self.word_texts = list(word_index), list(word_texts)
        # A new sentence has no `.cond` (and no word timings), so the item's
        # captured levels must not leak into it.
        self._base_hints = None
        print(f"text: {text}")
        print(f"phones: {phones!r}  ({len(ids)} ids, {len(word_texts)} words)")
        return True

    def _clamp_frames(self, frames: int) -> int:
        """Clamp to ``max_positions - K + 1`` (the delayed sequence length)."""
        if self.net is not None:
            K = self.net.codebook_num
        else:
            K = (int(self.config.codec.codebook_num)
                 + int(self.config.pitch_codec.codebook_num))
        limit = max(1, int(self.config.transformer.get("max_positions", 4096))
                    - K + 1)
        frames = max(1, int(frames))
        if frames > limit:
            print(f"(frames clamped to {limit}: max_positions - K + 1)")
            return limit
        return frames

    # ------------------------------------------------------------------ #
    # Hint controls (v0.8+ checkpoints, M5)
    # ------------------------------------------------------------------ #
    def _hint_widths(self):
        """``(n_intensity, n_register, n_pitch, n_slot)`` level counts."""
        net = self.net
        return (int(getattr(net, "n_intensity_levels", 0) or 0),
                int(getattr(net, "n_register_levels", 0) or 0),
                int(getattr(net, "n_pitch_levels", 0) or 0),
                int(getattr(net, "n_slot_buckets", 0) or 0))

    def _null_base_hints(self, n_phone: int) -> dict:
        """The trained null state as a 1-D (B=1 squeezed) dict."""
        batch = self.net.null_hints(1, n_phone, dtype=torch.long)
        return {k: (v[0].detach().cpu() if torch.is_tensor(v) else v)
                for k, v in batch.items()}

    @staticmethod
    def _word_slots(text: str) -> list:
        """One entry per word of ``text`` (``None`` for a ``{...}`` group).

        The word list the *phonemizer* uses, not ``text.split()``: a brace
        group is one word with no text to embed, so keeping the ``None``
        placeholder means later words keep their index instead of shifting.
        """
        slots = []
        for kind, chunk in split_arpabet_braces(text):
            if kind == "arpabet":
                slots.append(None)
            else:
                slots.extend(chunk.split())
        return slots

    def _word_bert_model(self) -> str:
        """The encoder id this checkpoint was trained with, when it recorded it."""
        return (getattr(self.net, "_word_bert_model", None)
                or "bert-base-uncased")

    def _word_bert_vectors(self):
        """``(vecs, row_valid)`` for the word-semantic source, or ``(None, None)``.

        ``hint_word_mode`` decides the source: ``text`` embeds the current
        sentence's words, ``hint_word_text`` overrides it (the swap test), and
        ``cond`` returns nothing here because the val item's own vectors are
        already in ``_base_hints``.
        """
        if self.hint_word_text is not None:
            src = str(self.hint_word_text)
        elif self.hint_word_mode == "text":
            src = str(self.text or "")
        else:
            return None, None
        if not src.strip():
            return None, None
        key = src
        if key not in self._word_cache:
            try:
                words = (list(self.word_texts) if self.hint_word_text is None
                         else self._word_slots(src))
                words = [_bert_word(w) for w in words]
                vec = self._extractor().hint_word_bert(
                    words, model_name=self._word_bert_model(), device="cpu")
            except Exception as exc:
                print(f"  word hint unavailable: {exc}")
                vec, words = None, []
            valid = torch.tensor([bool(str(w or "").strip()) for w in words],
                                 dtype=torch.bool)
            self._word_cache[key] = ((vec, valid) if vec is not None
                                     else (None, None))
        return self._word_cache[key]

    def _word_vectors_for(self, n_phone: int):
        """Per-phone ``(hint_word, hint_word_valid)`` for the current text.

        A word's vector is repeated over its phones; a word with no vector
        (``{...}`` group, a shorter swap sentence, a BERT failure) goes through
        the model's trained ``word_null`` via the mask instead of ``proj(0)``.
        """
        vecs, row_valid = self._word_bert_vectors()
        if vecs is None:
            return None, None
        if self.word_index is None or len(self.word_index) != int(n_phone):
            print("  word hints: no word-aligned phone sequence for this text "
                  "(set it with `t`/`y`); using word_null")
            return None, None
        want = int(getattr(self.net, "hint_word_dim", vecs.shape[1]))
        if int(vecs.shape[1]) != want:
            print(f"  word hints: encoder gives {int(vecs.shape[1])} dims but "
                  f"the model wants {want}; using word_null")
            return None, None
        nrow = int(vecs.shape[0])
        if nrow == 0:
            return None, None
        idx = torch.tensor([int(i) for i in self.word_index], dtype=torch.long)
        safe = idx.clamp(0, nrow - 1)
        valid = (idx < nrow) & row_valid[safe]
        out = torch.zeros((int(n_phone), want), dtype=torch.float32)
        out[valid] = vecs[idx[valid]].to(torch.float32)
        return out, valid

    @staticmethod
    def _shift_levels(levels, delta: int, n_levels: int):
        """Shift the *present* per-phone buckets; fill uniformly when all absent.

        Absent (0) stays absent when the base has any present bucket, so a
        val item's unvoiced/undocumented words keep their state.  On arbitrary
        text (no per-word hints at all) a non-zero shift sets every phone to
        the middle bucket + ``delta`` so the control still does something.
        """
        out = levels.clone().long()
        present = (out >= 1) & (out <= int(n_levels))
        if bool(present.any()):
            out[present] = (out[present] + int(delta)).clamp(1, int(n_levels))
        else:
            mid = int(n_levels) // 2 + 1
            out[:] = max(1, min(int(n_levels), mid + int(delta)))
        return out

    def _hints_for(self, n_phone: int):
        """Hints argument for ``generate``/``infill`` with the controls applied."""
        if not getattr(self.net, "hints", False):
            return None
        if self.hint_mode == "omit":
            return "omit"
        if self.hint_mode == "cond" and self._base_hints is not None:
            base = {k: (v.clone() if torch.is_tensor(v) else v)
                    for k, v in self._base_hints.items()}
        else:
            base = self._null_base_hints(n_phone)

        n_int, n_reg, n_pitch, n_slot = self._hint_widths()
        anchors = getattr(self.net, "hint_pitch_mode", "table") == "anchors"
        A = int(getattr(self.net, "hint_pitch_anchors", 3))

        def resize(v, tail=()):
            out = torch.zeros((n_phone,) + tuple(tail),
                              dtype=(v.dtype if v is not None else torch.float32))
            if v is not None and v.numel():
                m = min(n_phone, int(v.shape[0]))
                out[:m] = v[:m].to(out.dtype)
            return out

        # Per-phone arrays must match this generation's phone sequence.
        base["hint_pitch"] = resize(base.get("hint_pitch"))
        base["hint_slot"] = resize(base.get("hint_slot"))
        if anchors:
            base["hint_pitch_val"] = resize(base.get("hint_pitch_val"), (A,))
            base["hint_pitch_pos"] = resize(base.get("hint_pitch_pos"), (A,))
            base["hint_pitch_n"] = resize(base.get("hint_pitch_n"))
            base["hint_pitch_state"] = resize(base.get("hint_pitch_state"))
            base["hint_phone_frac"] = resize(base.get("hint_phone_frac"))
        if getattr(self.net, "hint_word_sem", False):
            wdim = int(self.net.hint_word_dim)
            if self.hint_word_mode == "null":
                base["hint_word"] = torch.zeros((n_phone, wdim),
                                                dtype=torch.float32)
                base["hint_word_valid"] = torch.zeros(n_phone, dtype=torch.bool)
            elif (self.hint_word_mode == "text"
                  or self.hint_word_text is not None):
                vec, valid = self._word_vectors_for(n_phone)
                if vec is None:
                    base["hint_word"] = torch.zeros((n_phone, wdim),
                                                    dtype=torch.float32)
                    base["hint_word_valid"] = torch.zeros(
                        n_phone, dtype=torch.bool)
                else:
                    base["hint_word"] = vec
                    base["hint_word_valid"] = valid
            else:
                # `cond`: the val item's own BERT states, resized like every
                # other per-phone array.  They are usable only when the phone
                # sequence is unchanged; a shorter/different text falls back to
                # the trained null state rather than a misaligned word.
                hw = base.get("hint_word")
                if hw is not None and int(hw.shape[-1]) != wdim:
                    hw = None
                base["hint_word"] = resize(hw, (wdim,))
                v = base.get("hint_word_valid")
                if v is None:
                    base["hint_word_valid"] = torch.zeros(n_phone,
                                                         dtype=torch.bool)
                else:
                    keep = torch.zeros(n_phone, dtype=torch.bool)
                    m = min(n_phone, int(v.shape[0]))
                    if m == n_phone and int(v.numel()) == n_phone:
                        keep[:m] = v[:m].bool()
                    base["hint_word_valid"] = keep
        if base.get("hint_int") is None:
            base["hint_int"] = torch.zeros((), dtype=torch.long)
        if base.get("hint_reg") is None:
            base["hint_reg"] = torch.zeros((), dtype=torch.long)

        # Absolute level overrides.
        if self.hint_intensity is not None and n_int:
            base["hint_int"] = torch.tensor(
                max(0, min(n_int, int(self.hint_intensity))), dtype=torch.long)
        if self.hint_register is not None and n_reg:
            base["hint_reg"] = torch.tensor(
                max(0, min(n_reg, int(self.hint_register))), dtype=torch.long)

        # Per-word shifts: anchors are continuous semitones, the v1 table is
        # bucketed at 2 st/bin.
        if anchors:
            if float(self.hint_pitch_st):
                present = base["hint_pitch_state"] == 1
                if bool(present.any()):
                    num = base["hint_pitch_n"].clamp(max=A)
                    mask = (torch.arange(A).view(1, A)
                            < num.unsqueeze(-1)).to(torch.float32)
                    base["hint_pitch_val"] = (base["hint_pitch_val"]
                                              + mask * float(self.hint_pitch_st))
                else:
                    # Arbitrary text carries no `.cond`: synthesise one anchor
                    # so the control still does something (the v1 path fills
                    # the middle bucket for the same reason).
                    base["hint_pitch_state"] = torch.ones(n_phone, dtype=torch.long)
                    base["hint_pitch_n"] = torch.ones(n_phone, dtype=torch.long)
                    val = torch.zeros(n_phone, A)
                    val[:, 0] = float(self.hint_pitch_st)
                    base["hint_pitch_val"] = val
                    pos = torch.zeros(n_phone, A)
                    pos[:, 0] = 0.5
                    base["hint_pitch_pos"] = pos
                    base["hint_phone_frac"] = torch.full((n_phone,), 0.5)
        else:
            dp = int(round(float(self.hint_pitch_st) / 2.0))
            if dp and n_pitch:
                base["hint_pitch"] = self._shift_levels(
                    base["hint_pitch"], dp, n_pitch)
        if self.hint_slot_shift and n_slot:
            base["hint_slot"] = self._shift_levels(
                base["hint_slot"], int(self.hint_slot_shift), n_slot)

        return {k: (v.unsqueeze(0) if torch.is_tensor(v) else v)
                for k, v in base.items()}

    def hint_state(self) -> str:
        """One-line description of the current hint controls."""
        if self.net is not None and not getattr(self.net, "hints", False):
            return "n/a (checkpoint has no hint tables)"
        n_int, n_reg, n_pitch, n_slot = self._hint_widths()
        inten = ("base" if self.hint_intensity is None
                 else str(self.hint_intensity))
        reg = ("base" if self.hint_register is None else str(self.hint_register))
        if getattr(self.net, "hint_word_sem", False):
            word = self.hint_word_mode
            if self.hint_word_text is not None:
                word = f"text({self.hint_word_text!r})"
        else:
            word = "n/a"
        return (f"base={self.hint_mode}, intensity={inten}/{n_int}, "
                f"register={reg}/{n_reg}, pitch={self.hint_pitch_st:+.1f}st, "
                f"slot={self.hint_slot_shift:+d}, word={word}")

    def _hint_arg(self):
        """Kept for call sites that only need the base mode (no controls)."""
        return self._hints_for(len(self.phone_ids))

    def set_hint_intensity(self, raw) -> None:
        text = str(raw or "").strip().lower()
        if not text:
            return
        if text in ("base", "item", "none", "off"):
            self.hint_intensity = None
        else:
            n_int = self._hint_widths()[0]
            try:
                value = int(text, 0)
            except ValueError:
                print(f"not a level: {raw!r}")
                return
            self.hint_intensity = max(0, min(n_int, value))
        print(f"intensity: {self.hint_state()}")

    def set_hint_register(self, raw) -> None:
        text = str(raw or "").strip().lower()
        if not text:
            return
        if text in ("base", "item", "none", "off"):
            self.hint_register = None
        else:
            n_reg = self._hint_widths()[1]
            try:
                value = int(text, 0)
            except ValueError:
                print(f"not a level: {raw!r}")
                return
            self.hint_register = max(0, min(n_reg, value))
        print(f"register: {self.hint_state()}")

    def set_hint_pitch(self, raw) -> None:
        text = str(raw or "").strip()
        if not text:
            return
        try:
            self.hint_pitch_st = float(text)
        except ValueError:
            print(f"not a number: {raw!r}")
            return
        print(f"word pitch shift: {self.hint_pitch_st:+.1f} st "
              f"(= {int(round(self.hint_pitch_st / 2.0)):+d} bucket(s))")

    def set_hint_slot(self, raw) -> None:
        text = str(raw or "").strip()
        if not text:
            return
        value = _parse_int(text)
        if value is None:
            return
        self.hint_slot_shift = int(value)
        print(f"word slot shift: {self.hint_slot_shift:+d} "
              f"(x{2.0 ** self.hint_slot_shift:.2f} duration)")

    def set_hint_word(self, raw, announce: bool = True) -> None:
        text = str(raw or "").strip()
        if not text:
            return
        low = text.lower()
        if low.startswith("swap:"):
            # One-token form of the swap test: `d swap:<a sentence>`.
            self.hint_word_mode = "text"
            self.hint_word_text = text.split(":", 1)[1].strip() or None
        elif low in ("cond", "item"):
            self.hint_word_mode, self.hint_word_text = "cond", None
        elif low in ("text", "current"):
            self.hint_word_mode, self.hint_word_text = "text", None
        elif low in ("null", "off", "none"):
            self.hint_word_mode, self.hint_word_text = "null", None
        elif low in ("clear", "-"):
            self.hint_word_mode, self.hint_word_text = "cond", None
        else:
            # A literal sentence: inject *its* per-word BERT states instead of
            # the current text's (the word-semantics swap test).  Word i of the
            # current sentence gets word i of this one; words it does not cover
            # go through word_null.
            self.hint_word_mode, self.hint_word_text = "text", text
        if (announce and self.hint_word_mode == "text"
                and self.word_index is None):
            print("  note: no word-aligned phone sequence yet (press t/y first); "
                  "the channel stays word_null until then.")
        if announce:
            print(f"word semantics: {self.hint_state()}")

    def generate(self, frames: Optional[int] = None) -> bool:
        """Sample codes for the current text and decode them to audio."""
        if not self.phone_ids:
            print("No text yet -- press t (text) or y (say) first.")
            return False
        frames = self._clamp_frames(self.frames if frames is None else frames)
        phone = torch.tensor(self.phone_ids, dtype=torch.long,
                             device=self.device).unsqueeze(0)
        plen = torch.tensor([len(self.phone_ids)], dtype=torch.long,
                            device=self.device)
        sid = torch.tensor([int(self.sid)], dtype=torch.long,
                           device=self.device)
        t0 = time.perf_counter()
        with self._autocast():
            codes, lens = self.net.generate(
                phone, plen, sid, max_frames=frames,
                temperature=self.temperature, top_k=self.top_k,
                return_lengths=True, hints=self._hint_arg())

        dt = time.perf_counter() - t0
        codes = codes[0]
        n = int(codes.shape[-1])
        print(f"generated {n} code frames = {n / self.frame_rate:.2f} s "
              f"@{self.frame_rate:g} Hz in {dt * 1e3:.0f} ms "
              f"(sid {self.sid}, temp {self.temperature:g}, "
              f"top_k {self.top_k})")
        if self.net.eos_token:
            print(f"  EOS length {int(lens[0])} of at most {frames} frames")
        self.last_codes, self.last_sid = codes, int(self.sid)
        self.last_audio = self.last_f0 = None
        self.clear_fill()
        self._report_codes(codes)
        self._decode_audio(codes, self.sid)
        return True

    def _report_codes(self, codes) -> None:
        """Per-codebook usage + a cheap 'is it looping?' number."""
        n_spec = self.net.spec_codebook_num
        usage = " ".join(
            f"{int(torch.unique(codes[k]).numel())}/{int(self.net.vocab_sizes[k])}"
            for k in range(codes.shape[0]))
        print(f"  unique codes/book (spec 0-{n_spec - 1}, pitch {n_spec}-"
              f"{codes.shape[0] - 1}): {usage}")
        if codes.shape[-1] > 1:
            same = (codes[:, 1:] == codes[:, :-1]).float().mean().item()
            print(f"  identical to previous frame: {same * 100:.1f} % "
                  f"(very high == the decoder is looping)")

    def _decode_audio(self, codes, sid) -> bool:
        """(K, n) codes -> waveform + f0 through the frozen stage-1 Generator."""
        net_g = self._generator()
        if net_g is None:
            return False
        n_spec = self.net.spec_codebook_num
        codes = codes.to(self.device)
        gen_sid = int(self.stage1_sid_map.get(int(sid), int(sid)))
        sid_t = torch.tensor([gen_sid], dtype=torch.long, device=self.device)
        try:
            # `infer_from_codes` builds a graph unless we say otherwise (it is
            # also used from training-time sampling), and nothing here trains.
            with torch.no_grad():
                audio, f0 = net_g.infer_from_codes(
                    codes[:n_spec].unsqueeze(1), codes[n_spec:].unsqueeze(1),
                    sid_t, return_pitch=True)
        except Exception as exc:
            print(f"  stage-1 decode failed: {exc}")
            return False
        self.last_audio = audio[0].squeeze(0).float().cpu()
        self.last_f0 = f0[0].float().cpu()
        sr = int(self.config.data.sampling_rate)
        voiced = self.last_f0 > 0
        frac = float(voiced.float().mean()) if voiced.numel() else 0.0
        med = float(self.last_f0[voiced].median()) if bool(voiced.any()) else 0.0
        print(f"  audio: {self.last_audio.numel() / sr:.2f} s @ {sr} Hz, "
              f"voiced {frac * 100:.0f} %, median f0 {med:.0f} Hz")
        if self.auto_play:
            self.player.play(self.last_audio, sr)
        return True

    def play(self) -> bool:
        """Play the last decoded audio (``b``)."""
        if self.last_audio is None:
            print("Nothing decoded yet -- press g (generate) or a (audio).")
            return False
        return self.player.play(self.last_audio,
                                int(self.config.data.sampling_rate))

    # ------------------------------------------------------------------ #
    # Fill-in-the-middle
    # ------------------------------------------------------------------ #
    def fill_middle(self, span: Optional[int] = None,
                    start: Optional[int] = None) -> bool:
        """Re-generate a span of the *last generation* (``n``, ``fill``).

        This is the "regenerate the bad bit" direction of FIM: throw away
        ``span`` code frames starting at ``start`` of the last sampled codes and
        sample them again with :meth:`CodeTransformer.infill`, conditioned on
        the untouched prefix *and* suffix of the same clip.  Nothing else
        changes, so the clip keeps its length and its speaker.

        With ``span``/``start`` left at ``None`` the draw mirrors
        ``dataset_codes.CodeDataset._sample_fim`` (training conditions the model
        on exactly that span distribution): ``span`` in
        ``[fim_min_frames, fim_mid_frac * n]``, ``start`` uniform -- except
        that a non-empty prefix *and* suffix are preferred, because a "middle"
        needs two sides to be interesting.

        The result goes to :attr:`fill_codes` and **not** to
        :attr:`last_codes`: the generation stays pristine, so a second ``n``
        fills a different span of the same clean sample instead of re-filling
        whatever the previous fill produced (which would compound the
        damage).  ``a``/``b``/``c``/``w``/``x`` look at
        :attr:`current_codes`, i.e. the filled clip while a fill exists, so you
        can still listen to and save it; ``g``/``v``/``s`` drop it.

        Needs a FIM checkpoint (``net.fim``, i.e. the SUF/MID classes were
        trained) and something already sampled to work on.
        """
        if self.net is None:
            print("Model not loaded.")
            return False
        if not self.net.fim:
            print("This checkpoint has no FIM support (fim=False): its SUF/MID "
                  "sentinel classes were never trained, so an infill would be "
                  "noise.")
            print("  Only checkpoints_stage2/transformer_fim.ckpt can do this "
                  "-- press s to switch to it.")
            return False
        if self.last_codes is None or self.last_sid is None:
            print("Nothing sampled yet -- press g (generate) or v (val item) "
                  "first; n fills a span of what those produced.")
            return False
        if not self.phone_ids:
            print("No phonemes for the current codes -- press t (text) first.")
            return False

        n = int(self.last_codes.shape[-1])
        if n < 2:
            print(f"The last sample is only {n} frame(s) long, nothing to fill.")
            return False

        # Span length: the training distribution, capped so the clip keeps at
        # least one known frame (`infill` splices prefix + middle + suffix).
        tcfg = self.config.transformer
        min_mid = max(1, int(tcfg.get("fim_min_frames", 1) or 1))
        mid_frac = float(tcfg.get("fim_mid_frac", 0.5) or 0.5)
        lo = min(min_mid, n - 1)
        hi = max(lo, min(n - 1, int(round(mid_frac * n))))
        if span is None:
            span = random.randint(lo, hi)
        span = max(1, min(int(span), n - 1))
        if start is None:
            # Prefer both sides non-empty (a real "middle"); any split is
            # allowed when the span leaves room for only one of them.
            lo_s, hi_s = (1, n - span - 1) if n - span >= 2 else (0, n - span)
            start = random.randint(lo_s, hi_s)
        start = max(0, min(int(start), n - span))

        print(f"fill: frames {start}..{start + span - 1} of {n} "
              f"({span / self.frame_rate:.2f} s at "
              f"{start / self.frame_rate:.2f} s .. "
              f"{(start + span) / self.frame_rate:.2f} s of "
              f"{n / self.frame_rate:.2f} s)")
        print(f"  # = re-generated, . = kept "
              f"(kept prefix {start}, suffix {n - start - span} frames)")
        for line in ascii_fill_map(n, start, span):
            print("  " + line)

        codes = self.last_codes.to(self.device)
        prefix = codes[:, :start].unsqueeze(0)        # (1, K, start)
        suffix = codes[:, start + span:].unsqueeze(0)  # (1, K, n-start-span)
        original = codes[:, start:start + span]
        phone = torch.tensor([list(self.phone_ids)], dtype=torch.long,
                             device=self.device)
        plen = torch.tensor([len(self.phone_ids)], dtype=torch.long,
                            device=self.device)
        sid_t = torch.tensor([int(self.last_sid)], dtype=torch.long,
                             device=self.device)
        t0 = time.perf_counter()
        try:
            with self._autocast():
                middle = self.net.infill(
                    phone, plen, sid_t, prefix, suffix, span,
                    temperature=self.temperature, top_k=self.top_k,
                    hints=self._hint_arg(),
                    best_of=self.infill_best_of)

        except Exception as exc:
            print(f"  infill failed: {exc}")
            return False
        dt = time.perf_counter() - t0
        middle = middle[0]

        self.fill_codes = torch.cat(
            [codes[:, :start], middle, codes[:, start + span:]], dim=-1)
        self.last_fill = (start, span)
        self.last_audio = self.last_f0 = None
        print(f"  sampled {span} middle frames in {dt * 1e3:.0f} ms "
              f"(sid {self.last_sid}, temp {self.temperature:g}, "
              f"top_k {self.top_k})")
        if (getattr(self.net, "fim_lookahead", 0) > 0
                and suffix.shape[-1] > 0):
            try:
                seam = self.net.infill_seam_score(
                    phone, plen, sid_t, prefix, middle.unsqueeze(0), suffix,
                    hints=self._hint_arg())
                print(f"  seam CE {float(seam[0]):.3f} nats -- teacher-forced "
                      f"CE of the true suffix start given this span "
                      f"(lower = joins up better)")
            except Exception as exc:
                print(f"  seam score failed: {exc}")
        self._report_codes(self.fill_codes)
        # How close the re-generated span is to the codes it replaced.  Both
        # sides are in-distribution, so this is the "did it actually use the
        # context" number, not a sampling-quality score.
        print("  re-generated middle vs the span it replaced:")
        self._report_match(original, middle)
        self._decode_audio(self.fill_codes, self.last_sid)
        print("  (a/b/c/w/x now act on the filled clip, g/v/s drop it; the "
              "next n fills another span of the same generation)")
        return True

    # ------------------------------------------------------------------ #
    # Validation-item comparison
    # ------------------------------------------------------------------ #
    def _val_hint_kwargs(self) -> dict:
        """CodeDataset kwargs so val batches carry their ``.cond`` hints.

        Training-only dropout/null-augmentation is deliberately *not* passed:
        a val item must show the hints it actually has.
        """
        tcfg = self.config.transformer
        if not bool(tcfg.get("hints", False)):
            return {}
        return {
            "hints": True,
            "hints_json": tcfg.get("hints_json"),
            "hint_pitch_edges": tcfg.get("hint_pitch_edges", None),
            "n_slot_buckets": int(tcfg.get("n_slot_buckets", 8)),
        }

    def _val_dataset(self):
        if self._val_ds is None:
            from dataset_codes import CodeDataset

            path = str(self.config.train.val_filelist)
            if not os.path.isfile(path):
                print(f"Val filelist not found: {path}")
                return None
            self._val_ds = CodeDataset(
                path, is_train=False, max_frames=None,
                max_phone_length=self.config.train.get("max_phone_length"),
                speaker_map=self.config.train.get("speaker_map"),
                **self._val_hint_kwargs())
            print(f"{len(self._val_ds)} validation item(s) from {path}")
        return self._val_ds

    def val_item(self, index: Optional[int] = None) -> bool:
        """Free-run from a val item's phonemes and score it against its codes."""
        ds = self._val_dataset()
        if ds is None or len(ds) == 0:
            return False
        from dataset_codes import CodeCollator

        if index is None:
            index = self._val_cursor
        index = int(index) % len(ds)
        self._val_cursor = (index + 1) % len(ds)

        batch = CodeCollator(config=self.config)([ds[index]])
        batch = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        self.set_phone_ids(
            [int(i) for i in batch["phone_ids"][0][
                :int(batch["phone_lengths"][0])]])
        self.sid = int(batch["sid"][0])
        # Keep this item's own levels so `cond` (and the sliders) can use them.
        self._base_hints = self._capture_base_hints(batch)
        code_len = int(batch["code_lengths"][0])
        frames = self._clamp_frames(min(self.frames, code_len))

        print(f"val item {index}/{len(ds) - 1}: sid {self.sid}, {code_len} "
              f"frames = {code_len / self.frame_rate:.2f} s, "
              f"{len(self.phone_ids)} phone ids")
        print(f"  phones: {self._phones_text()!r}")

        with self._autocast():
            codes, lens = self.net.generate(
                batch["phone_ids"], batch["phone_lengths"], batch["sid"],
                max_frames=frames, temperature=self.temperature,
                top_k=self.top_k, return_lengths=True,
                hints=self._hints_for(int(batch["phone_ids"].shape[1])))

        pred = codes[0]
        n = int(pred.shape[-1])
        gt = torch.cat([batch["spec_codes"][0], batch["pitch_codes"][0]],
                       dim=0)[:, :n]
        self.last_codes, self.last_sid = pred, self.sid
        self.last_audio = self.last_f0 = None
        self.clear_fill()
        print(f"generated {n} of at most {frames} frames"
              + (f" (EOS length {int(lens[0])})" if self.net.eos_token else ""))
        self._report_codes(pred)
        self._report_match(gt, pred)
        self._decode_audio(pred, self.sid)
        return True

    def _capture_base_hints(self, batch) -> Optional[dict]:
        """The val item's hint levels as a 1-D dict (``None`` if it has none)."""
        if not getattr(self.net, "hints", False) or "hint_int" not in batch:
            return None
        n_phone = int(batch["phone_lengths"][0])
        keep = ("hint_int", "hint_reg")
        out = {}
        for key in keep:
            v = batch.get(key)
            if v is not None:
                out[key] = v[0].detach().cpu()
        for key in ("hint_pitch", "hint_slot"):
            v = batch.get(key)
            if v is not None:
                out[key] = v[0, :n_phone].detach().cpu()
        for key in ("hint_pitch_val", "hint_pitch_pos", "hint_pitch_n",
                    "hint_pitch_state", "hint_phone_frac",
                    # V2.1 word state (per phone, trailing 768 dims).
                    "hint_word", "hint_word_valid"):
            v = batch.get(key)
            if v is not None:
                out[key] = v[0, :n_phone].detach().cpu()
        return out or None

    def set_phone_ids(self, ids) -> None:
        """Set the conditioning *without* phonemizing (val items carry ids)."""
        self.phone_ids = [int(i) for i in ids]
        self.text = ""
        # No source tokens, hence no word-aligned BERT state: `cond` uses the
        # item's own `.cond` vectors, `text`/a swap sentence fall back to null.
        self.word_index, self.word_texts = None, []

    def _phones_text(self) -> str:
        """ARPAbet string for the current ids (never raises)."""
        try:
            return self._extractor().tokenizer_g2p.decode(self.phone_ids)
        except Exception as exc:
            return f"<undecodable: {exc}>"

    def _report_match(self, gt, pred) -> None:
        """Free-running code agreement per group, as 2_training.py logs it."""
        n_spec = self.net.spec_codebook_num
        for name, sl, pad in (
                ("spec", slice(0, n_spec), int(self.net.vocab_sizes[0])),
                ("pitch", slice(n_spec, None), int(self.net.vocab_sizes[-1]))):
            g, p = gt[sl], pred[sl]
            valid = g != pad
            if not bool(valid.any()):
                continue
            frac = float((g == p).float()[valid].mean())
            print(f"  {name} code match vs target: {frac * 100:.1f} % "
                  f"({int(valid.sum())} target codes)")

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    def save_audio(self, path: Optional[str] = None) -> bool:
        """Write the last decoded waveform to a wav (stage-1 sample rate)."""
        if self.last_audio is None:
            print("Nothing decoded yet -- press g (generate), v or n (fill).")
            return False
        path = (path or "").strip()
        if not path:
            os.makedirs(self.out_dir, exist_ok=True)
            self._writes += 1
            tag = "fill_" if self.fill_codes is not None else ""
            path = os.path.join(
                self.out_dir,
                f"{tag}say_sid{self.last_sid}_"
                f"{int(self.current_codes.shape[-1])}f_"
                f"{time.strftime('%H%M%S')}_{self._writes}.wav")
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        import numpy as np
        import soundfile as sf

        sr = int(self.config.data.sampling_rate)
        wav = np.clip(self.last_audio.numpy(), -1.0, 1.0).astype(np.float32)
        sf.write(path, wav, sr)
        print(f"wrote {path} ({wav.shape[0] / sr:.2f} s @ {sr} Hz)")
        return True

    def dump_codes(self, path: Optional[str] = None) -> bool:
        """``torch.save`` the last ``(K, n)`` code tensor (``x``)."""
        codes = self.current_codes
        if codes is None:
            print("Nothing sampled yet -- press g (generate), v or n (fill).")
            return False
        path = (path or "").strip()
        if not path:
            os.makedirs(self.out_dir, exist_ok=True)
            tag = "fill_codes" if self.fill_codes is not None else "codes"
            path = os.path.join(self.out_dir, f"{tag}_sid{self.last_sid}_"
                                              f"{time.strftime('%H%M%S')}.pt")
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(codes.cpu(), path)
        print(f"wrote {path}  (K, n) = {tuple(codes.shape)}")
        return True

    def list_speakers(self) -> None:
        """Speaker ids present in the val split, commonest first."""
        ds = self._val_dataset()
        if ds is None:
            return
        counts: dict[int, int] = {}
        for sid in ds.speaker_ids():
            counts[sid] = counts.get(sid, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])
        print(f"{len(ranked)} speaker(s) in {self.config.train.val_filelist} "
              f"({len(ds)} items):")
        for sid, n in ranked[:40]:
            print(f"  sid {sid:>5}   {n:>5} item(s)")
        if len(ranked) > 40:
            print(f"  ... {len(ranked) - 40} more")

    def print_status(self) -> None:
        if self.net is not None:
            layout = (f"eos_token={self.net.eos_token}, fim={self.net.fim}, "
                      f"region_cond={self.net.region_cond}, "
                      f"length_cond={self.net.length_cond}")
        else:
            layout = "model not loaded"
        if self._audio_disabled:
            audio = "off (--no_audio)"
        elif not self.generator_checkpoint:
            audio = "off (no generator_checkpoint in the config)"
        elif self._net_g_failed:
            audio = "off (load failed)"
        else:
            audio = f"on -> {self.generator_checkpoint}"
        print(f"checkpoint : {self.checkpoint}")
        print(f"device     : {self.device}   precision: {self.precision}   "
              f"input: {self.reader.describe()}")
        print(f"layout     : {layout}")
        print(f"text       : {self.text or '<none>'}"
              + (f"  ({len(self.phone_ids)} phone ids)" if self.phone_ids
                 else ""))
        print(f"sid        : {self.sid}")
        print(f"frames     : {self.frames} "
              f"(~{self.frames / self.frame_rate:.2f} s at "
              f"{self.frame_rate:g} Hz)")
        print(f"temperature: {self.temperature:g}   top_k: {self.top_k}")
        print(f"hints      : {self.hint_state()}")
        print(f"audio      : {audio}")
        print(f"playback   : {self.player.describe()}"
              f"   auto-play: {'on' if self.auto_play else 'off (press b)'}")
        if self.last_codes is not None:
            sr = int(self.config.data.sampling_rate)
            extra = (f", audio {self.last_audio.numel() / sr:.2f} s"
                     if self.last_audio is not None else ", not decoded")
            print(f"last codes : {tuple(self.last_codes.shape)}, "
                  f"sid {self.last_sid}{extra}")
        if self.last_fill is not None and self.fill_codes is not None:
            start, span = self.last_fill
            n = int(self.fill_codes.shape[-1])
            print(f"last fill  : frames {start}..{start + span - 1} of {n} "
                  f"({span / self.frame_rate:.2f} s at "
                  f"{start / self.frame_rate:.2f} s) re-generated -- "
                  f"a/b/c/w/x use this clip, g/v/s drop it")

    def print_help(self) -> None:
        print(HELP_TEXT)

    def report_memory(self) -> None:
        """``z``: GPU/CPU footprint of the loaded models and the last result."""
        def mb(nbytes: int) -> str:
            return f"{nbytes / 1024 ** 2:.1f} MB"

        def gib(nbytes: int) -> str:
            return f"{nbytes / 1024 ** 3:.2f} GiB"

        cuda = str(self.device).startswith("cuda")
        print(f"memory @ {self.device} (sampling precision {self.precision})")
        if cuda and torch.cuda.is_available():
            dev = torch.device(self.device)
            alloc = torch.cuda.memory_allocated(dev)
            reserved = torch.cuda.memory_reserved(dev)
            peak_alloc = torch.cuda.max_memory_allocated(dev)
            peak_res = torch.cuda.max_memory_reserved(dev)
            free, total = torch.cuda.mem_get_info(dev)
            print(f"  gpu      : {gib(alloc)} allocated / {gib(reserved)} "
                  f"reserved of {gib(total)} total, {gib(free)} free")
            print(f"  peak     : {gib(peak_alloc)} allocated / "
                  f"{gib(peak_res)} reserved (since process start)")
            print(f"  cache    : {gib(reserved - alloc)} in the allocator's "
                  f"free blocks (reused before asking the driver)")
        elif cuda:
            print("  gpu      : n/a (CUDA requested but unavailable)")
        else:
            print("  gpu      : n/a (CPU device)")

        def model_line(name, module):
            if module is None:
                return
            n = sum(p.numel() for p in module.parameters())
            b = sum(p.numel() * p.element_size() for p in module.parameters())
            print(f"  {name:<9}: {mb(b)} ({n / 1e6:.1f} M parameters, "
                  f"{next(module.parameters()).dtype})")

        model_line("transform", self.net)
        # `_net_g` is created lazily, so this line only appears once decoded.
        model_line("generator", self._net_g)
        if self.net is not None:
            top = sorted(
                ((n, t.numel() * t.element_size())
                 for n, t in self.net.named_parameters()),
                key=lambda kv: kv[1], reverse=True)[:3]
            print("  largest  : " + ", ".join(f"{n} {mb(b)}" for n, b in top))
        print("  note     : --precision casts the transformer's weights; the "
              "stage-1 vocoder stays fp32")

        codes = self.current_codes
        if codes is not None:
            print(f"  codes    : {tuple(codes.shape)} {codes.dtype} "
                  f"{mb(codes.numel() * codes.element_size())} "
                  f"on {codes.device}")
        if self.last_audio is not None:
            sr = int(self.config.data.sampling_rate)
            print(f"  audio    : {self.last_audio.numel() / sr:.2f} s, "
                  f"{mb(self.last_audio.numel() * self.last_audio.element_size())}"
                  f" on {self.last_audio.device}")
        if resource is not None:
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux reports KiB, macOS bytes.
            peak = peak * (1 if sys.platform == "darwin" else 1024)
            print(f"  cpu      : peak RSS {gib(peak)} (whole process)")


HELP_TEXT = """
Commands (single keystroke; keys marked (prompt) ask for a value afterwards)

  m  help                         p  print current settings
  l  list speakers (val split)    c  summarise the last sampled codes
  t  text        (prompt)         y  say = text + generate (prompt)
  g  generate from current text   v  val item (prompt: index, blank = next)
  i  speaker id  (prompt)         f  max code frames (prompt)
  h  temperature (prompt)         k  top-k (prompt, `none` disables)
  a  decode the last codes again  w  write the last audio to a wav (prompt)
  x  dump the last codes tensor   q  quit (^C / ^D also work)
  z  GPU/CPU memory report        b  play the last audio (space too)
  s  switch stage-2 checkpoint (prompt: run name or path; `?` lists them)
  n  fill a random span of the last sample (prompt: frames or start:frames)

  Hint controls (hint-enabled checkpoints; p shows the current state):
  e  intensity level  (prompt: 0-8, blank = keep; `base` = the item's value)
  r  register level   (prompt: 0-5, blank = keep)
  j  word pitch shift (prompt: semitones, e.g. +2 / -4; 2 st per bucket)
  u  word slot shift  (prompt: log2 duration, +1 = twice as long)
  d  word semantics   (prompt: cond | text | null | swap:<sentence>)

  `cond` uses the loaded val item's own `.cond` per-word BERT states; arbitrary
  text has no `.cond`, so it falls back to the trained word_null state.  `d
  text` embeds the current sentence's words with the corpus's BERT encoder
  (one vector per word, repeated over that word's phones), and `d <sentence>`
  (or `d swap:<sentence>`) swaps in that sentence's word vectors instead (word
  i of the current sentence gets word i of the swap; words it does not cover
  are word_null) -- the word-level swap test.  Word-pitch and slot shifts act
  on the present per-word buckets; on OOD text they fill every phone uniformly.

  Blank at a prompt keeps the current value; blank at a wav/codes prompt
  writes into the output directory with a generated name.  `s` rebuilds the
  model in place, re-deriving eos_token/fim/region_cond from the new
  checkpoint, and clears the last codes/audio (they came from the old one).
  `--play` plays every clip as soon as it is decoded (`b` replays it).
  `--precision bf16|fp16` builds the transformer in that dtype (weights *and*
  activations; fp32 is the default, matching stage 2's 32-true convention,
  while the stage-1 vocoder always decodes fp32 -- its pitch path hard-codes
  fp32).  `z` prints the GPU allocator/peak numbers and the model footprint,
  so the two precisions can be compared directly.
  `n` needs a FIM checkpoint (checkpoints_stage2/transformer_fim.ckpt) and a
  sample to work on: it drops a span of the last codes and re-generates it
  from the kept prefix + suffix, printing a `#`/`.` map of where the span sat.
  The generation itself is untouched, so `n` again fills another span of the
  same sample, and `a`/`b`/`c`/`w`/`x` look at the filled clip until the next
  `g`/`v`/`s` clears it.
"""


def _parse_fill_spec(raw):
    """Parse a fill prompt into ``(start, span)``, or ``None`` if unparsable.

    ``"40"`` -> ``(None, 40)`` -- a 40-frame span somewhere; ``"20:40"`` ->
    ``(20, 40)`` -- start pinned too (a ``20,40`` comma works as well); blank /
    ``None`` -> ``(None, None)``, i.e. both drawn at random.
    """
    if raw is None:
        return None, None
    text = str(raw).strip().replace(",", ":")
    if not text:
        return None, None
    try:
        if ":" in text:
            a, b = text.split(":", 1)
            return int(a, 0), int(b, 0)
        return None, int(text, 0)
    except ValueError:
        print(f"not a fill span: {text!r} (use `frames` or `start:frames`)")
        return None


def _parse_int(raw) -> Optional[int]:
    """``None`` for blank / unparsable input, else ``int``."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return int(text, 0)
    except ValueError:
        print(f"not an integer: {text!r}")
        return None


def run_repl(repl: "Stage2REPL") -> None:
    """The fdisk-style loop: one keystroke -> one command."""
    banner(repl)
    while True:
        try:
            key, rest = repl.reader.read_command()
        except Exception:
            traceback.print_exc()
            continue
        if key is None:  # EOF / ^C / ^D
            print()
            return
        if key == "":
            continue
        if not repl.reader.line_mode:
            print(key)  # cbreak disables ECHO, so echo the command ourselves
        if not dispatch(repl, key, rest):
            return


def banner(repl: "Stage2REPL") -> None:
    print()
    print(f"Stage-2 inference REPL ({repl.config.exp_name}), "
          f"{repl.reader.describe()} input")
    print(f"  text={repl.text or '<none>'}  sid={repl.sid}  "
          f"frames={repl.frames}  temp={repl.temperature:g}  "
          f"top_k={repl.top_k}  precision={repl.precision}  "
          f"audio={'on' if repl.audio and not repl._net_g_failed else 'off'}")
    if repl.net is not None and getattr(repl.net, "hints", False):
        print(f"  hints={repl.hint_state()}")
    print("  press m for help, q to quit.")


def dispatch(repl: "Stage2REPL", key: str, rest: str = "") -> bool:
    """Run one command. Returns ``False`` when the REPL should exit."""
    key = key.lower()
    if key == " ":  # space replays the last clip (single-key mode)
        key = "b"
    try:
        if key == "q":
            print("bye.")
            return False
        if key in ("m", "?"):
            repl.print_help()
        elif key == "p":
            repl.print_status()
        elif key == "l":
            repl.list_speakers()
        elif key == "s":
            raw = rest or repl.reader.read_line(
                f"Stage-2 checkpoint (run name or path, `?` lists) "
                f"[{repl.checkpoint}]: ")
            if raw is not None:
                repl.switch_checkpoint(raw)
        elif key == "c":
            if repl.current_codes is None:
                print("Nothing sampled yet -- press g (generate).")
            else:
                repl._report_codes(repl.current_codes)
        elif key == "t":
            text = rest or repl.reader.read_line(
                "Text ({...} = manual ARPAbet): ")
            if text is not None:
                repl.set_text(text)
        elif key == "y":
            text = rest or repl.reader.read_line("Say: ")
            if text is not None and repl.set_text(text):
                repl.generate()
        elif key == "g":
            if rest:
                repl.set_text(rest)
            repl.generate()
        elif key == "v":
            raw = rest or repl.reader.read_line(
                f"Val item index [{repl._val_cursor}] (blank = next): ")
            repl.val_item(_parse_int(raw))
        elif key == "i":
            raw = rest or repl.reader.read_line(f"Speaker id [{repl.sid}]: ")
            value = _parse_int(raw)
            if value is not None:
                repl.sid = value
                print(f"sid = {repl.sid}")
        elif key == "f":
            raw = rest or repl.reader.read_line(
                f"Max code frames [{repl.frames}]: ")
            value = _parse_int(raw)
            if value is not None:
                repl.frames = repl._clamp_frames(value)
                print(f"frames = {repl.frames} "
                      f"(~{repl.frames / repl.frame_rate:.2f} s)")
        elif key == "h":
            raw = rest or repl.reader.read_line(
                f"Temperature [{repl.temperature:g}]: ")
            if raw is not None and str(raw).strip():
                try:
                    repl.temperature = max(1e-2, float(raw))
                    print(f"temperature = {repl.temperature:g}")
                except ValueError:
                    print(f"not a number: {raw!r}")
        elif key == "k":
            raw = rest or repl.reader.read_line(
                f"Top-k [{repl.top_k}] ('none' disables): ")
            if raw is not None and str(raw).strip():
                if str(raw).strip().lower() in ("none", "off"):
                    repl.top_k = None
                else:
                    value = _parse_int(raw)
                    if value is not None:
                        repl.top_k = value if value > 0 else None
                print(f"top_k = {repl.top_k}")
        elif key == "e":
            raw = rest or repl.reader.read_line(
                "Intensity level (0-8; blank = keep, `base` = the item's "
                f"value) [{repl.hint_intensity if repl.hint_intensity is not None else 'base'}]: ")
            repl.set_hint_intensity(raw)
        elif key == "r":
            raw = rest or repl.reader.read_line(
                "Register level (0-5; blank = keep, `base` = the item's "
                f"value) [{repl.hint_register if repl.hint_register is not None else 'base'}]: ")
            repl.set_hint_register(raw)
        elif key == "j":
            raw = rest or repl.reader.read_line(
                f"Word-pitch shift in semitones [{repl.hint_pitch_st:+.1f}]: ")
            repl.set_hint_pitch(raw)
        elif key == "u":
            raw = rest or repl.reader.read_line(
                f"Word-slot shift (log2 duration) [{repl.hint_slot_shift:+d}]: ")
            repl.set_hint_slot(raw)
        elif key == "d":
            raw = rest or repl.reader.read_line(
                "Word-semantics source: cond | text | null | <a sentence> "
                f"[{repl.hint_word_mode}]: ")
            repl.set_hint_word(raw)
        elif key == "a":
            if repl.current_codes is None:
                print("Nothing sampled yet -- press g (generate).")
            else:
                repl._decode_audio(repl.current_codes, repl.last_sid)
        elif key == "b":
            repl.play()
        elif key == "z":
            repl.report_memory()
        elif key == "n":
            raw = rest or repl.reader.read_line(
                "Fill span (frames, or start:frames; blank = random): ")
            spec = _parse_fill_spec(raw)
            if spec is not None:
                repl.fill_middle(span=spec[1], start=spec[0])
        elif key == "w":
            path = rest or repl.reader.read_line(
                f"Wav path [{repl.out_dir}/...]: ")
            if path is not None:
                repl.save_audio(path)
        elif key == "x":
            path = rest or repl.reader.read_line(
                f"Codes path [{repl.out_dir}/...]: ")
            if path is not None:
                repl.dump_codes(path)
        else:
            print(f"Unknown command {key!r} -- press m for help.")
    except KeyboardInterrupt:
        print("\n(interrupted)")
    except Exception:
        traceback.print_exc()
    return True


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def resolve_checkpoint(explicit: Optional[str], exp_name: str) -> str:
    """``--checkpoint`` or the run's own ``last.ckpt`` / ``best-checkpoint``."""
    if explicit:
        if not os.path.isfile(explicit):
            raise SystemExit(f"ERROR: checkpoint not found: {explicit}")
        return explicit
    run_dir = os.path.join("checkpoints_stage2", exp_name)
    for name in ("last.ckpt", "best-checkpoint.ckpt"):
        path = os.path.join(run_dir, name)
        if os.path.isfile(path):
            return path
    raise SystemExit(
        f"ERROR: no checkpoint in {run_dir}/ (looked for last.ckpt, "
        f"best-checkpoint.ckpt).\n"
        f"       Pass --checkpoint <path>, or run "
        f"`--list_checkpoints` to see the available runs.")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="fdisk-style REPL for stage-2 (CodeTransformer) "
                    "inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Inside the REPL press 'm' for the key list.",
    )
    parser.add_argument("--config", default="configs/transformer.yaml",
                        help="stage-2 config (default: %(default)s)")
    parser.add_argument("--checkpoint", default=None,
                        help="stage-2 checkpoint (default: "
                             "checkpoints_stage2/<exp_name>/last.ckpt)")
    parser.add_argument("--generator_checkpoint", default=None,
                        help="stage-1 Generator for audio decode (default: "
                             "train.generator_checkpoint from the config)")
    parser.add_argument("--no_audio", action="store_true",
                        help="never load the stage-1 Generator (codes only)")
    parser.add_argument("--play", action="store_true",
                        help="play every decoded clip through pydub as soon as "
                             "it is generated (needs ffplay or simpleaudio; "
                             "`b` replays it, space works too)")
    parser.add_argument("--device", default=None,
                        help="cuda/cpu (default: cuda when available)")
    parser.add_argument("--precision", default="fp32",
                        choices=("fp32", "bf16", "fp16"),
                        help="dtype for the transformer's weights and sampling "
                             "path (default: %(default)s, matching stage 2's "
                             "32-true convention). fp16/bf16 need CUDA; the "
                             "stage-1 vocoder always decodes fp32. `z` in the "
                             "REPL reports the resulting memory use.")
    parser.add_argument("--line_mode", action="store_true",
                        help="read whole lines instead of single keystrokes "
                             "(automatic when stdin is not a TTY)")
    parser.add_argument("--out_dir", default="scratch/repl_out",
                        help="where `w`/`x` write without a path "
                             "(default: %(default)s)")
    # Initial state / one-shot mode.
    parser.add_argument("--text", default=None,
                        help="phonemize this and sample once, then exit "
                             "(use {HH AH0 L OW1} for manual ARPAbet)")
    parser.add_argument("--val", type=int, default=None,
                        help="free-run val item N, then exit")
    parser.add_argument("--list_speakers", action="store_true",
                        help="list the val split's speaker ids, then exit")
    parser.add_argument("--list_checkpoints", action="store_true",
                        help="list the stage-2 checkpoints, then exit "
                             "(no model is loaded)")
    parser.add_argument("--sid", type=int, default=None, help="speaker id")
    parser.add_argument("--frames", type=int, default=None,
                        help="max code frames to generate")
    parser.add_argument("--temp", type=float, default=None,
                        help="sampling temperature")
    parser.add_argument("--topk", default=None,
                        help="top-k truncation ('none' disables)")
    parser.add_argument("--out", default=None,
                        help="one-shot mode: write the decoded audio here")
    parser.add_argument("--dump_codes", default=None,
                        help="one-shot mode: torch.save the sampled codes here")
    parser.add_argument("--hints", default=None,
                        choices=("cond", "null", "omit"),
                        help="base hint state: 'cond' = the loaded val item's "
                             "own .cond hints (arbitrary text has none and "
                             "falls back to null), 'null' = the trained "
                             "'not provided' state (level 0 / word_null), "
                             "'omit' = send no vectors at all (pre-Phase-0, "
                             "out of distribution). Default: train.sample_hints.")
    parser.add_argument("--hint_intensity", type=int, default=None,
                        help="intensity level override (0-8; 0 = absent)")
    parser.add_argument("--hint_register", type=int, default=None,
                        help="register level override (0-5; 0 = absent)")
    parser.add_argument("--hint_pitch", type=float, default=None,
                        help="per-word pitch shift in semitones (2 st per "
                             "bucket; applied to present word hints)")
    parser.add_argument("--hint_slot", type=int, default=None,
                        help="per-word log2 duration shift (+1 = twice as long)")
    parser.add_argument("--hint_word", default=None,
                        help="per-word BERT channel: 'cond' = the loaded val "
                             "item's own states, 'text' = embed the current "
                             "sentence's words, 'null' = word_null, or "
                             "'swap:<a sentence>' to inject that sentence's "
                             "states (the word-swap test)")
    parser.add_argument("--hint_word_text", default=None,
                        help="inject this sentence's per-word BERT states "
                             "instead of the current text's (word-swap test)")
    parser.add_argument("--seed", type=int, default=None,
                        help="torch seed, so paired A/B draws are comparable")
    parser.add_argument("--infill_best_of", type=int, default=1,
                        help="fill (`n`): draw this many candidate spans and "
                             "keep the one that best predicts the real suffix "
                             "start (needs a fim_lookahead checkpoint)")
    parser.add_argument("--stage1_sid_map", default=None,
                        help="speaker_map.py JSON (stage-2 -> stage-1 ids), "
                             "applied only when the frozen Generator decodes; "
                             "fixes a run trained in a different id space")
    return parser.parse_args(argv)



def main(argv=None) -> int:
    args = parse_args(argv)

    from omegaconf import OmegaConf

    torch.set_float32_matmul_precision("high")  # stage-2 convention (TF32)
    config = OmegaConf.load(args.config)
    if args.list_checkpoints:
        # Before resolve_checkpoint: listing must work even when the config's
        # own exp_name has no checkpoint yet.
        list_checkpoints(args.checkpoint)
        return 0
    checkpoint = resolve_checkpoint(args.checkpoint, str(config.exp_name))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    gen_ckpt = (None if args.no_audio else
                (args.generator_checkpoint
                 or config.train.get("generator_checkpoint", None)))

    repl = Stage2REPL(
        config, checkpoint, device,
        generator_checkpoint=gen_ckpt,
        audio=not args.no_audio,
        out_dir=args.out_dir,
        auto_play=args.play,
        reader=KeyReader(force_line_mode=args.line_mode),
        hint_mode=args.hints,
        precision=args.precision,
        stage1_sid_map=args.stage1_sid_map)


    # Initial state, before the (slow) model build, so --list_speakers stays
    # cheap and a bad --sid is obvious from the banner.
    if args.sid is not None:
        repl.sid = int(args.sid)
    if args.frames is not None:
        repl.frames = int(args.frames)
    if args.temp is not None:
        repl.temperature = float(args.temp)
    if args.hint_intensity is not None:
        repl.hint_intensity = int(args.hint_intensity)
    if args.hint_register is not None:
        repl.hint_register = int(args.hint_register)
    if args.hint_pitch is not None:
        repl.hint_pitch_st = float(args.hint_pitch)
    if args.hint_slot is not None:
        repl.hint_slot_shift = int(args.hint_slot)
    if args.hint_word is not None:
        repl.set_hint_word(str(args.hint_word), announce=False)
    if args.hint_word_text is not None:
        repl.hint_word_mode = "text"
        repl.hint_word_text = str(args.hint_word_text)
    if args.seed is not None:
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
    if args.infill_best_of and int(args.infill_best_of) > 1:
        repl.infill_best_of = int(args.infill_best_of)
    if args.topk is not None:
        text = str(args.topk).strip().lower()
        if text in ("none", "off", "0"):
            repl.top_k = None
        else:
            try:
                repl.top_k = int(text)
            except ValueError:
                raise SystemExit(f"ERROR: --topk expects an integer or "
                                 f"'none', got {args.topk!r}")

    print(f"device: {device}   precision: {repl.precision}   "
          f"input: {repl.reader.describe()}")
    if args.list_speakers:
        repl.list_speakers()   # cheap: only the val filelist, no model
        return 0
    repl.load_models()

    if args.text is not None or args.val is not None:
        if args.text is not None:
            ok = repl.set_text(args.text) and repl.generate()
        else:
            ok = repl.val_item(args.val)
        if not ok:
            return 1
        if args.out:
            repl.save_audio(args.out)
        if args.dump_codes:
            repl.dump_codes(args.dump_codes)
        return 0

    run_repl(repl)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:  # e.g. `| head`
        sys.exit(0)

