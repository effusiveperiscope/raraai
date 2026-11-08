#!/usr/bin/env python3
"""fdisk-style interactive REPL for stage-2 (``CodeTransformer``) inference.

Loads a stage-2 checkpoint plus, optionally, the frozen stage-1 ``Generator``
that turns sampled codes back into audio, and then lets you poke at the model
from a single-keystroke menu: phonemize a sentence, pick a speaker, sample
codes, listen to / save the result, or compare a free-running sample against a
validation item's ground truth.

Fill-in-the-middle (:meth:`CodeTransformer.infill`) is deliberately *not* wired
in: the shipped runs are pre-FIM, so the SUF/MID sentinel classes are untrained
and infill output would be meaningless. ``generate`` (plain AR) is the only
sampling path used here.

Usage
-----
    # interactive, shipped config + its checkpoint (single keys, fdisk style)
    python scratch/stage2_repl.py

    # a specific run / checkpoint, no stage-1 audio decode
    python scratch/stage2_repl.py \\
        --checkpoint checkpoints_stage2/transformer_v0.5a/last.ckpt --no_audio

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

    s  switch stage-2 checkpoint (prompt: run name or path; ``?`` lists runs)

Notes
-----
* Speaker ids are in the **stage-1** id space (``speaker_map.py`` canonicalises
  them); that is the only id the frozen ``Generator`` can voice.
* The phonemizer is ``FeatureExtractor(..., load_qwen=False)`` -- g2p_en + the
  g2pen tokenizer only, no ASR / aligner / RMVPE weights and no audio input.
  Same path as ``2_preprocess.py --phones_from transcript``.
* ``transformer.eos_token`` / ``fim`` / ``region_cond`` change the codebook
  embedding/head shapes, so a config that disagrees with its checkpoint would
  silently re-initialise them (``load_state_dict_mismatch`` skips whole
  tensors). The layout is therefore read back **from the checkpoint** and the
  config adjusted before the model is built; the printed notes say when.
* Read-only: this never writes a checkpoint or touches a log tree.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from typing import Optional, Tuple

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from commons import load_state_dict_mismatch  # noqa: E402
from modeling.transformer import CodeTransformer  # noqa: E402


# ---------------------------------------------------------------------- #
# fdisk-style single-key input
# ---------------------------------------------------------------------- #
# Long names accepted in --line_mode (key mode only ever needs the letter).
LINE_ALIASES = {
    "help": "m", "status": "p", "info": "p", "print": "p", "list": "l",
    "text": "t", "say": "y", "gen": "g", "generate": "g", "val": "v",
    "value": "v", "sid": "i", "id": "i", "frames": "f", "temp": "h",
    "temperature": "h", "topk": "k", "audio": "a", "write": "w",
    "save": "w", "dump": "x", "codes": "c",
    "switch": "s", "load": "s", "open": "s", "ckpt": "s",
    "checkpoint": "s", "checkpoints": "s",
    "quit": "q", "exit": "q",
}


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
# REPL
# ---------------------------------------------------------------------- #
class Stage2REPL:
    """Small state machine around ``CodeTransformer.generate`` (+ stage 1)."""

    def __init__(self, config, checkpoint: str, device: str,
                 generator_checkpoint: Optional[str] = None,
                 audio: bool = True,
                 out_dir: str = "scratch/repl_out",
                 reader: Optional[KeyReader] = None):
        self.config = config
        self.checkpoint = checkpoint
        self.device = device
        self.generator_checkpoint = generator_checkpoint
        self.audio = bool(audio) and generator_checkpoint is not None
        self._audio_disabled = not audio
        self.out_dir = out_dir
        self.reader = reader or KeyReader()

        self.frame_rate = (int(config.data.sampling_rate)
                           // int(config.data.hop_length))
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

        self.net: Optional[CodeTransformer] = None
        # The eos/fim/region_cond layout is *derived from the checkpoint* (see
        # `_align_special_layout`), so the config is mutated on every load.  The
        # values as configured are kept here so switching checkpoints at runtime
        # (`s`) always starts from the same place instead of compounding.
        self._layout_originals = {
            k: config.transformer.get(k, None)
            for k in ("eos_token", "fim", "region_cond")}
        self._fe = None            # FeatureExtractor (lazy: g2p + tokenizer)
        self._net_g = None         # stage-1 Generator (lazy)
        self._net_g_failed = False
        self._val_ds = None
        self._val_cursor = 0
        self._writes = 0

        self.last_codes = None     # (K, n) long, on self.device
        self.last_sid: Optional[int] = None
        self.last_audio = None     # (T,) float32 CPU waveform
        self.last_f0 = None        # (T,) float32 CPU f0 in Hz

    # ------------------------------------------------------------------ #
    # Models
    # ------------------------------------------------------------------ #
    def load_models(self) -> None:
        """Build the transformer (+ stage-1 Generator) and report the load."""
        print(f"Stage-2 checkpoint: {self.checkpoint}")
        state = read_state_dict(self.checkpoint)
        self._align_special_layout(state)
        self.net = CodeTransformer(self.config).to(self.device)
        print(self.net.describe())
        n_loaded, n_ckpt, n_bad, n_missing = load_weights(self.net, state)
        print(f"Loaded {n_loaded}/{n_ckpt} checkpoint tensor(s)"
              + (f"; {n_bad} skipped, {n_missing} missing"
                 if n_bad or n_missing else ""))
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
        net = CodeTransformer(self.config).to(self.device)
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
            ids = [int(i) for i in fe.phonemize_text(text)]
        except Exception as exc:
            print(f"Phonemization failed: {exc}")
            return False
        if not ids:
            print(f"{text!r} produced no phonemes; text left unchanged.")
            return False
        self.text, self.phone_ids = text, ids
        print(f"text: {text}")
        print(f"phones: {fe.tokenizer_g2p.decode(ids)!r}  ({len(ids)} ids)")
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
        codes, lens = self.net.generate(
            phone, plen, sid, max_frames=frames,
            temperature=self.temperature, top_k=self.top_k,
            return_lengths=True)
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
        sid_t = torch.tensor([int(sid)], dtype=torch.long, device=self.device)
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
        return True

    # ------------------------------------------------------------------ #
    # Validation-item comparison
    # ------------------------------------------------------------------ #
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
                speaker_map=self.config.train.get("speaker_map"))
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
        code_len = int(batch["code_lengths"][0])
        frames = self._clamp_frames(min(self.frames, code_len))

        print(f"val item {index}/{len(ds) - 1}: sid {self.sid}, {code_len} "
              f"frames = {code_len / self.frame_rate:.2f} s, "
              f"{len(self.phone_ids)} phone ids")
        print(f"  phones: {self._phones_text()!r}")

        codes, lens = self.net.generate(
            batch["phone_ids"], batch["phone_lengths"], batch["sid"],
            max_frames=frames, temperature=self.temperature,
            top_k=self.top_k, return_lengths=True)
        pred = codes[0]
        n = int(pred.shape[-1])
        gt = torch.cat([batch["spec_codes"][0], batch["pitch_codes"][0]],
                       dim=0)[:, :n]
        self.last_codes, self.last_sid = pred, self.sid
        self.last_audio = self.last_f0 = None
        print(f"generated {n} of at most {frames} frames"
              + (f" (EOS length {int(lens[0])})" if self.net.eos_token else ""))
        self._report_codes(pred)
        self._report_match(gt, pred)
        self._decode_audio(pred, self.sid)
        return True

    def set_phone_ids(self, ids) -> None:
        """Set the conditioning *without* phonemizing (val items carry ids)."""
        self.phone_ids = [int(i) for i in ids]
        self.text = ""

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
            print("Nothing decoded yet -- press g (generate) or a (audio).")
            return False
        path = (path or "").strip()
        if not path:
            os.makedirs(self.out_dir, exist_ok=True)
            self._writes += 1
            path = os.path.join(
                self.out_dir,
                f"say_sid{self.last_sid}_{int(self.last_codes.shape[-1])}f_"
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
        """``torch.save`` the last sampled ``(K, n)`` code tensor."""
        if self.last_codes is None:
            print("Nothing sampled yet -- press g (generate).")
            return False
        path = (path or "").strip()
        if not path:
            os.makedirs(self.out_dir, exist_ok=True)
            path = os.path.join(self.out_dir, f"codes_sid{self.last_sid}_"
                                              f"{time.strftime('%H%M%S')}.pt")
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(self.last_codes.cpu(), path)
        print(f"wrote {path}  (K, n) = {tuple(self.last_codes.shape)}")
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
                      f"region_cond={self.net.region_cond}")
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
        print(f"device     : {self.device}   input: {self.reader.describe()}")
        print(f"layout     : {layout}")
        print(f"text       : {self.text or '<none>'}"
              + (f"  ({len(self.phone_ids)} phone ids)" if self.phone_ids
                 else ""))
        print(f"sid        : {self.sid}")
        print(f"frames     : {self.frames} "
              f"(~{self.frames / self.frame_rate:.2f} s at "
              f"{self.frame_rate:g} Hz)")
        print(f"temperature: {self.temperature:g}   top_k: {self.top_k}")
        print(f"audio      : {audio}")
        if self.last_codes is not None:
            sr = int(self.config.data.sampling_rate)
            extra = (f", audio {self.last_audio.numel() / sr:.2f} s"
                     if self.last_audio is not None else ", not decoded")
            print(f"last codes : {tuple(self.last_codes.shape)}, "
                  f"sid {self.last_sid}{extra}")

    def print_help(self) -> None:
        print(HELP_TEXT)


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
  s  switch stage-2 checkpoint (prompt: run name or path; `?` lists them)

  Blank at a prompt keeps the current value; blank at a wav/codes prompt
  writes into the output directory with a generated name.  `s` rebuilds the
  model in place, re-deriving eos_token/fim/region_cond from the new
  checkpoint, and clears the last codes/audio (they came from the old one).
"""


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
          f"top_k={repl.top_k}  "
          f"audio={'on' if repl.audio and not repl._net_g_failed else 'off'}")
    print("  press m for help, q to quit.")


def dispatch(repl: "Stage2REPL", key: str, rest: str = "") -> bool:
    """Run one command. Returns ``False`` when the REPL should exit."""
    key = key.lower()
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
            if repl.last_codes is None:
                print("Nothing sampled yet -- press g (generate).")
            else:
                repl._report_codes(repl.last_codes)
        elif key == "t":
            text = rest or repl.reader.read_line("Text: ")
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
        elif key == "a":
            if repl.last_codes is None:
                print("Nothing sampled yet -- press g (generate).")
            else:
                repl._decode_audio(repl.last_codes, repl.last_sid)
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
    parser.add_argument("--device", default=None,
                        help="cuda/cpu (default: cuda when available)")
    parser.add_argument("--line_mode", action="store_true",
                        help="read whole lines instead of single keystrokes "
                             "(automatic when stdin is not a TTY)")
    parser.add_argument("--out_dir", default="scratch/repl_out",
                        help="where `w`/`x` write without a path "
                             "(default: %(default)s)")
    # Initial state / one-shot mode.
    parser.add_argument("--text", default=None,
                        help="phonemize this and sample once, then exit")
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
        reader=KeyReader(force_line_mode=args.line_mode))

    # Initial state, before the (slow) model build, so --list_speakers stays
    # cheap and a bad --sid is obvious from the banner.
    if args.sid is not None:
        repl.sid = int(args.sid)
    if args.frames is not None:
        repl.frames = int(args.frames)
    if args.temp is not None:
        repl.temperature = float(args.temp)
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

    print(f"device: {device}   input: {repl.reader.describe()}")
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

