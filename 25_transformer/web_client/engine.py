"""Inference runtime for the stage-2 web client.

The heavy lifting (phonemization, checkpoint/layout discovery, stage-1 audio
decode, hint bookkeeping, FIM infill) already exists in
``scratch/stage2_repl.py``.  This module does **not** reimplement it: it builds
a :class:`StudioREPL` (a ``Stage2REPL`` subclass that also derives the LoRA
layout from a checkpoint) and exposes request-shaped, thread-safe operations on
top of it.

Design points that matter for the server:

* one :class:`LoadedSlot` per ``model/variant``, each with its own lock, so a
  generation is serialized *per loaded model*.  The server additionally runs a
  single worker thread, which is what makes a single GPU safe.
* generations are pure: ``generate``/``infill`` return a :class:`Render`
  (codes + audio + metadata) instead of mutating the REPL's "last result".
  That is what lets the server keep one history per browser session.
* model construction is guarded by a process-wide load lock and the configured
  ``max_loaded_models`` (1 by default: a 1.1 GB transformer plus a 0.5 GB
  vocoder already use a third of a 16 GB card).
* every optional layout switch (EOS / FIM / RoPE / hints / LoRA) is derived
  from the checkpoint, exactly as the REPL does, so a checkpoint cannot be
  silently loaded into a mismatched config.
"""
from __future__ import annotations

import copy
import json
import os
import random
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (REPO_ROOT, os.path.join(REPO_ROOT, "scratch")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from stage2_repl import (  # noqa: E402  (path set up above)
    KeyReader,
    Stage2REPL,
    layout_from_rows,
    phonemize_with_arpabet,
    read_state_dict,
)


# ---------------------------------------------------------------------- #
# Errors / registry
# ---------------------------------------------------------------------- #
class RequestError(Exception):
    """A bad request (HTTP 400): unphonemizable text, out-of-range value..."""

    status_code = 400


class EngineError(Exception):
    """A runtime failure that is not the caller's fault (HTTP 500)."""

    status_code = 500


@dataclass
class Variant:
    """One loadable checkpoint (a base run or a LoRA fine-tune of it)."""

    id: str
    label: str
    config: str
    checkpoint: str
    generator_checkpoint: Optional[str] = None
    notes: str = ""
    is_lora: bool = False

    @property
    def key(self) -> str:
        return self.id


@dataclass
class ModelEntry:
    id: str
    label: str
    notes: str
    variants: List[Variant]

    def variant(self, variant_id: Optional[str]) -> Variant:
        if not variant_id:
            return self.variants[0]
        for v in self.variants:
            if v.id == variant_id:
                return v
        raise RequestError(f"unknown variant {variant_id!r} for model "
                           f"{self.id!r}")


def _abs(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    path = os.path.expanduser(str(path))
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def load_registry(path: str) -> Dict[str, Any]:
    """Parse the YAML registry into models/variants/defaults/server blocks."""
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    server = dict(raw.get("server") or {})
    defaults = dict(raw.get("defaults") or {})
    models_raw = raw.get("models") or {}
    if not models_raw:
        raise SystemExit(f"{path}: no `models:` block")

    models: List[ModelEntry] = []
    for model_id, block in models_raw.items():
        block = dict(block or {})
        if not block.get("config") or not block.get("checkpoint"):
            raise SystemExit(f"{path}: model {model_id!r} needs `config` and "
                             f"`checkpoint`")
        model_cfg = _abs(block["config"])
        model_ckpt = _abs(block["checkpoint"])
        gen = _abs(block.get("generator_checkpoint"))
        variants_raw = block.get("variants") or {}
        variants: List[Variant] = []
        if variants_raw:
            for vid, vblock in variants_raw.items():
                vblock = dict(vblock or {})
                variants.append(Variant(
                    id=str(vid),
                    label=str(vblock.get("label") or vid),
                    config=_abs(vblock.get("config")) or model_cfg,
                    checkpoint=_abs(vblock.get("checkpoint")) or model_ckpt,
                    generator_checkpoint=(_abs(vblock.get("generator_checkpoint"))
                                          or gen),
                    notes=str(vblock.get("notes") or ""),
                    is_lora=bool(vblock.get("lora", False)),
                ))
        else:
            variants.append(Variant(id="base", label="Base",
                                    config=model_cfg, checkpoint=model_ckpt,
                                    generator_checkpoint=gen))
        if not any(v.id == "base" for v in variants):
            variants.insert(0, Variant(id="base", label="Base",
                                       config=model_cfg, checkpoint=model_ckpt,
                                       generator_checkpoint=gen))
        models.append(ModelEntry(
            id=str(model_id),
            label=str(block.get("label") or model_id),
            notes=str(block.get("notes") or ""),
            variants=variants,
        ))
    return {"server": server, "defaults": defaults, "models": models}


# ---------------------------------------------------------------------- #
# REPL subclass: checkpoint-derived LoRA layout
# ---------------------------------------------------------------------- #
def _strip_net(key: str) -> str:
    return key[len("net."):] if key.startswith("net.") else key


def infer_lora_targets(state: Dict[str, torch.Tensor]) -> List[str]:
    """Which ``finetune.targets`` produced this checkpoint's adapters.

    ``lora_marker`` records rank/alpha/count but not *which* modules were
    adapted, and a wrong target list either fails injection or silently drops
    trained adapters.  The module path of every ``.lora_*`` tensor is enough.
    """
    targets = set()
    for raw_key in state:
        key = _strip_net(raw_key)
        if not (".lora_A" in key or ".lora_B" in key
                or ".lora_qkv_A" in key or ".lora_qkv_B" in key):
            continue
        if key.startswith("encoder.") and ".linear" in key:
            targets.add("encoder_ffn")
        elif key.startswith("decoder."):
            if ".self_attn.out_proj" in key:
                targets.add("decoder_attn")
            elif "lora_qkv" in key:
                targets.add("decoder_qkv")
            elif ".linear" in key:
                targets.add("decoder_ffn")
    return [t for t in ("encoder_ffn", "decoder_ffn", "decoder_attn",
                        "decoder_qkv") if t in targets]


class StudioREPL(Stage2REPL):
    """``Stage2REPL`` + a LoRA layout derived from the checkpoint.

    The parent's ``_align_special_layout`` matches EOS/FIM/RoPE/hints to the
    checkpoint.  LoRA is the same kind of contract (rank/alpha/targets change
    the parameter set), so it is handled in the same place -- before the model
    is built.  Without this a LoRA checkpoint would load with every base tensor
    matching and the result would silently be the base model
    (``load_state_dict_mismatch`` refuses that, but there is no point in
    crashing when the checkpoint says exactly what to build).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ft = self.config.get("finetune", None)
        self._ft_original = (OmegaConf.to_container(ft, resolve=True)
                             if ft is not None else {})
        self._ft_original.setdefault("enabled", False)

    def _reset_finetune(self) -> None:
        self.config.finetune = OmegaConf.create(copy.deepcopy(self._ft_original))

    def _align_special_layout(self, state, source=None) -> None:
        self._reset_finetune()
        super()._align_special_layout(state, source)
        self._align_lora(state, source or self.checkpoint)

    def _align_lora(self, state: Dict[str, torch.Tensor],
                    source: str) -> None:
        has_lora = any(".lora_" in k for k in state)
        if not has_lora:
            self.config.finetune.enabled = False
            return

        marker = None
        for key in state:
            if key == "lora_marker" or key.endswith(".lora_marker"):
                marker = state[key]
                break
        base = OmegaConf.to_container(self.config.finetune, resolve=True) or {}
        rank = int(base.get("rank", 16) or 16)
        alpha = float(base.get("alpha", 32.0) or 32.0)
        if marker is not None and torch.is_tensor(marker) and marker.numel() >= 2:
            rank, alpha = int(marker[0]), float(marker[1])
        targets = infer_lora_targets(state)
        if not targets:
            raise EngineError(f"{source}: LoRA adapters found but no adapted "
                              f"module could be identified")
        n_marker = (int(marker[2]) if marker is not None
                    and torch.is_tensor(marker) and marker.numel() >= 3 else None)
        if n_marker is not None and n_marker != len(targets):
            raise EngineError(
                f"{source}: lora_marker says {n_marker} adapted target group(s) "
                f"but the tensor names resolve to {targets}")
        print(f"LoRA: {source} -> rank {rank}, alpha {alpha:g}, "
              f"targets {targets}")
        self.config.finetune = OmegaConf.create({
            **base, "enabled": True, "rank": rank, "alpha": alpha,
            "targets": targets,
        })


# ---------------------------------------------------------------------- #
# Loaded model slot
# ---------------------------------------------------------------------- #
@dataclass
class LoadedSlot:
    model_id: str
    variant_id: str
    key: str
    repl: StudioREPL
    lock: threading.Lock = field(default_factory=threading.Lock)
    caps: Dict[str, Any] = field(default_factory=dict)
    loaded_at: float = field(default_factory=time.time)


def slot_key(model_id: str, variant_id: str) -> str:
    return f"{model_id}:{variant_id}"


# ---------------------------------------------------------------------- #
# Render (one generation's data, independent of REPL state)
# ---------------------------------------------------------------------- #
@dataclass
class Render:
    model_id: str
    variant_id: str
    kind: str
    text: str
    sid: int
    phone_ids: List[int]
    phones: str
    word_index: Optional[List[int]]
    word_texts: List[Optional[str]]
    frames: int
    frame_rate: float
    sampling_rate: int
    temperature: float
    top_k: Optional[int]
    seed: int
    hints_state: str
    hints_params: Dict[str, Any]
    codes: torch.Tensor                      # (K, n) on CPU
    audio: Optional[np.ndarray]              # 1-D float32, or None
    f0: Optional[np.ndarray]
    eos_length: Optional[int] = None
    unique_codes: List[int] = field(default_factory=list)
    repeat_frac: float = 0.0
    timings: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    source_job: Optional[str] = None
    span: Optional[Dict[str, int]] = None
    seam_ce: Optional[float] = None
    best_of: int = 1
    job_id: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.frames / self.frame_rate if self.frame_rate else 0.0

    def f0_summary(self) -> Dict[str, float]:
        if self.f0 is None or self.f0.size == 0:
            return {"voiced_frac": 0.0, "median_hz": 0.0}
        voiced = self.f0 > 0
        return {
            "voiced_frac": float(voiced.mean()),
            "median_hz": (float(np.median(self.f0[voiced]))
                          if voiced.any() else 0.0),
        }

    def wav_bytes(self, subtype: str = "PCM_16") -> Optional[bytes]:
        if self.audio is None:
            return None
        import io
        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, np.clip(self.audio, -1.0, 1.0), self.sampling_rate,
                 format="WAV", subtype=subtype)
        return buf.getvalue()

    def codes_bytes(self) -> bytes:
        import io

        buf = io.BytesIO()
        torch.save({"codes": self.codes.cpu(), "sid": int(self.sid),
                    "frame_rate": self.frame_rate, "text": self.text,
                    "model": self.model_id, "variant": self.variant_id},
                   buf)
        return buf.getvalue()


# ---------------------------------------------------------------------- #
# Engine
# ---------------------------------------------------------------------- #
class StudioEngine:
    def __init__(self, registry: Dict[str, Any], server_cfg: Dict[str, Any]):
        self.registry = registry
        self.server_cfg = dict(server_cfg or {})
        self.max_loaded = max(1, int(self.server_cfg.get("max_loaded_models", 1)))
        self.precision = str(self.server_cfg.get("precision", "fp32"))
        self.device = (self.server_cfg.get("device")
                       or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.out_dir = _abs(self.server_cfg.get("out_dir")) or os.path.join(
            REPO_ROOT, "web_client", "outputs")
        os.makedirs(self.out_dir, exist_ok=True)

        self._slots: "OrderedDict[str, LoadedSlot]" = OrderedDict()
        self._slots_lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._probe_lock = threading.Lock()
        self._state_cache: Dict[str, Dict[str, Any]] = {}
        self._config_cache: Dict[str, Any] = {}
        self._speaker_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._global_extractor = None

        torch.set_float32_matmul_precision("high")   # stage-2 convention (TF32)

    # ------------------------------------------------------------------ #
    # Registry lookups
    # ------------------------------------------------------------------ #
    def model(self, model_id: Optional[str]) -> ModelEntry:
        models = self.registry["models"]
        if model_id:
            for m in models:
                if m.id == model_id:
                    return m
            raise RequestError(f"unknown model {model_id!r}")
        fixed = self.server_cfg.get("fixed_model")
        if fixed:
            for m in models:
                if m.id == fixed:
                    return m
        return models[0]

    def variant(self, model_id: Optional[str],
                variant_id: Optional[str]) -> Tuple[ModelEntry, Variant]:
        fixed = self.server_cfg.get("fixed_model")
        if fixed:
            if model_id and model_id != fixed:
                raise RequestError(
                    f"this server is pinned to model {fixed!r}; {model_id!r} "
                    f"is not available")
            model_id = fixed
        model = self.model(model_id)
        fixed_variant = self.server_cfg.get("fixed_variant")
        if fixed_variant and model.id == self.server_cfg.get("fixed_model"):
            variant_id = fixed_variant
        return model, model.variant(variant_id)

    def visible_models(self) -> List[ModelEntry]:
        fixed = self.server_cfg.get("fixed_model")
        if fixed:
            return [self.model(fixed)]
        return list(self.registry["models"])

    # ------------------------------------------------------------------ #
    # Config / checkpoint inspection (cached, cheap)
    # ------------------------------------------------------------------ #
    def _config(self, path: str):
        with self._probe_lock:
            cached = self._config_cache.get(path)
        if cached is not None:
            return OmegaConf.create(copy.deepcopy(cached))
        cfg = OmegaConf.load(path)
        with self._probe_lock:
            self._config_cache[path] = OmegaConf.to_container(cfg, resolve=True)
        return OmegaConf.create(copy.deepcopy(self._config_cache[path]))

    @staticmethod
    def _mtime(path: str) -> Tuple[float, int]:
        try:
            st = os.stat(path)
            return st.st_mtime, st.st_size
        except OSError:
            return 0.0, 0

    def _state_index(self, path: str) -> Dict[str, Any]:
        """Keys + a few shapes of a checkpoint, without materialising tensors.

        ``mmap=True`` + ``weights_only=True`` makes this a header read on a
        ~1 GB file, so the model picker can show real capabilities before any
        model is loaded.  A checkpoint that cannot be header-read falls back to
        an empty index (the capabilities then come from the config and the
        runtime merge).
        """
        stamp = self._mtime(path)
        with self._probe_lock:
            cached = self._state_cache.get(path)
        if cached is not None and cached["stamp"] == stamp:
            return cached
        keys: List[str] = []
        shapes: Dict[str, Tuple[int, ...]] = {}
        marker = None
        lookahead = None
        try:
            raw = torch.load(path, map_location="cpu", weights_only=True,
                             mmap=True)
            sd = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
            for key, value in sd.items():
                k = _strip_net(key)
                keys.append(k)
                if torch.is_tensor(value) and k in (
                        "decoder.embeds.0.weight", "lora_marker",
                        "fim_layout_marker", "pitch_val_dir", "word_proj.weight"):
                    shapes[k] = tuple(value.shape)
                    if k == "lora_marker":
                        marker = [float(x) for x in value.reshape(-1).tolist()]
                    elif k == "fim_layout_marker":
                        lookahead = int(value.reshape(-1)[0].item())
            del raw, sd
        except Exception as exc:  # pragma: no cover - unusual checkpoint
            print(f"web_client: could not header-read {path}: {exc}")
        info = {"stamp": stamp, "keys": keys, "shapes": shapes,
                "lora_marker": marker, "fim_lookahead": lookahead}
        with self._probe_lock:
            self._state_cache[path] = info
        return info

    @staticmethod
    def _has(keys: List[str], suffix: str) -> bool:
        return any(k == suffix or k.endswith("." + suffix) for k in keys)

    def probe_caps(self, model_id: Optional[str],
                   variant_id: Optional[str]) -> Dict[str, Any]:
        """Capabilities visible *before* loading (checkpoint header + config)."""
        model, variant = self.variant(model_id, variant_id)
        cfg = self._config(variant.config)
        idx = self._state_index(variant.checkpoint)
        keys = idx["keys"]
        shapes = idx["shapes"]

        vocab = int(cfg.codec.codebook_size)
        eos = bool(cfg.transformer.get("eos_token", False))
        fim = bool(cfg.transformer.get("fim", False))
        rows = shapes.get("decoder.embeds.0.weight")
        if rows:
            try:
                eos, fim = layout_from_rows(int(rows[0]), vocab)
            except ValueError:
                pass

        has_hints = any(self._has(keys, k) for k in (
            "int_emb.weight", "int_dir", "pitch_emb.weight", "pitch_val_dir",
            "word_proj.weight"))
        anchors = self._has(keys, "pitch_val_dir")
        pitch_anchors = (int(shapes["pitch_val_dir"][0])
                         if anchors and "pitch_val_dir" in shapes else 0)
        word_dim = (int(shapes["word_proj.weight"][1])
                    if "word_proj.weight" in shapes else 0)

        # Hint level widths live in hints.json (the same file the model reads).
        n_int, n_reg, n_slot, n_pitch = 8, 5, 8, 9
        hints_json = _abs(cfg.transformer.get("hints_json"))
        if hints_json and os.path.isfile(hints_json):
            try:
                with open(hints_json, "r", encoding="utf-8") as fh:
                    hj = json.load(fh)
                levels = hj.get("levels") or {}
                n_int = int((levels.get("intensity") or {}).get("n_levels")
                            or n_int)
                n_reg = int((levels.get("register") or {}).get("n_levels")
                            or n_reg)
                words = hj.get("words") or {}
                n_slot = int(words.get("n_slot_buckets") or n_slot)
                if not anchors:
                    n_pitch = len(words.get("pitch_edges") or []) + 2 or n_pitch
            except Exception as exc:
                print(f"web_client: could not read {hints_json}: {exc}")

        sr = int(cfg.data.sampling_rate)
        frame_rate = (sr // int(cfg.data.hop_length)
                      // int(cfg.data.get("code_down_factor", 4) or 4))
        max_positions = int(cfg.transformer.get("max_positions", 4096))
        codebooks = (int(cfg.codec.codebook_num)
                     + int(cfg.pitch_codec.codebook_num))
        positional = max(1, max_positions - codebooks + 1)
        train_window = int(cfg.train.get("max_frames", 0) or 0)
        max_frames = min(positional, train_window) if train_window else positional
        marker = idx.get("lora_marker")
        has_lora = any(".lora_" in k for k in keys)

        return {
            "model": model.id,
            "variant": variant.id,
            "label": model.label,
            "variant_label": variant.label,
            "config": os.path.relpath(variant.config, REPO_ROOT),
            "checkpoint": os.path.relpath(variant.checkpoint, REPO_ROOT),
            "checkpoint_bytes": self._mtime(variant.checkpoint)[1],
            "audio": bool(variant.generator_checkpoint),
            "fim": bool(fim),
            "eos": bool(eos),
            "fim_lookahead": int(idx.get("fim_lookahead") or 0),
            "region_cond": self._has(keys, "decoder.region_emb.weight"),
            "length_cond": self._has(keys, "decoder.len_emb.weight"),
            "rope": self._has(keys, "rope_marker"),
            "hints": has_hints,
            "decoder_hints": (has_hints and (self._has(keys, "dec_int_emb.weight")
                                             or self._has(keys, "dec_int_dir"))),
            "hint_monotone": self._has(keys, "int_dir"),
            "word_sem": self._has(keys, "word_proj.weight"),
            "word_dim": word_dim,
            "pitch_mode": "anchors" if anchors else "table",
            "pitch_anchors": pitch_anchors,
            "n_intensity": n_int,
            "n_register": n_reg,
            "n_pitch": n_pitch,
            "n_slot": n_slot,
            "n_speakers": int(cfg.transformer.get("n_speakers", 4096)),
            "sampling_rate": sr,
            "frame_rate": frame_rate,
            "max_frames": max_frames,
            "max_positions_frames": positional,
            "lora": bool(has_lora),
            "lora_rank": int(marker[0]) if marker else 0,
            "lora_alpha": float(marker[1]) if marker else 0.0,
            "loaded": self.is_loaded(model.id, variant.id),
        }

    # ------------------------------------------------------------------ #
    # Speakers (from the run's val filelist, mapped like the dataset does)
    # ------------------------------------------------------------------ #
    def speakers(self, model_id: Optional[str],
                 variant_id: Optional[str]) -> List[Dict[str, Any]]:
        model, variant = self.variant(model_id, variant_id)
        cache_key = variant.config
        with self._probe_lock:
            if cache_key in self._speaker_cache:
                return self._speaker_cache[cache_key]
        cfg = self._config(variant.config)
        val = _abs(cfg.train.get("val_filelist"))
        counts: Dict[int, int] = {}
        names: Dict[int, Dict[str, int]] = {}
        if val and os.path.isfile(val):
            try:
                from dataset_codes import CodeDataset

                ds = CodeDataset(
                    val, is_train=False, max_frames=None,
                    max_phone_length=cfg.train.get("max_phone_length"),
                    speaker_map=cfg.train.get("speaker_map"))
                for item in ds.items:
                    sid = int(item["sid"])
                    counts[sid] = counts.get(sid, 0) + 1
                    label = _speaker_label(item.get("phone_ids"))
                    slot = names.setdefault(sid, {})
                    slot[label] = slot.get(label, 0) + 1
            except Exception as exc:
                print(f"web_client: could not read speakers from {val}: {exc}")
        out = []
        for sid, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            best = (max(names.get(sid, {"": 0}).items(), key=lambda kv: kv[1])[0]
                    if names.get(sid) else "")
            out.append({"id": sid, "label": best or f"Speaker {sid}",
                        "count": count})
        with self._probe_lock:
            self._speaker_cache[cache_key] = out
        return out

    # ------------------------------------------------------------------ #
    # Loading / eviction
    # ------------------------------------------------------------------ #
    def is_loaded(self, model_id: str, variant_id: str) -> bool:
        with self._slots_lock:
            return slot_key(model_id, variant_id) in self._slots

    def loaded_keys(self) -> List[str]:
        with self._slots_lock:
            return list(self._slots)

    def ensure_loaded(self, model_id: Optional[str], variant_id: Optional[str],
                      stage=None) -> LoadedSlot:
        model, variant = self.variant(model_id, variant_id)
        key = slot_key(model.id, variant.id)
        with self._slots_lock:
            slot = self._slots.get(key)
            if slot is not None:
                self._slots.move_to_end(key)
                return slot

        with self._load_lock:
            with self._slots_lock:
                slot = self._slots.get(key)
                if slot is not None:
                    self._slots.move_to_end(key)
                    return slot
                # Evict *before* building: two transformers + two vocoders can
                # exceed a 16 GB card, and the load is cheap enough to redo.
                while len(self._slots) >= self.max_loaded:
                    old_key, old = next(iter(self._slots.items()))
                    self._slots.pop(old_key)
                    print(f"web_client: evicted {old_key} "
                          f"(max_loaded_models={self.max_loaded})")
                    del old
                    if str(self.device).startswith("cuda"):
                        torch.cuda.empty_cache()
            if stage:
                stage(f"loading {model.label} / {variant.label} "
                      f"(first use of this model)")
            print(f"web_client: loading {key} from {variant.checkpoint}")
            cfg = self._config(variant.config)
            cfg.exp_name = str(cfg.get("exp_name", "web_client"))
            repl = StudioREPL(
                cfg, variant.checkpoint, self.device,
                generator_checkpoint=variant.generator_checkpoint,
                audio=bool(variant.generator_checkpoint),
                out_dir=self.out_dir,
                reader=KeyReader(force_line_mode=True),
                hint_mode="null",
                precision=self.precision,
            )
            try:
                repl.load_models()
            except SystemExit as exc:      # layout guard inside the REPL
                raise EngineError(str(exc)) from None
            except Exception as exc:
                raise EngineError(f"could not load {key}: {exc}") from exc

            slot = LoadedSlot(model_id=model.id, variant_id=variant.id, key=key,
                              repl=repl)
            slot.caps = self._runtime_caps(model, variant, repl)
            with self._slots_lock:
                self._slots[key] = slot
                self._slots.move_to_end(key)
            return slot

    def _runtime_caps(self, model: ModelEntry, variant: Variant,
                      repl: StudioREPL) -> Dict[str, Any]:
        caps = self.probe_caps(model.id, variant.id)
        net = repl.net
        n_int, n_reg, n_pitch, n_slot = repl._hint_widths()
        caps.update({
            "loaded": True,
            "fim": bool(net.fim),
            "eos": bool(net.eos_token),
            "fim_lookahead": int(getattr(net, "fim_lookahead", 0) or 0),
            "region_cond": bool(net.region_cond),
            "length_cond": bool(net.length_cond),
            "rope": bool(getattr(net, "rope", False)),
            "hints": bool(getattr(net, "hints", False)),
            "decoder_hints": bool(getattr(net, "decoder_hints", False)),
            "hint_monotone": bool(getattr(net, "hint_monotone", False)),
            "word_sem": bool(getattr(net, "hint_word_sem", False)),
            "word_dim": int(getattr(net, "hint_word_dim", 0) or 0),
            "pitch_mode": str(getattr(net, "hint_pitch_mode", "table")),
            "pitch_anchors": int(getattr(net, "hint_pitch_anchors", 0) or 0),
            "n_intensity": n_int,
            "n_register": n_reg,
            "n_pitch": n_pitch,
            "n_slot": n_slot,
            "lora": bool(getattr(net, "lora_enabled", False)),
            "lora_rank": (int(net.lora_config["rank"])
                          if getattr(net, "lora_enabled", False) else 0),
            "frame_rate": repl.frame_rate,
            "max_frames": repl._clamp_frames(int(caps.get("max_frames", 512))),
            "audio": bool(repl.audio and not repl._net_g_failed),
            "description": net.describe(),
        })
        return caps

    def unload_all(self) -> None:
        with self._slots_lock:
            self._slots.clear()
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Text / hints
    # ------------------------------------------------------------------ #
    def _phonemize(self, text: str):
        """Phonemize outside any model (so /api/preview works pre-load)."""
        if not text or not str(text).strip():
            raise RequestError("text is empty")
        fe = self._extractor()
        try:
            phones, ids, word_index, word_texts = phonemize_with_arpabet(fe, text)
        except ValueError as exc:
            raise RequestError(str(exc)) from None
        except Exception as exc:
            raise EngineError(f"phonemization failed: {exc}") from exc
        if not ids:
            raise RequestError("that text produced no phonemes (try adding "
                               "words, or check your {ARPAbet} groups)")
        return phones, [int(i) for i in ids], list(word_index), list(word_texts)

    def _extractor(self):
        if self._global_extractor is None:
            from features import FeatureExtractor

            # The extractor only needs a config for the phoneme tokenizer; use
            # the default model's config so the vocabulary agrees.
            model = self.model(None)
            cfg = self._config(model.variants[0].config)
            self._global_extractor = FeatureExtractor(cfg, device="cpu",
                                                      load_qwen=False)
        return self._global_extractor

    @staticmethod
    def _apply_hints(repl: StudioREPL, params: Dict[str, Any]) -> None:
        """Push request hint controls onto the REPL before ``_hints_for``."""
        net = repl.net
        n_int, n_reg, _n_pitch, n_slot = repl._hint_widths()
        has_hints = bool(getattr(net, "hints", False))

        # Arbitrary text has no `.cond`, so the honest base state is the
        # trained null state (`cond` == null here); `omit` stays available as
        # the historical ablation.
        mode = str(params.get("base", "null") or "null").lower()
        repl.hint_mode = "omit" if mode == "omit" else "null"

        def level(value, width, name):
            if value is None or width <= 0:
                return None
            try:
                value = int(value)
            except (TypeError, ValueError):
                raise RequestError(f"{name} must be an integer level")
            if value < 0 or value > width:
                raise RequestError(f"{name} must be between 0 and {width}")
            return value if value > 0 else None

        repl.hint_intensity = level(params.get("intensity"), n_int, "intensity")
        repl.hint_register = level(params.get("register"), n_reg, "register")

        if params.get("pitch_shift") is not None:
            try:
                shift = float(params["pitch_shift"])
            except (TypeError, ValueError):
                raise RequestError("pitch_shift must be a number of semitones")
            if not -24.0 <= shift <= 24.0:
                raise RequestError("pitch_shift must be within +/-24 semitones")
            repl.hint_pitch_st = shift
        else:
            repl.hint_pitch_st = 0.0

        if params.get("slot_shift") is not None:
            try:
                shift = int(params["slot_shift"])
            except (TypeError, ValueError):
                raise RequestError("slot_shift must be an integer")
            if not -4 <= shift <= 4:
                raise RequestError("slot_shift must be within -4..+4")
            repl.hint_slot_shift = shift
        else:
            repl.hint_slot_shift = 0

        word = str(params.get("word_semantics", "null") or "null").lower()
        if not has_hints or not getattr(net, "hint_word_sem", False):
            word = "null"
        if word not in ("null", "text"):
            raise RequestError("word_semantics must be 'null' or 'text'")
        repl.hint_word_mode = word
        repl.hint_word_text = None

    @staticmethod
    def _hint_params(params: Dict[str, Any]) -> Dict[str, Any]:
        keys = ("base", "intensity", "register", "pitch_shift", "slot_shift",
                "word_semantics")
        return {k: params.get(k) for k in keys}

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def generate(self, model_id: Optional[str], variant_id: Optional[str],
                 req: Dict[str, Any], stage=None) -> Render:
        text = str(req.get("text") or "").strip()
        if not text:
            raise RequestError("text is empty")
        max_chars = int(self.server_cfg.get("max_text_chars", 2000) or 2000)
        if len(text) > max_chars:
            raise RequestError(f"text is too long ({max_chars} characters max)")

        slot = self.ensure_loaded(model_id, variant_id, stage=stage)
        with slot.lock:
            repl = slot.repl
            caps = slot.caps
            if stage:
                stage("phonemizing")
            phones, ids, word_index, word_texts = self._phonemize(text)

            notes: List[str] = []
            max_phone = int(repl.config.train.get("max_phone_length", 512)
                            or 512)
            if len(ids) > max_phone:
                ids = ids[:max_phone]
                notes.append(f"phoneme sequence truncated to {max_phone} ids "
                             f"(the model's training window)")
                # Keep word alignment only for the kept prefix.
                word_index = word_index[:max_phone]
                phones = repl._extractor().tokenizer_g2p.decode(ids)

            sid = _req_int(req.get("sid"), "sid", default=0, lo=0,
                           hi=int(caps.get("n_speakers", 4096)) - 1)
            frames = _req_int(req.get("frames"), "frames", default=200,
                              lo=1, hi=int(caps.get("max_frames", 512)))
            temperature = _req_float(req.get("temperature"), "temperature",
                                     default=0.9, lo=0.01, hi=5.0)
            top_k = _req_topk(req.get("top_k"))
            seed = _req_seed(req.get("seed"))

            repl.text = text
            repl.phone_ids = list(ids)
            repl.word_index = word_index
            repl.word_texts = word_texts
            repl._base_hints = None
            repl.sid = sid
            repl.temperature = temperature
            repl.top_k = top_k
            self._apply_hints(repl, req)
            hints = self._hint_params(req)
            frames = repl._clamp_frames(frames)

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            phone = torch.tensor(ids, dtype=torch.long,
                                 device=repl.device).unsqueeze(0)
            plen = torch.tensor([len(ids)], dtype=torch.long,
                                device=repl.device)
            sid_t = torch.tensor([sid], dtype=torch.long, device=repl.device)

            if stage:
                stage(f"sampling {frames} frames of codes")
            t0 = time.perf_counter()
            try:
                with repl._autocast(), torch.no_grad():
                    codes, lens = repl.net.generate(
                        phone, plen, sid_t, max_frames=frames,
                        temperature=temperature, top_k=top_k,
                        return_lengths=True, hints=repl._hints_for(len(ids)))
            except Exception as exc:
                raise EngineError(f"sampling failed: {exc}") from exc
            gen_ms = (time.perf_counter() - t0) * 1e3
            codes = codes[0].detach()

            if stage:
                stage("decoding codes to audio")
            t1 = time.perf_counter()
            audio, f0 = self._decode(repl, codes, sid)
            decode_ms = (time.perf_counter() - t1) * 1e3
            if audio is None:
                notes.append("audio decode unavailable (no stage-1 "
                             "generator checkpoint); codes are still valid")

            render = self._make_render(
                slot, kind="generate", text=text, sid=sid, phone_ids=ids,
                phones=phones, word_index=word_index, word_texts=word_texts,
                frames=int(codes.shape[-1]), temperature=temperature,
                top_k=top_k, seed=seed, hints_state=repl.hint_state(),
                hints_params=hints, codes=codes,
                eos_length=(int(lens[0]) if repl.net.eos_token else None),
                audio=audio, f0=f0,
                timings={"generate_ms": gen_ms, "decode_ms": decode_ms},
                notes=notes)
            return render

    def infill(self, model_id: Optional[str], variant_id: Optional[str],
               source: Render, req: Dict[str, Any], stage=None) -> Render:
        if source is None:
            raise RequestError("source generation not found (or expired)")
        model, variant = self.variant(model_id, variant_id)
        if (source.model_id != model.id
                or source.variant_id != variant.id):
            raise RequestError(
                "the source clip came from a different model/variant; "
                "regenerate it with the selected model first")
        slot = self.ensure_loaded(model.id, variant.id, stage=stage)
        with slot.lock:
            repl = slot.repl
            if not bool(repl.net.fim):
                raise RequestError(
                    "this checkpoint has no FIM support (its SUF/MID sentinel "
                    "classes were never trained), so a span cannot be "
                    "regenerated -- pick a FIM model")

            codes = source.codes.detach().to(repl.device)
            n = int(codes.shape[-1])
            if n < 3:
                raise RequestError(f"the source clip is only {n} frame(s) "
                                   f"long; nothing to fill")

            temp = (_req_float(req.get("temperature"), "temperature",
                               default=source.temperature, lo=0.01, hi=5.0))
            top_k = (_req_topk(req.get("top_k"))
                     if req.get("top_k") is not None else source.top_k)
            seed = _req_seed(req.get("seed"))
            best_of = _req_int(req.get("best_of"), "best_of", default=1,
                               lo=1, hi=8)
            rng = random.Random(seed)
            random.seed(seed)

            tcfg = repl.config.transformer
            min_mid = max(1, int(tcfg.get("fim_min_frames", 1) or 1))
            mid_frac = float(tcfg.get("fim_mid_frac", 0.5) or 0.5)
            lo = min(min_mid, n - 2)
            hi = max(lo, min(n - 2, int(round(mid_frac * n))))
            span = req.get("frames")
            span = (rng.randint(lo, hi) if span in (None, "")
                    else _req_int(span, "frames", default=lo, lo=1, hi=n - 1))
            span = max(1, min(int(span), n - 1))
            start = req.get("start")
            if start in (None, ""):
                lo_s, hi_s = ((1, n - span - 1) if n - span >= 2
                              else (0, n - span))
                start = rng.randint(lo_s, hi_s)
            start = max(0, min(int(start), n - span))

            # Restore the source's conditioning exactly (its phone ids, not a
            # re-phonemization, so the codes and the text cannot drift apart).
            repl.text = source.text
            repl.phone_ids = list(source.phone_ids)
            repl.word_index = source.word_index
            repl.word_texts = list(source.word_texts)
            repl._base_hints = None
            repl.sid = int(source.sid)
            repl.temperature = temp
            repl.top_k = top_k
            hint_params = dict(source.hints_params)
            for key in ("intensity", "register", "pitch_shift", "slot_shift",
                        "word_semantics", "base"):
                if req.get(key) is not None:
                    hint_params[key] = req[key]
            self._apply_hints(repl, hint_params)

            phone = torch.tensor([list(source.phone_ids)], dtype=torch.long,
                                 device=repl.device)
            plen = torch.tensor([len(source.phone_ids)], dtype=torch.long,
                                device=repl.device)
            sid_t = torch.tensor([int(source.sid)], dtype=torch.long,
                                 device=repl.device)
            prefix = codes[:, :start].unsqueeze(0)
            suffix = codes[:, start + span:].unsqueeze(0)

            if stage:
                stage(f"regenerating frames {start}..{start + span - 1} "
                      f"of {n}")
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            t0 = time.perf_counter()
            try:
                with repl._autocast(), torch.no_grad():
                    middle = repl.net.infill(
                        phone, plen, sid_t, prefix, suffix, span,
                        temperature=temp, top_k=top_k,
                        hints=repl._hints_for(len(source.phone_ids)),
                        best_of=best_of)
            except Exception as exc:
                raise EngineError(f"infill failed: {exc}") from exc
            gen_ms = (time.perf_counter() - t0) * 1e3
            middle = middle[0].detach()
            filled = torch.cat([codes[:, :start], middle,
                                codes[:, start + span:]], dim=-1)

            seam_ce = None
            if (int(getattr(repl.net, "fim_lookahead", 0) or 0) > 0
                    and suffix.shape[-1] > 0):
                try:
                    with repl._autocast(), torch.no_grad():
                        seam = repl.net.infill_seam_score(
                            phone, plen, sid_t, prefix, middle.unsqueeze(0),
                            suffix, hints=repl._hints_for(len(source.phone_ids)))
                    seam_ce = float(seam[0])
                except Exception as exc:
                    print(f"web_client: seam score failed: {exc}")

            if stage:
                stage("decoding the filled clip")
            t1 = time.perf_counter()
            audio, f0 = self._decode(repl, filled, source.sid)
            decode_ms = (time.perf_counter() - t1) * 1e3
            notes = []
            if audio is None:
                notes.append("audio decode unavailable (no stage-1 generator)")

            render = self._make_render(
                slot, kind="infill", text=source.text, sid=int(source.sid),
                phone_ids=list(source.phone_ids), phones=source.phones,
                word_index=source.word_index,
                word_texts=list(source.word_texts),
                frames=int(filled.shape[-1]), temperature=temp, top_k=top_k,
                seed=seed, hints_state=repl.hint_state(),
                hints_params=self._hint_params(hint_params), codes=filled,
                eos_length=None, audio=audio, f0=f0,
                timings={"generate_ms": gen_ms, "decode_ms": decode_ms},
                notes=notes, source_job=source.job_id,
                span={"start": int(start), "frames": int(span)},
                seam_ce=seam_ce, best_of=int(best_of))
            return render

    # ------------------------------------------------------------------ #
    # Shared render assembly / decode
    # ------------------------------------------------------------------ #
    def _make_render(self, slot: LoadedSlot, *, kind, text, sid, phone_ids,
                     phones, word_index, word_texts, frames, temperature,
                     top_k, seed, hints_state, hints_params, codes,
                     eos_length, audio, f0, timings, notes, source_job=None,
                     span=None, seam_ce=None, best_of=1) -> Render:
        codes_cpu = codes.detach().to("cpu")
        n_spec = int(slot.repl.net.spec_codebook_num)
        unique = [int(torch.unique(codes_cpu[k]).numel())
                  for k in range(codes_cpu.shape[0])]
        repeat = (float((codes_cpu[:, 1:] == codes_cpu[:, :-1]).float().mean())
                  if codes_cpu.shape[-1] > 1 else 0.0)
        return Render(
            model_id=slot.model_id,
            variant_id=slot.variant_id,
            kind=kind,
            text=text,
            sid=int(sid),
            phone_ids=[int(i) for i in phone_ids],
            phones=phones,
            word_index=list(word_index) if word_index is not None else None,
            word_texts=list(word_texts),
            frames=int(frames),
            frame_rate=float(slot.repl.frame_rate),
            sampling_rate=int(slot.repl.config.data.sampling_rate),
            temperature=float(temperature),
            top_k=top_k,
            seed=int(seed),
            hints_state=hints_state,
            hints_params=hints_params,
            codes=codes_cpu,
            audio=audio,
            f0=f0,
            eos_length=eos_length,
            unique_codes=unique,
            repeat_frac=repeat,
            timings=timings,
            notes=list(notes or []),
            source_job=source_job,
            span=span,
            seam_ce=seam_ce,
            best_of=int(best_of),
        )

    def _decode(self, repl: StudioREPL, codes: torch.Tensor, sid: int):
        """(K, n) codes -> (waveform float32, f0 Hz) or (None, None)."""
        net_g = repl._generator()
        if net_g is None:
            return None, None
        n_spec = int(repl.net.spec_codebook_num)
        codes = codes.to(repl.device)
        sid_t = torch.tensor([int(sid)], dtype=torch.long, device=repl.device)
        try:
            with torch.no_grad():
                audio, f0 = net_g.infer_from_codes(
                    codes[:n_spec].unsqueeze(1),
                    codes[n_spec:].unsqueeze(1),
                    sid_t, return_pitch=True)
        except Exception as exc:
            print(f"web_client: stage-1 decode failed: {exc}")
            return None, None
        wav = audio[0].reshape(-1).float().cpu().numpy()
        pitch = f0[0].reshape(-1).float().cpu().numpy()
        return wav, pitch

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #
    def preview(self, text: str) -> Dict[str, Any]:
        phones, ids, word_index, word_texts = self._phonemize(text)
        return {
            "phones": phones,
            "n_phone_ids": len(ids),
            "words": [w for w in word_texts if w],
            "duration_hint": len(ids) / 12.0,
        }

    def status(self) -> Dict[str, Any]:
        info = {
            "device": self.device,
            "precision": self.precision,
            "loaded": self.loaded_keys(),
            "max_loaded_models": self.max_loaded,
        }
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            dev = torch.device(self.device)
            free, total = torch.cuda.mem_get_info(dev)
            info["gpu"] = {
                "name": torch.cuda.get_device_name(dev),
                "allocated_mb": torch.cuda.memory_allocated(dev) / 2**20,
                "reserved_mb": torch.cuda.memory_reserved(dev) / 2**20,
                "free_mb": free / 2**20,
                "total_mb": total / 2**20,
            }
        return info


# ---------------------------------------------------------------------- #
# Small parsers / helpers
# ---------------------------------------------------------------------- #
def _speaker_label(path: Optional[str]) -> str:
    """``data/.../Yae Miko_English__346238.wav.phone_ids`` -> ``Yae Miko``."""
    if not path:
        return ""
    base = os.path.basename(str(path))
    for suffix in (".phone_ids", ".wav"):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
    name = base.split("__")[0]
    for suffix in ("_English", "_Japanese", "_Chinese", "_Korean"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name.replace("_", " ").strip()


def _req_int(value, name, default=None, lo=None, hi=None) -> int:
    if value is None or value == "":
        if default is None:
            raise RequestError(f"{name} is required")
        value = default
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise RequestError(f"{name} must be an integer") from None
    if lo is not None and value < lo:
        raise RequestError(f"{name} must be >= {lo}")
    if hi is not None and value > hi:
        raise RequestError(f"{name} must be <= {hi}")
    return value


def _req_float(value, name, default=None, lo=None, hi=None) -> float:
    if value is None or value == "":
        if default is None:
            raise RequestError(f"{name} is required")
        value = default
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise RequestError(f"{name} must be a number") from None
    if lo is not None and value < lo:
        raise RequestError(f"{name} must be >= {lo}")
    if hi is not None and value > hi:
        raise RequestError(f"{name} must be <= {hi}")
    return value


def _req_topk(value) -> Optional[int]:
    if value is None or value == "" or value == 0 or value == "none":
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise RequestError("top_k must be an integer or 0 ('off')") from None
    if value < 0:
        raise RequestError("top_k must be >= 0")
    return value if value > 0 else None


def _req_seed(value) -> int:
    if value is None or value == "":
        return random.SystemRandom().randrange(1, 2**31 - 1)
    return _req_int(value, "seed", lo=0, hi=2**31 - 1)
