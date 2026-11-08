"""Dataset / collation for the stage-2 discrete-code transformer.

Reads the filelists produced by ``2_preprocess.py``, whose lines have the
form::

    phone_ids_path|spec_codes_path|pitch_codes_path|sid

with

    phone_ids    (n_phone,)                 long
    spec_codes   (Q_spec,  T_codes)         long
    pitch_codes  (Q_pitch, T_codes)         long

The collator pads to the longest item in the batch and additionally reports
``phone_lengths`` and ``code_lengths`` (number of valid code frames).  The
transformer uses those lengths to build the delay pattern and its loss mask,
so padded entries never contribute.  It also reports ``complete``, a boolean
per item: ``True`` when the (possibly cropped) segment still ends where the
clip ends, so the transformer may put an end-of-utterance target there
(``transformer.eos_token``); random training crops are usually mid-clip and
must not get one.

With ``transformer.fim`` the collator also emits ``fim`` (bool), ``fim_prefix_len``
and ``fim_mid_len`` per item.  ``CodeDataset`` draws them for a fraction
``train.fim_rate`` of *training* items (never validation), and the transformer
reorders those items to ``prefix <SUF> suffix <MID> middle`` at loss time.  A
plain AR item (``fim=False``) is left untouched, so AR and FIM samples share a
batch.
"""

from __future__ import annotations

import random
from typing import List, Optional

import torch
from torch.utils.data import Dataset

import speaker_map as speaker_map_lib


class CodeDataset(Dataset):
    def __init__(
        self,
        filelist: str,
        is_train: bool = True,
        max_frames: Optional[int] = None,
        max_phone_length: Optional[int] = None,
        random_seed: int = 42,
        speaker_map=None,
        fim: bool = False,
        fim_rate: float = 0.0,
        fim_min_frames: int = 8,
        fim_mid_frac: float = 0.5,
    ):
        self.is_train = is_train
        self.max_frames = max_frames
        self.max_phone_length = max_phone_length
        self.rng = random.Random(random_seed)
        # Fill-in-the-middle span sampling (see ``modeling/transformer.py``).
        # Per train item, ``fim_rate`` is the probability of returning a
        # prefix/suffix/middle split instead of the plain AR sample.  Spans are
        # drawn here (not in the model) so the model stays a pure function and
        # the split is reproducible with the dataset seed.
        self.fim = bool(fim)
        self.fim_rate = float(fim_rate)
        self.fim_min_frames = int(fim_min_frames)
        self.fim_mid_frac = float(fim_mid_frac)

        # ``speaker_map`` is the same JSON structure used by 2_preprocess
        # (see speaker_map.py).  Preprocessed filelists no longer contain
        # the audio path, so only its ``sid_map`` (raw id -> canonical id)
        # can be applied here; path-based rules must be applied during
        # preprocessing.  Passed as a path or an already-loaded dict.
        if isinstance(speaker_map, str):
            speaker_map = speaker_map_lib.load_speaker_map(speaker_map)
        self.speaker_map = speaker_map

        self.items: List[dict] = []
        with open(filelist, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) < 4:
                    continue
                sid = int(parts[3])
                if self.speaker_map is not None:
                    resolved = speaker_map_lib.resolve_sid(
                        self.speaker_map, path=None, raw_sid=sid)
                    if resolved is None:
                        continue
                    sid = resolved
                self.items.append(
                    {
                        "phone_ids": parts[0],
                        "spec_codes": parts[1],
                        "pitch_codes": parts[2],
                        "sid": sid,
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def speaker_ids(self) -> List[int]:
        return [it["sid"] for it in self.items]

    def _crop_frames(self, spec: torch.Tensor, pitch: torch.Tensor):
        """Cut the item to ``max_frames`` and report whether its end survived.

        Returns ``(spec, pitch, complete)``.  ``complete`` is ``True`` when the
        kept window still contains the last frame of the clip, i.e. the segment
        is the true end of the utterance rather than a mid-clip random crop.
        The transformer only places an end-of-utterance target on ``complete``
        items (see ``CodeTransformer._append_eos``): an EOS in the middle of a
        clip would teach the model to fall silent after ``max_frames``.
        """
        n = min(spec.shape[1], pitch.shape[1])
        spec, pitch = spec[:, :n], pitch[:, :n]
        complete = True
        if self.max_frames is not None and n > self.max_frames:
            if self.is_train:
                start = self.rng.randint(0, n - self.max_frames)
            else:
                start = 0
            spec = spec[:, start : start + self.max_frames]
            pitch = pitch[:, start : start + self.max_frames]
            complete = start + self.max_frames >= n
        return spec, pitch, complete

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        phone_ids = torch.load(item["phone_ids"], weights_only=True).long().reshape(-1)
        spec = torch.load(item["spec_codes"], weights_only=True).long()
        pitch = torch.load(item["pitch_codes"], weights_only=True).long()
        spec = spec.reshape(spec.shape[0], -1)
        pitch = pitch.reshape(pitch.shape[0], -1)
        spec, pitch, complete = self._crop_frames(spec, pitch)

        if self.max_phone_length is not None and phone_ids.numel() > self.max_phone_length:
            phone_ids = phone_ids[: self.max_phone_length]

        fim_flag, prefix_len, mid_len = self._sample_fim(spec.shape[1])
        return {
            "phone_ids": phone_ids,
            "spec_codes": spec,
            "pitch_codes": pitch,
            "sid": item["sid"],
            "complete": complete,
            "fim": fim_flag,
            "fim_prefix_len": prefix_len,
            "fim_mid_len": mid_len,
        }

    def _sample_fim(self, n_frames: int):
        """Draw a (prefix, middle) split with probability ``fim_rate``.

        Returns ``(fim, prefix_len, mid_len)``; ``(False, 0, 0)`` for a plain AR
        item (and for every validation item: ``is_train`` gates it so the val
        split stays a fixed AR yardstick).  ``mid_len`` is at least
        ``fim_min_frames`` and the span is centred by drawing ``prefix_len``
        uniformly in ``[0, n - mid_len]`` (so the suffix can be empty only when
        the prefix is long, never a degenerate all-middle clip unless the item is
        very short).
        """
        if (not self.is_train or not self.fim or self.fim_rate <= 0.0
                or n_frames <= self.fim_min_frames):
            return False, 0, 0
        if self.rng.random() >= self.fim_rate:
            return False, 0, 0
        min_mid = self.fim_min_frames
        max_mid = max(min_mid, min(n_frames - 1,
                                   int(round(self.fim_mid_frac * n_frames))))
        mid_len = self.rng.randint(min_mid, max_mid)
        mid_len = min(mid_len, n_frames - 1)  # leave >= 1 frame for prefix+suffix
        prefix_len = self.rng.randint(0, n_frames - mid_len)
        return True, prefix_len, mid_len


class CodeCollator:
    def __init__(self, config=None, phone_pad_id: int = 0, code_pad_id: int = 0):
        if config is not None:
            tcfg = getattr(config, "transformer", None)
            if tcfg is not None:
                phone_pad_id = int(getattr(tcfg, "phone_pad_id", phone_pad_id))
        self.phone_pad_id = phone_pad_id
        self.code_pad_id = code_pad_id

    def __call__(self, batch: List[dict]) -> dict:
        max_phone = max(int(x["phone_ids"].numel()) for x in batch)
        max_frames = max(int(x["spec_codes"].shape[1]) for x in batch)

        phone_ids = torch.full(
            (len(batch), max_phone), self.phone_pad_id, dtype=torch.long
        )
        phone_lengths = torch.zeros(len(batch), dtype=torch.long)

        q_spec = batch[0]["spec_codes"].shape[0]
        q_pitch = batch[0]["pitch_codes"].shape[0]
        spec_codes = torch.full(
            (len(batch), q_spec, max_frames), self.code_pad_id, dtype=torch.long
        )
        pitch_codes = torch.full(
            (len(batch), q_pitch, max_frames), self.code_pad_id, dtype=torch.long
        )
        code_lengths = torch.zeros(len(batch), dtype=torch.long)
        sids = torch.zeros(len(batch), dtype=torch.long)
        complete = torch.zeros(len(batch), dtype=torch.bool)
        # FIM span selectors (no-op unless the model has transformer.fim).
        fim = torch.zeros(len(batch), dtype=torch.bool)
        fim_prefix_len = torch.zeros(len(batch), dtype=torch.long)
        fim_mid_len = torch.zeros(len(batch), dtype=torch.long)

        for i, x in enumerate(batch):
            p = x["phone_ids"]
            phone_ids[i, : p.numel()] = p
            phone_lengths[i] = p.numel()

            spec = x["spec_codes"]
            pitch = x["pitch_codes"]
            n = min(spec.shape[1], pitch.shape[1], max_frames)
            spec_codes[i, :, :n] = spec[:, :n]
            pitch_codes[i, :, :n] = pitch[:, :n]
            code_lengths[i] = n
            sids[i] = x["sid"]
            complete[i] = bool(x.get("complete", True))
            fim[i] = bool(x.get("fim", False))
            fim_prefix_len[i] = int(x.get("fim_prefix_len", 0))
            fim_mid_len[i] = int(x.get("fim_mid_len", 0))

        return {
            "phone_ids": phone_ids,
            "phone_lengths": phone_lengths,
            "sid": sids,
            "spec_codes": spec_codes,
            "pitch_codes": pitch_codes,
            "code_lengths": code_lengths,
            # True where this segment ends the clip -> the transformer adds an
            # EOS target there (no-op unless transformer.eos_token is set).
            "complete": complete,
            # Fill-in-the-middle: which items to reorder and their span lengths
            # (no-op unless transformer.fim; AR and FIM items can share a batch).
            "fim": fim,
            "fim_prefix_len": fim_prefix_len,
            "fim_mid_len": fim_mid_len,
        }
