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
so padded entries never contribute.
"""

from __future__ import annotations

import random
from typing import List, Optional

import torch
from torch.utils.data import Dataset


class CodeDataset(Dataset):
    def __init__(
        self,
        filelist: str,
        is_train: bool = True,
        max_frames: Optional[int] = None,
        max_phone_length: Optional[int] = None,
        random_seed: int = 42,
    ):
        self.is_train = is_train
        self.max_frames = max_frames
        self.max_phone_length = max_phone_length
        self.rng = random.Random(random_seed)

        self.items: List[dict] = []
        with open(filelist, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) < 4:
                    continue
                self.items.append(
                    {
                        "phone_ids": parts[0],
                        "spec_codes": parts[1],
                        "pitch_codes": parts[2],
                        "sid": int(parts[3]),
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def _crop_frames(self, spec: torch.Tensor, pitch: torch.Tensor):
        n = min(spec.shape[1], pitch.shape[1])
        spec, pitch = spec[:, :n], pitch[:, :n]
        if self.max_frames is not None and n > self.max_frames:
            if self.is_train:
                start = self.rng.randint(0, n - self.max_frames)
            else:
                start = 0
            spec = spec[:, start : start + self.max_frames]
            pitch = pitch[:, start : start + self.max_frames]
        return spec, pitch

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        phone_ids = torch.load(item["phone_ids"], weights_only=True).long().reshape(-1)
        spec = torch.load(item["spec_codes"], weights_only=True).long()
        pitch = torch.load(item["pitch_codes"], weights_only=True).long()
        spec = spec.reshape(spec.shape[0], -1)
        pitch = pitch.reshape(pitch.shape[0], -1)
        spec, pitch = self._crop_frames(spec, pitch)

        if self.max_phone_length is not None and phone_ids.numel() > self.max_phone_length:
            phone_ids = phone_ids[: self.max_phone_length]

        return {
            "phone_ids": phone_ids,
            "spec_codes": spec,
            "pitch_codes": pitch,
            "sid": item["sid"],
        }


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

        return {
            "phone_ids": phone_ids,
            "phone_lengths": phone_lengths,
            "sid": sids,
            "spec_codes": spec_codes,
            "pitch_codes": pitch_codes,
            "code_lengths": code_lengths,
        }
