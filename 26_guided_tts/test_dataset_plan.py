"""CPU guard for ``dataset_plan.py`` on a synthetic 2-clip corpus.

Run ``python test_dataset_plan.py``.  No real data, no GPU.  It checks the two
things that would silently corrupt training: a word-aligned crop still
partitions its frame axis, and a cached plan reloads identical to the freshly
derived one.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from dataset_plan import PlanDataset, collate_plans, cache_path
from plan import load_plan, load_plan_cache


def _write_clip(root: Path, name: str, sid: int, n_frames: int = 12):
    """Two words of 2 phones each; frames 0-1 and the tail are silence."""
    phone = np.array([1, 2, 3, 4], dtype=np.int64)
    pwi = np.array([0, 0, 1, 1], dtype=np.int16)
    W = 2
    starts = np.array([0, 6], dtype=np.int32)
    slots = np.array([6, n_frames - 6], dtype=np.int32)
    T = n_frames
    spec_codes = torch.randint(0, 64, (8, T), dtype=torch.long)
    pitch_codes = torch.randint(0, 128, (2, T), dtype=torch.long)
    cond = {
        "n_phone": 4,
        "phone_word_idx": torch.from_numpy(pwi),
        "word_start_f": torch.from_numpy(starts),
        "word_slot_f": torch.from_numpy(slots),
        "pitch_word_voiced": torch.ones(W, dtype=torch.bool),
        "pitch_word_st": torch.zeros(W, dtype=torch.float16),
        "pitch_anchors_st": torch.zeros(W, 2, dtype=torch.float16),
        "pitch_anchors_pos": torch.tensor([[0.0, 1.0], [0.0, 1.0]],
                                          dtype=torch.float16),
        "pitch_anchor_n": torch.full((W,), 2, dtype=torch.int8),
        "register_st": torch.tensor([0.0]),
        "intensity": torch.tensor([5.0]),
    }
    # 100 Hz spec/f0: silent in frames 0-1 and from 10 on
    speech = np.zeros(T, dtype=bool)
    speech[2:10] = True
    spec = np.full((4 * T, 769), 1e-3, dtype=np.float32)
    f0 = np.zeros(4 * T, dtype=np.float32)
    for t in range(T):
        if speech[t]:
            spec[4 * t:4 * t + 4] = 1.0
            f0[4 * t:4 * t + 4] = 120.0
    base = root / name
    torch.save(torch.from_numpy(phone), str(base) + ".phone_ids")
    torch.save(spec_codes, str(base) + ".spec_codes")
    torch.save(pitch_codes, str(base) + ".pitch_codes")
    torch.save(cond, str(base) + ".cond")
    torch.save(torch.from_numpy(spec), str(base) + ".spec")
    torch.save(torch.from_numpy(f0), str(base) + ".f0")
    return base, phone, cond


def test_dataset_and_cache():
    tmp = Path(tempfile.mkdtemp(prefix="plan_ds_"))
    try:
        b1, phone, cond = _write_clip(tmp, "a", sid=7)
        b2, _, _ = _write_clip(tmp, "b", sid=7, n_frames=16)
        fl = tmp / "list.txt"
        fl.write_text(
            f"{b1}.phone_ids|{b1}.spec_codes|{b1}.pitch_codes|7\n"
            f"{b2}.phone_ids|{b2}.spec_codes|{b2}.pitch_codes|7\n")
        cache = tmp / "plans"

        ds = PlanDataset(fl, max_frames=None, train=False, cache_dir=cache,
                         write_cache=True, verbose=False)
        assert len(ds) == 2
        item = ds[0]
        n = int(item["n_frames"])
        assert n == 12
        assert item["spec_codes"].shape == (8, n)
        assert item["pitch_codes"].shape == (2, n)
        assert item["frame_feats"].shape == (n, 6)
        assert int(item["phone_dur"].sum()) == 4 * int(item["word_dur"].sum())
        assert (int(item["lead_sil"]) + int(item["word_dur"].sum())
                + int(item["pause_after"].sum())) == n
        # f0 is present and voiced only where the plan is
        assert item["f0"].shape == (4 * n,)
        assert bool(item["has_f0"])
        assert item["f0_valid"].any()

        # cache exists and reloads to the same plan
        c = cache_path(cache, str(b1) + ".phone_ids")
        assert c.exists(), c
        ids = torch.load(str(b1) + ".phone_ids", weights_only=False).reshape(-1).numpy()
        a = load_plan_cache(c, ids, speaker=7)
        b = load_plan(str(b1) + ".phone_ids", speaker=7)
        for k in ("phone_dur", "word_start", "word_dur", "pause_after",
                  "lead_sil", "n_frames"):
            assert np.array_equal(np.asarray(getattr(a, k)),
                                  np.asarray(getattr(b, k))), k
        assert np.allclose(a.pitch_st, b.pitch_st, atol=5e-2)

        # touch every item so the lazy cache build covers both clips
        for i in range(len(ds)):
            ds[i]

        # require_cache skips nothing here, and crops stay word-aligned
        ds2 = PlanDataset(fl, max_frames=5, train=False, cache_dir=cache,
                          require_cache=True, verbose=False)
        for i in range(len(ds2)):
            x = ds2[i]
            n = int(x["n_frames"])
            assert x["spec_codes"].shape == (8, n)
            assert (int(x["lead_sil"]) + int(x["word_dur"].sum())
                    + int(x["pause_after"].sum())) == n

        batch = collate_plans([ds2[0], ds2[1]])
        assert batch["spec_codes"].shape[0] == 2
        assert batch["frame_mask"].sum(1).tolist() == [
            int(ds2[0]["n_frames"]), int(ds2[1]["n_frames"])]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_dataset_and_cache()
    print("ok  test_dataset_and_cache")
    print("dataset tests passed")
