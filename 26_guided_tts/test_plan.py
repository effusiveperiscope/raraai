"""CPU guard for ``plan.py``: build a plan from a synthetic clip.

Run ``python test_plan.py`` from ``26_guided_tts``.  No checkpoints, no GPU,
no corpus: the conditions that go wrong silently in real data (equal word
starts, edge silence folded into a slot, stale ``.cond``) are constructed
explicitly and asserted.
"""

from __future__ import annotations

import numpy as np
import torch

from plan import (
    CODE_FPS, F0_REF_HZ, Plan, PlanError, plan_from_cond,
    speech_mask_from_spec,
)


def _cond(starts, slots, phone_word, anchors, register=0.0, voiced=None):
    starts = np.asarray(starts, dtype=np.int32)
    slots = np.asarray(slots, dtype=np.int32)
    phone_word = np.asarray(phone_word, dtype=np.int16)
    W = starts.size
    anchors = np.asarray(anchors, dtype=np.float32)          # (W, A, 2)
    voiced = np.ones(W, dtype=bool) if voiced is None else np.asarray(voiced)
    return {
        "n_phone": int(phone_word.size),
        "phone_word_idx": torch.from_numpy(phone_word),
        "word_start_f": torch.from_numpy(starts),
        "word_slot_f": torch.from_numpy(slots),
        "pitch_word_voiced": torch.from_numpy(voiced),
        "pitch_word_st": torch.zeros(W, dtype=torch.float16),
        "pitch_anchors_st": torch.from_numpy(anchors[..., 1]).to(torch.float16),
        "pitch_anchors_pos": torch.from_numpy(anchors[..., 0]).to(torch.float16),
        "pitch_anchor_n": torch.full((W,), anchors.shape[1], dtype=torch.int8),
        "register_st": torch.tensor([register], dtype=torch.float32),
        "intensity": torch.tensor([5.0], dtype=torch.float32),
    }


def _spec(n_code, speech):
    """(4T, 769) magnitude spec: 1.0 where speech, the floor elsewhere."""
    speech = np.asarray(speech, dtype=bool)
    assert speech.size == n_code
    a = np.full((4 * n_code, 769), 1e-3, dtype=np.float32)
    for t, v in enumerate(speech):
        if v:
            a[4 * t:4 * t + 4] = 1.0
    return a


def test_partition_and_silence():
    # two words: [0, 8) and [8, 20); silence in frames 0-1 and 18-19
    speech = np.array([False, False] + [True] * 16 + [False, False])
    cond = _cond(
        starts=[0, 8], slots=[8, 12], phone_word=[0, 0, 0, 0, 1, 1, 1, 1],
        anchors=[
            [[0.0, 0.0], [1.0, 12.0]],     # word 0: rise an octave
            [[0.0, 0.0], [1.0, 0.0]],      # word 1: flat
        ],
    )
    plan = plan_from_cond(cond, np.zeros(8), n_frames=20, spec=_spec(20, speech))
    plan.check()

    assert plan.lead_sil == 2, plan.lead_sil
    assert plan.word_dur.tolist() == [6, 10], plan.word_dur.tolist()
    assert plan.pause_after.tolist() == [0, 2], plan.pause_after.tolist()
    assert plan.word_start.tolist() == [2, 8]
    total = plan.lead_sil + plan.word_dur.sum() + plan.pause_after.sum()
    assert total == 20
    assert plan.phone_dur.sum() == 4 * plan.word_dur.sum() == 64
    assert plan.phone_dur.min() >= 1

    # segments are chronological and cover exactly T frames
    segs = list(plan.segments())
    assert segs[0] == ("sil", 2)
    assert segs[-1] == ("sil", 2)
    assert sum(n for *_, n in segs) == 20

    # pitch: median 100 Hz (register 0), word 0 rises 100 -> 200 Hz
    hz = plan.pitch_hz()
    assert hz.shape == (80,)
    assert abs(hz[8] - 100.0) < 3.0, hz[8]        # first frame of word 0
    assert abs(hz[31] - 200.0) < 8.0, hz[31]      # last frame of word 0
    assert abs(hz[40] - 100.0) < 1.0, hz[40]      # word 1 flat
    assert hz[0] == 0 and hz[79] == 0             # silence unvoiced
    assert plan.voiced[8] and not plan.voiced[0]
    assert abs(F0_REF_HZ * 2 ** (plan.register_st / 12) - 100.0) < 1e-6


def test_equal_starts_are_grouped():
    # words 0 and 1 share a start; the run spans frames 4..12 and must be
    # split (phone-count weighted) without overlapping the next word.
    speech = np.zeros(20, dtype=bool); speech[2:18] = True
    cond = _cond(
        starts=[0, 0, 10], slots=[1, 10, 10],
        phone_word=[0, 0, 1, 1, 2],
        anchors=[[[0.5, 0.0]]] * 3,
    )
    plan = plan_from_cond(cond, np.zeros(5), n_frames=20, spec=_spec(20, speech))
    plan.check()
    assert plan.grouped_words == 1
    assert plan.word_dur[:2].sum() == 8, plan.word_dur[:2]     # frames 2..10
    assert plan.word_start[2] == 10
    assert plan.word_dur.min() >= 1


def test_missing_spec_keeps_slots():
    cond = _cond(starts=[0, 6], slots=[6, 6], phone_word=[0, 0, 1, 1],
                 anchors=[[[0.5, 0.0]], [[0.5, 2.0]]])
    plan = plan_from_cond(cond, np.zeros(4), n_frames=12, spec=None)
    plan.check()
    assert not plan.has_spec
    assert plan.lead_sil == 0
    assert plan.pause_after.tolist() == [0, 0]
    assert plan.word_dur.tolist() == [6, 6]


def test_stale_and_degenerate_raise():
    cond = _cond(starts=[0, 6], slots=[6, 6], phone_word=[0, 0, 1, 1],
                 anchors=[[[0.5, 0.0]], [[0.5, 0.0]]])
    bad = dict(cond, n_phone=99)
    try:
        plan_from_cond(bad, np.zeros(4), n_frames=12, spec=None)
    except PlanError:
        pass
    else:
        raise AssertionError("stale n_phone was accepted")

    # two words forced into a single frame: cannot give both one frame
    cond2 = _cond(starts=[0, 0, 1], slots=[1, 1, 9],
                  phone_word=[0, 1, 2],
                  anchors=[[[0.5, 0.0]]] * 3)
    try:
        plan_from_cond(cond2, np.zeros(3), n_frames=10, spec=None)
    except PlanError:
        pass
    else:
        raise AssertionError("zero-length word was accepted")


def test_speech_mask():
    # 3 code frames: speech, silence, speech
    a = _spec(3, [True, False, True])
    m = speech_mask_from_spec(a, 3)
    assert m.tolist() == [True, False, True], m.tolist()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("plan tests passed")
