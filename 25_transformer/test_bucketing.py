"""CPU smoke test for the two v2.3 throughput/augmentation additions.

* ``dataset_codes.LengthBucketBatchSampler`` -- coverage, batch sizes,
  ``__len__`` agreement, per-epoch determinism, and the actual win: mean batch
  width well below a uniformly shuffled baseline.
* ``CodeDataset.compute_lengths`` -- values are the post-``max_frames`` width and
  the ``<filelist>.lengths.npz`` cache round-trips and invalidates when the
  clamp changes.
* ``transformer.input_dropout`` -- training-only, no parameters / ``state_dict``
  change, eval output bit-identical, and when everything is masked the decoder
  logits stop depending on the code values of the masked columns while still
  depending on the (never masked) start column.

Runs entirely on CPU and needs no real corpus (it writes a 4-clip fake one into
a temp dir), so it is safe next to a GPU training job.

Usage: ``python test_bucketing.py`` (exits non-zero on failure).
"""

from __future__ import annotations

import math
import os
import random
import tempfile

import torch
from omegaconf import OmegaConf

from dataset_codes import CodeDataset, LengthBucketBatchSampler
from modeling.transformer import CodeTransformer


def check(cond, msg):
    if not cond:
        raise SystemExit(f"FAIL: {msg}")


# --------------------------------------------------------------------------- #
# 1. LengthBucketBatchSampler
# --------------------------------------------------------------------------- #
def test_sampler():
    rng = random.Random(0)
    # Corpus-like lengths: many short clips, a long tail (see AGENTS.md v2.2).
    lengths = [max(8, min(512, int(rng.lognormvariate(4.6, 0.7))))
               for _ in range(4000)]
    bs, pool = 16, 8
    s = LengthBucketBatchSampler(lengths, bs, pool_batches=pool, drop_last=True)

    batches = list(iter(s))
    check(len(batches) == len(s),
          f"__len__ {len(s)} != {len(batches)} yielded batches")
    check(all(len(b) == bs for b in batches), "a batch has the wrong size")
    flat = [i for b in batches for i in b]
    check(len(flat) == len(set(flat)), "an index was emitted twice in one epoch")
    check(set(flat) <= set(range(len(lengths))), "index out of range")
    # drop_last covers all whole pools, i.e. everything but the pool tail.
    covered = len(batches) * bs
    check(len(lengths) - covered < bs * pool,
          f"dropped {len(lengths) - covered} items, more than one pool")

    def mean_batch_max(order_batches):
        return sum(max(lengths[i] for i in b) for b in order_batches) / len(order_batches)

    # Baseline: the same batches taken from a uniformly shuffled permutation.
    rnd = list(range(len(lengths)))
    random.Random(1).shuffle(rnd)
    baseline = [rnd[i:i + bs] for i in range(0, len(rnd) - bs + 1, bs)]
    bucketed_t = mean_batch_max(batches)
    random_t = mean_batch_max(baseline)
    check(bucketed_t < 0.75 * random_t,
          f"bucketing barely helped: {bucketed_t:.0f} vs {random_t:.0f}")
    print(f"  mean batch width: random {random_t:.0f} -> bucketed {bucketed_t:.0f} "
          f"frames ({100 * (1 - bucketed_t / random_t):.0f} % less)")

    # Reproducible per epoch, different across epochs (a fresh sampler with the
    # same seed must reproduce epoch 0 exactly).
    first = list(iter(LengthBucketBatchSampler(lengths, bs, pool)))
    check(first == batches, "same seed produced a different first epoch")
    second = list(iter(s))                      # advances s.epoch
    check(second != batches, "epoch 1 repeated epoch 0")
    s.set_epoch(0)
    check(list(iter(s)) == batches, "set_epoch(0) did not reproduce epoch 0")

    # drop_last=False covers everything, possibly with one short batch.
    s2 = LengthBucketBatchSampler(lengths, bs, pool, drop_last=False)
    b2 = list(iter(s2))
    check(len(b2) == len(s2), f"__len__ {len(s2)} != {len(b2)} (drop_last=False)")
    check(sum(len(b) for b in b2) == len(lengths),
          "drop_last=False lost items")
    check(all(len(b) <= bs for b in b2), "drop_last=False oversized batch")

    # A dataset smaller than one pool: only reachable with drop_last=False.
    small = LengthBucketBatchSampler([10, 20, 30], 2, pool_batches=4,
                                     drop_last=False)
    check(len(list(iter(small))) == len(small), "small dataset batch count")
    print(f"  sampler: {len(s)} batches/epoch at bs={bs}, pool={pool}")


# --------------------------------------------------------------------------- #
# 2. CodeDataset.compute_lengths (+ cache)
# --------------------------------------------------------------------------- #
def _write_fake_corpus(root, frames):
    lines = []
    for i, n in enumerate(frames):
        base = os.path.join(root, f"clip{i}.wav")
        torch.save(torch.randint(0, 32, (max(n, 4),)),
                   base + ".phone_ids")
        torch.save(torch.randint(0, 8, (8, n)), base + ".spec_codes")
        torch.save(torch.randint(0, 8, (2, n)), base + ".pitch_codes")
        lines.append(f"{base}.phone_ids|{base}.spec_codes|"
                     f"{base}.pitch_codes|{i}")
    path = os.path.join(root, "train.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def test_compute_lengths():
    with tempfile.TemporaryDirectory() as root:
        frames = [7, 40, 300, 512]
        filelist = _write_fake_corpus(root, frames)
        ds = CodeDataset(filelist, is_train=True, max_frames=256)
        lengths = ds.compute_lengths()
        check(lengths == [7, 40, 256, 256],
              f"lengths {lengths} != post-clamp [7, 40, 256, 256]")
        cache = filelist + ".lengths.npz"
        check(os.path.exists(cache), "length cache was not written")

        # Fresh dataset, same filelist -> cache hit, same values.
        ds2 = CodeDataset(filelist, is_train=True, max_frames=256)
        check(ds2.compute_lengths() == lengths, "cache read returned other lengths")

        # A different clamp must invalidate (the cache records max_frames).
        ds3 = CodeDataset(filelist, is_train=True, max_frames=64)
        check(ds3.compute_lengths() == [7, 40, 64, 64],
              "cache was reused across a different max_frames")
        # ...and no clamp at all is a third layout.
        ds4 = CodeDataset(filelist, is_train=True, max_frames=None)
        check(ds4.compute_lengths() == frames, "max_frames=None not handled")

        # The cache is keyed by tensor path, so a blocklist that drops clips
        # must not invalidate (or misalign) it: the remaining items still get
        # their own lengths.  This is the case the real corpus triggers (456
        # train clips are dropped by the two blocklists).
        block = os.path.join(root, "block.txt")
        with open(block, "w", encoding="utf-8") as f:
            f.write(os.path.join(root, "clip0.wav.phone_ids") + "\n")
        ds5 = CodeDataset(filelist, is_train=True, max_frames=256,
                          alignment_blocklist=block)
        check(len(ds5) == len(frames) - 1, "blocklist did not drop a clip")
        check(ds5.compute_lengths() == [40, 256, 256],
              f"blocklisted corpus lengths {ds5.compute_lengths()} misaligned")
        print("  compute_lengths: values, cache hit/invalidation and blocklist "
              "independence OK")


# --------------------------------------------------------------------------- #
# 3. Decoder input-token dropout
# --------------------------------------------------------------------------- #
def make_config(input_dropout: float):
    return OmegaConf.create({
        "transformer": {
            "hidden_size": 32, "n_heads": 4, "encoder_layers": 1,
            "decoder_layers": 2, "dim_feedforward": 64, "dropout": 0.0,
            "max_positions": 512, "phone_vocab_size": 32, "phone_pad_id": 0,
            "n_speakers": 8, "decoder_speaker_cond": True,
            "memory_norm": "temporal", "residual_scale": True,
            "eos_token": True, "rope": True,
            "codebook_weights": None, "codebook_weight_warmup_epochs": 0,
            "input_dropout": input_dropout,
        },
        "codec": {"codebook_num": 2, "codebook_size": 8},
        "pitch_codec": {"codebook_num": 1, "codebook_size": 8},
    })


def test_input_dropout():
    torch.manual_seed(0)
    off = CodeTransformer(make_config(0.0))
    torch.manual_seed(0)
    on = CodeTransformer(make_config(0.5))
    check(on.input_dropout == 0.5 and off.input_dropout == 0.0,
          "input_dropout not stored on the model")
    check(list(off.state_dict()) == list(on.state_dict()),
          "input_dropout changed the state_dict keys")
    check(all(torch.equal(off.state_dict()[k], on.state_dict()[k])
              for k in off.state_dict()),
          "input_dropout changed the initial weights")

    B, K, L, N = 2, off.codebook_num, 12, 6
    codes = torch.randint(0, 8, (B, K, L)).long()
    memory = torch.randn(B, N, off.hidden_size)
    mmask = torch.ones(B, N, dtype=torch.bool)
    sid = torch.zeros(B, dtype=torch.long)

    def logits(model, c=codes):
        return torch.stack(model.decoder(c, memory, mmask, sid=sid), dim=1)

    off.eval(); on.eval()
    check(torch.equal(logits(off), logits(on)),
          "input_dropout is not a no-op in eval")

    # Training mode: on differs from off, off is deterministic, and a p=0.5
    # decoder is stochastic across calls.
    off.train(); on.train()
    torch.manual_seed(1)
    o1 = logits(off)
    check(torch.equal(o1, logits(off)), "p=0 changed the training output")
    torch.manual_seed(1)
    a = logits(on)
    torch.manual_seed(2)
    b = logits(on)
    check(not torch.equal(a, b), "p=0.5 did not randomise the decoder input")

    # Mask-everything semantics: with p=1 the code values of columns >= 1 must
    # not reach the logits (only the start column does).
    on.decoder.input_dropout = 1.0
    c1 = codes.clone()
    c2 = codes.clone()
    c2[:, :, 1:] = (c2[:, :, 1:] + 1) % 8
    l1, l2 = logits(on, c1), logits(on, c2)
    check(torch.equal(l1[:, :, 1:], l2[:, :, 1:]),
          "masked columns' code values still reach the logits")
    c3 = codes.clone()
    c3[:, :, 0] = (c3[:, :, 0] + 1) % 8
    check(not torch.equal(logits(on, c3), l1),
          "the start column was masked too (its code no longer matters)")
    on.decoder.input_dropout = 0.5

    # Gradients still flow (the knob is applied to the embedding output, not by
    # dropping the graph).
    on.train()
    loss = sum(lg.float().pow(2).mean() for lg in
               on.decoder(codes, memory, mmask, sid=sid))
    loss.backward()
    check(on.decoder.embeds[0].weight.grad is not None
          and torch.isfinite(on.decoder.embeds[0].weight.grad).all(),
          "no finite gradient reached the codebook embeddings")

    # CodeTransformer validates the range.
    for bad in (1.0, -0.1):
        try:
            CodeTransformer(make_config(bad))
        except ValueError:
            pass
        else:
            raise SystemExit(f"FAIL: input_dropout={bad} was accepted")
    print("  input_dropout: eval no-op, train stochastic, start column kept, "
          "gradients OK")


if __name__ == "__main__":
    print("== LengthBucketBatchSampler")
    test_sampler()
    print("== CodeDataset.compute_lengths")
    test_compute_lengths()
    print("== decoder input dropout")
    test_input_dropout()
    print("PASS")
