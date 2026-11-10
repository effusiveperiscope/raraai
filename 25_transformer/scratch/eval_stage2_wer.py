"""WER of stage-2 checkpoints: predict codes -> stage-1 audio -> ASR -> WER.

This is the end-to-end intelligibility check.  ``val_loss`` and code-match
fractions are proxies; this measures the thing that matters: does a speech
recogniser understand the synthesised audio, and does a smaller number mean a
better checkpoint?

Pipeline per val item (one row of the stage-2 filelist):

    phone_ids --> CodeTransformer.generate --> (spec, pitch) codes
              --> frozen stage-1 Generator.infer_from_codes --> 48 kHz wav
              --> Qwen3-ASR (16 kHz) --> text --> WER vs the reference text

Two arms are reported:

``generated``
    codes sampled by the checkpoint under test (the real number).
``reference``
    the item's *ground-truth* codes through the same stage-1 Generator.  This
    does not depend on the transformer at all, so it is computed once and shown
    as the floor: it is the WER contributed by the vocoder + ASR + the
    reference transcript, i.e. the best a perfect code predictor could score.
    A checkpoint whose WER is close to this floor has nothing left to gain
    through better codes.

Reference text comes from the *source* stage-2 filelists (``path|sid|text``),
matched to the preprocessed filelist by basename -- the preprocessed tensors are
named ``<basename>.phone_ids`` etc., so no audio path survives into
``data/*/val_stage2.txt``.

Usage::

    # compare best vs last on 20 val utterances (needs the stage-1 generator)
    python scratch/eval_stage2_wer.py \
        --checkpoints checkpoints_stage2/5090_v0.7_fim2/best-checkpoint.ckpt \
                      checkpoints_stage2/5090_v0.7_fim2/last.ckpt \
        --num_items 20 --generator_checkpoint checkpoints_stage1/base_v0.7/last.ckpt

    # content only: decode the item's own length instead of letting EOS stop
    # early (removes the model's duration errors from the WER)
    python scratch/eval_stage2_wer.py --ignore_eos --temperature 0 ...

    # keep the wavs to listen to (<tag>_<i>.wav, plus reference_<i>.wav)
    ... --save_dir scratch/wer_out

Notes:
  * ``--num_items`` are evenly spaced over the val filelist (deterministic), and
    ``--seed`` is reset before every checkpoint so all of them draw the same
    samples -- the comparison is paired.
  * Two decoding modes matter.  By default the model may emit EOS early, which
    is what it does at inference; that includes its own duration errors as
    word *deletions*.  ``--ignore_eos`` decodes exactly the item's length
    (oracle duration) so only content is measured.  ``--temperature 0`` is
    effectively greedy: deterministic, but it makes EOS fire earlier and can
    collapse into repeats, so compare checkpoints at the same setting.
  * Every WER is reported next to the ``reference`` arm (ground-truth codes
    through the same vocoder + ASR), which is the floor: on 24 genshin val
    utterances it is ~7 %, so most of a ~14 % WER is the transformer.
  * Sample size matters.  On 24 utterances (~440 words) the binomial 95 %
    interval on a single WER is roughly +/- 3 pp, so a sub-1 pp difference is
    not a decision; use ``--num_items 50``+ and read the result as a trend
    alongside the teacher-forced ``val_loss`` (which is far less noisy).
  * Normalisation before scoring: lowercase, apostrophes unified, everything
    except letters/digits/apostrophes dropped.  Numbers are compared as words
    ("48" != "forty-eight"); ``--verbose`` prints every pair so a mismatch like
    that is visible rather than silent.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import sys
from typing import Dict, List, Sequence, Tuple

# fairseq (pulled in by modeling.generator) calls logging.basicConfig(DEBUG) at
# import time; install our handler first so that call becomes a no-op
# (AGENTS.md §6.18).
logging.basicConfig(level=logging.WARNING)

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from commons import load_state_dict_mismatch, load_submodule_prefix  # noqa: E402
from dataset_codes import CodeDataset  # noqa: E402
from modeling.transformer import CodeTransformer  # noqa: E402

# Preprocessed tensors are named by the source basename plus one of these.
TENSOR_SUFFIXES = (".phone_ids", ".spec_codes", ".pitch_codes")
_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)*")


def normalize_text(text: str) -> str:
    """Lowercase, unify apostrophes, keep words only (WER tokenisation)."""
    text = (text or "").lower().replace("\u2019", "'").replace("\u2018", "'")
    return " ".join(_WORD_RE.findall(text))


def edit_counts(ref: Sequence[str], hyp: Sequence[str]) -> Tuple[int, int, int]:
    """Minimum (substitutions, insertions, deletions) turning ``ref`` into ``hyp``.

    Plain Levenshtein DP with a back-pointer table; ties are resolved
    substitution > deletion > insertion so the breakdown is stable.  Returned
    counts give ``WER = (S + D + I) / len(ref)``, i.e. the standard definition
    with ``ref`` as the denominator.
    """
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + cost,   # match / substitute
                           dp[i - 1][j] + 1,          # delete from ref
                           dp[i][j - 1] + 1)          # insert from hyp
    sub = ins = dele = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (
                0 if ref[i - 1] == hyp[j - 1] else 1):
            if ref[i - 1] != hyp[j - 1]:
                sub += 1
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return sub, ins, dele


def wer_summary(pairs: Sequence[Tuple[str, str]]) -> Dict[str, float]:
    """Corpus-level WER over ``(reference, hypothesis)`` text pairs."""
    s = i = d = words = 0
    for ref, hyp in pairs:
        r, h = normalize_text(ref).split(), normalize_text(hyp).split()
        words += len(r)
        a, b, c = edit_counts(r, h)
        s, i, d = s + a, i + b, d + c
    errors = s + i + d
    return {"wer": errors / words if words else float("nan"),
            "sub": s, "ins": i, "del": d, "errors": errors, "words": words,
            "n": len(pairs)}


def item_basename(path: str) -> str:
    """``data/x/foo.wav.phone_ids`` -> ``foo.wav`` (the source basename)."""
    base = os.path.basename(path)
    for suf in TENSOR_SUFFIXES:
        if base.endswith(suf):
            return base[: -len(suf)]
    return base


def load_references(paths: Sequence[str]) -> Dict[str, str]:
    """basename -> transcript from source filelists (``path|sid|text``)."""
    refs: Dict[str, str] = {}
    clashes = 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "longform" in line:
                    continue
                parts = line.split("|")
                if len(parts) != 3:
                    continue
                try:
                    int(parts[1])
                except ValueError:
                    continue
                key = os.path.basename(parts[0].replace("\\", "/"))
                text = parts[2].strip()
                if not text:
                    continue
                if key in refs and refs[key] != text:
                    clashes += 1
                    continue
                refs.setdefault(key, text)
    if clashes:
        print(f"NOTE: {clashes} basename collision(s) with differing text in the "
              f"source filelists; the first one wins.")
    return refs


def build_transformer(config, device):
    net = CodeTransformer(config).to(device)
    net.eval()
    return net


def load_transformer_weights(net, path):
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    sd = ck.get("state_dict", ck)
    prefix = "net." if any(k.startswith("net.") for k in sd) else ""
    load_submodule_prefix(net, prefix, sd, quiet=True)
    return ck.get("epoch"), ck.get("global_step")


def load_generator(checkpoint_path, config, device):
    """Frozen stage-1 ``Generator`` (mirrors ``2_training.py:load_generator``)."""
    from modeling.generator import Generator

    net_g = Generator(hp=config)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(k.startswith("net_g.") for k in state.keys()):
        load_submodule_prefix(net_g, "net_g.", state)
    else:
        load_state_dict_mismatch(net_g, state)
    net_g.to(device)
    net_g.eval()
    return net_g


def chunked(seq: Sequence, size: int):
    for i in range(0, len(seq), max(size, 1)):
        yield seq[i:i + max(size, 1)]



@torch.no_grad()
def synthesize_from_generated(net, net_g, items, args, device, pad_id):
    """Sample codes per item (oracle length cap) and decode them to audio."""
    wavs = []
    hop = code_frame_samples(net_g)
    for chunk in chunked(items, args.batch_size):
        B = len(chunk)
        Qs = int(chunk[0]["spec_codes"].shape[0])
        Qp = int(chunk[0]["pitch_codes"].shape[0])
        P = max(int(it["phone_ids"].numel()) for it in chunk)
        phone = torch.full((B, P), pad_id, dtype=torch.long, device=device)
        plen = torch.zeros(B, dtype=torch.long, device=device)
        sid = torch.zeros(B, dtype=torch.long, device=device)
        lens = []
        for i, it in enumerate(chunk):
            p = it["phone_ids"]
            phone[i, : p.numel()] = p.to(device)
            plen[i] = p.numel()
            sid[i] = int(it["sid"])
            lens.append(min(int(it["spec_codes"].shape[1]), args.max_frames))
        gen, glen = net.generate(phone, plen, sid, max_frames=max(lens),
                                 temperature=args.temperature,
                                 top_k=args.top_k,
                                 stop_on_eos=(False if args.ignore_eos else None),
                                 return_lengths=True)
        # infer_from_codes wants (Q, B, T) or (B, T, Q); generate returns (B, K, n).
        spec = gen[:, :Qs].transpose(0, 1).contiguous()
        pitch = gen[:, Qs:Qs + Qp].transpose(0, 1).contiguous()
        audio = net_g.infer_from_codes(spec, pitch, sid)        # (B, 1, N)
        for i in range(B):
            n = min(int(glen[i]), lens[i]) * hop
            wavs.append(audio[i, 0, :n].float().cpu().numpy())
    return wavs


@torch.no_grad()
def synthesize_from_reference(net_g, items, device):
    """Decode the item's ground-truth codes (the WER floor / control arm)."""
    wavs = []
    hop = code_frame_samples(net_g)
    for it in items:
        spec = it["spec_codes"].unsqueeze(1).to(device)          # (Q, 1, T)
        pitch = it["pitch_codes"].unsqueeze(1).to(device)
        sid = torch.tensor([int(it["sid"])], dtype=torch.long, device=device)
        audio = net_g.infer_from_codes(spec, pitch, sid)         # (1, 1, N)
        t = int(it["spec_codes"].shape[1])
        wavs.append(audio[0, 0, : t * hop].float().cpu().numpy())
    return wavs


def code_frame_samples(net_g) -> int:
    """Audio samples per *code* frame: ``hop_length * down_factor`` (40 ms here).

    Code frames are ``data.hop_length`` mel frames wide but the codec decimates
    the decoder rate by ``Generator.down_factor`` (4), so a code frame is
    4 x 10 ms and ``infer_from_codes`` returns ``T * 1920`` samples at 48 kHz.
    Using the bare ``hop_length`` here truncates the audio to 25 %.
    """
    hop = int(net_g.hp.data.hop_length)
    return hop * int(getattr(net_g, "down_factor", 1))


class Asr:
    """Thin wrapper around ``qwen_asr.Qwen3ASRModel`` (text only)."""

    def __init__(self, model_name, device, language):
        from qwen_asr import Qwen3ASRModel

        self.language = language
        self.model = Qwen3ASRModel.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map=str(device),
            max_inference_batch_size=8, max_new_tokens=256)

    def transcribe(self, wavs16k: Sequence[np.ndarray]) -> List[str]:
        audio = [(np.ascontiguousarray(w, dtype=np.float32), 16000)
                 for w in wavs16k]
        outs = self.model.transcribe(audio=audio, language=self.language)
        return [getattr(o, "text", "") or "" for o in outs]


def to_16k(wavs, src_rate):
    """48 kHz float32 numpy -> 16 kHz (torchaudio polyphase resampler)."""
    import torchaudio

    out = []
    for w in wavs:
        t = torch.from_numpy(np.clip(np.nan_to_num(w), -1.0, 1.0))
        if src_rate != 16000:
            t = torchaudio.functional.resample(t, src_rate, 16000)
        out.append(t.cpu().numpy().astype(np.float32))
    return out




def resolve_generator_path(path: str):
    """Accept a full path, a name inside ``checkpoints_stage1/``, or a run/name."""
    if not path:
        return None
    cands = [path, os.path.join("checkpoints_stage1", path)]
    if not os.path.isabs(path) and os.sep not in path:
        cands += sorted(glob.glob(os.path.join("checkpoints_stage1", "*",
                                               path)))
    for cand in cands:
        if os.path.exists(cand):
            return cand
    return None


def pick_indices(avail: List[int], n: int) -> List[int]:
    """``n`` evenly spaced entries of ``avail`` (deterministic, order-preserving)."""
    if n <= 0 or n >= len(avail):
        return list(avail)
    if n == 1:
        return [avail[len(avail) // 2]]
    return [avail[round(j * (len(avail) - 1) / (n - 1))] for j in range(n)]


def main():
    ap = argparse.ArgumentParser(
        description="WER of stage-2 checkpoints (generate codes -> decode -> "
                    "ASR -> WER).")
    ap.add_argument("--config", default="configs/transformer.yaml")
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--val_filelist", default=None,
                    help="default: config.train.val_filelist")
    ap.add_argument("--source_filelists", nargs="*", default=None,
                    help="filelists with transcripts (path|sid|text); default: "
                         "every filelists_2/*.txt")
    ap.add_argument("--generator_checkpoint", default=None,
                    help="stage-1 Generator used to decode codes; default: "
                         "config.train.generator_checkpoint")
    ap.add_argument("--num_items", type=int, default=20,
                    help="val utterances to score, evenly spaced over the "
                         "val filelist (0 = all items that have a reference)")
    ap.add_argument("--batch_size", type=int, default=4,
                    help="items per generation batch")
    ap.add_argument("--max_frames", type=int, default=1000,
                    help="upper bound on generated code frames (the item's own "
                         "length is used when shorter -- oracle duration)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="default: train.sample_temperature; 0 = greedy")
    ap.add_argument("--top_k", type=int, default=None,
                    help="default: train.sample_top_k; 0 disables")
    ap.add_argument("--ignore_eos", action="store_true",
                    help="decode exactly the item's own length instead of "
                         "letting the model emit EOS early; isolates content "
                         "errors from duration errors")
    ap.add_argument("--arms", default="generated,reference",
                    help="comma separated subset of generated,reference")
    ap.add_argument("--asr_model", default="Qwen/Qwen3-ASR-0.6B")
    ap.add_argument("--language", default="English")
    ap.add_argument("--save_dir", default=None,
                    help="write the decoded wavs here (<tag>_<i>.wav)")
    ap.add_argument("--verbose", action="store_true",
                    help="print reference / hypothesis per item")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed reset before each checkpoint's generation, so "
                         "every checkpoint draws the same samples (0 = fixed)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    config = OmegaConf.load(args.config)
    device = torch.device(args.device if args.device else
                          ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if args.temperature is None:
        args.temperature = float(config.train.get("sample_temperature", 0.9))
    if args.top_k is None:
        k = config.train.get("sample_top_k", None)
        args.top_k = int(k) if k else None
    elif args.top_k == 0:
        args.top_k = None
    arms = {a.strip() for a in args.arms.split(",") if a.strip()}
    if not arms <= {"generated", "reference"}:
        raise SystemExit("--arms must be a subset of generated,reference "
                         f"(got {sorted(arms)})")

    val_filelist = args.val_filelist or config.train.val_filelist
    sources = (args.source_filelists
               or sorted(glob.glob("filelists_2/*.txt")))
    print(f"reference filelists: {len(sources)}")
    refs = load_references(sources)
    print(f"  {len(refs)} transcripts "
          f"({', '.join(sources[:3])}{', ...' if len(sources) > 3 else ''})")

    # max_frames=None: score whole utterances, not the training window.
    dataset = CodeDataset(
        val_filelist, is_train=False, max_frames=None, max_phone_length=None,
        speaker_map=config.train.get("speaker_map", None),
    )
    avail = [i for i in range(len(dataset))
             if item_basename(dataset.items[i]["spec_codes"]) in refs]
    if not avail:
        raise SystemExit(
            "No val item has a reference transcript.  Pass --source_filelists "
            "with the filelist(s) the val set came from (they need "
            "'path|sid|text' lines).")
    picked = pick_indices(avail, args.num_items)
    items = [dataset[i] for i in picked]
    texts = [refs[item_basename(dataset.items[i]["spec_codes"])]
             for i in picked]
    print(f"val: {val_filelist} ({len(dataset)} items, {len(avail)} with a "
          f"reference) -> scoring {len(items)} items\n")

    gen_path = resolve_generator_path(
        args.generator_checkpoint or config.train.get("generator_checkpoint"))
    if gen_path is None:
        want = (args.generator_checkpoint
                or config.train.get("generator_checkpoint"))
        raise SystemExit(
            f"A stage-1 Generator checkpoint is required to turn codes into "
            f"audio.  Tried {want!r} (and checkpoints_stage1/...); pass a full "
            f"path with --generator_checkpoint, e.g. "
            f"checkpoints_stage1/base_v0.7/last.ckpt.")
    print(f"generator: {gen_path}")
    net_g = load_generator(gen_path, config, device)
    sample_rate = int(config.data.sampling_rate)
    frame_samples = code_frame_samples(net_g)
    labels = [f"{item_basename(dataset.items[i]['spec_codes'])} "
              f"({items[k]['spec_codes'].shape[1] * frame_samples / sample_rate:.2f}s)"
              for k, i in enumerate(picked)]
    asr = Asr(args.asr_model, device, args.language)
    print(f"asr: {args.asr_model}  device={device}  "
          f"temperature={args.temperature} top_k={args.top_k}  "
          f"code frame {1000 * frame_samples / sample_rate:.0f} ms\n")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    def save(tag, wavs):
        if not args.save_dir:
            return
        import soundfile as sf
        for i, w in enumerate(wavs):
            path = os.path.join(args.save_dir, f"{tag}_{i:03d}.wav")
            sf.write(path, np.clip(w, -1.0, 1.0), sample_rate)

    floor = None
    if "reference" in arms:
        ref_wavs = synthesize_from_reference(net_g, items, device)
        save("reference", ref_wavs)
        ref_hyps = asr.transcribe(to_16k(ref_wavs, sample_rate))
        floor = wer_summary(list(zip(texts, ref_hyps)))
        print(f"WER floor (ground-truth codes, same vocoder): "
              f"{floor['wer'] * 100:.2f}%  "
              f"[{floor['errors']}/{floor['words']} words]")
        if args.verbose:
            for i, (t, h) in enumerate(zip(texts, ref_hyps)):
                print(f"  [{i:02d}] {labels[i]}\n       ref: {t}\n       hyp: {h}")
        print()

    net = build_transformer(config, device)
    phone_pad = int(config.transformer.get("phone_pad_id", 0) or 0)
    rows = []
    for path in (args.checkpoints if "generated" in arms else []):
        epoch, step = load_transformer_weights(net, path)
        # Same draws for every checkpoint: the comparison is then paired, so a
        # WER difference is the weights and not the sampling noise.
        torch.manual_seed(args.seed)
        wavs = synthesize_from_generated(net, net_g, items, args, device,
                                         phone_pad)
        tag = f"{os.path.basename(os.path.dirname(path))}@{step}"
        save(tag, wavs)
        hyps = asr.transcribe(to_16k(wavs, sample_rate))
        summary = wer_summary(list(zip(texts, hyps)))
        rows.append((tag, summary))
        print(f"=== {os.path.basename(os.path.dirname(path))}/"
              f"{os.path.basename(path)}  (epoch {epoch}, step {step})")
        line = (f"    WER {summary['wer'] * 100:.2f}%   "
                f"S{summary['sub']} D{summary['del']} I{summary['ins']} "
                f"[{summary['errors']}/{summary['words']} words, "
                f"{summary['n']} utts]")
        if floor:
            line += f"   floor {floor['wer'] * 100:.2f}%"
        print(line)
        if args.verbose:
            for i, (t, h) in enumerate(zip(texts, hyps)):
                print(f"    [{i:02d}] {labels[i]}\n         ref: {t}\n         hyp: {h}")
        print()

    if len(rows) > 1:
        best = min(rows, key=lambda r: r[1]["wer"])
        print(f"lowest WER: {best[0]} ({best[1]['wer'] * 100:.2f}%)")


if __name__ == "__main__":
    main()


