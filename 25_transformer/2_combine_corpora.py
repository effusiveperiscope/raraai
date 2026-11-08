"""Combine separately preprocessed stage-2 corpora into one expanded corpus.

Every directory handled here is the output of ``2_preprocess.py``: three
tensors per clip (``*.phone_ids`` / ``*.spec_codes`` / ``*.pitch_codes``) plus
``train_stage2.txt`` / ``val_stage2.txt`` lines of the form::

    phone_ids_path|spec_codes_path|pitch_codes_path|sid

**Nothing is copied or re-encoded.** The emitted filelists reference the
existing tensor files in place, so combining is a few seconds and a re-mix
costs no GPU time.

Speaker-id policy (option B -- new speakers get brand-new ids)
-------------------------------------------------------------
The *base* corpus keeps its ids verbatim (by default the Genshin corpus, whose
ids already are the stage-1 ``Generator`` ids).  Each *extra* corpus then gets
a contiguous block of **fresh** ids on top, one per raw speaker, starting at
``max(base sid) + 1``.  Why fresh ids instead of reusing a stage-1 voice:

* the stage-2 transformer's speaker table is what has to hold them, and its
  ``n_speakers`` is fixed -- so the block must fit under ``--n_speakers``
  (default 4096, see ``configs/transformer.yaml``);
* the stage-1 ``Generator`` has **no trained embedding** for a new id (rows
  past the trained range are random ``nn.Embedding`` init), so a new id cannot
  be decoded as-is.

That last point is the whole reason for the fresh block: the extra voices are
**conditioning only**.  They contribute text coverage to the transformer and are
never rendered, so your existing voices keep their ids -- they are never averaged
with the extra data and stay decodable exactly as before.  No proxy mapping is
written: ``speaker_manifest.json`` records the name/gender per new id, which is
all you would need to build one later.

Usage
-----
::

    python 2_combine_corpora.py \\
        --base data/genshin_full_stage2 \\
        --extra data/mlp_stage2 --extra data/vctk_stage2 --extra data/expresso_read_stage2 \\
        --output_dir data/genshin_plus_extras \\
        --n_speakers 4096 --cap_multiplier 1.0 --dry_run

The ``--extra`` order decides the id blocks: the first extra corpus gets
``max(base sid) + 1`` onwards and each following one continues after it.

then point ``configs/transformer.yaml`` at
``data/genshin_plus_extras/train_stage2.txt`` (and ``.../val_stage2.txt``).

``--cap_multiplier`` caps each extra corpus at *multiplier x len(base train)*
clips, subsampled deterministically and proportionally per speaker.
``--cap_total_multiplier`` budgets the *sum* of the extras instead and splits it
across corpora by size -- use that one when mixing several corpora, since three
corpora at 0.9x each are 2.7x the base.  The transformer only sees lines, not
sample weights, so the ratio in these filelists *is* the loss ratio.  0 disables
both.

Deleting the output directory loses nothing: re-running with the same inputs
reproduces byte-identical filelists.

Outputs (in ``--output_dir``)
-----------------------------
``train_stage2.txt`` / ``val_stage2.txt``
    Base + extras, ids remapped.  ``val_stage2.txt`` is the **base val only**
    unless ``--include_extra_val``; keeping it unchanged is what keeps
    ``val_loss`` / ``val_text_delta`` comparable across runs.
``speaker_manifest.json``
    ``corpus -> raw sid -> new sid`` plus per-speaker clip counts, names and
    genders, and ``n_speakers_required`` to check against
    ``transformer.n_speakers``.  Speaker names/genders come from a
    ``*_speakers.json`` in the extra directory, so copy the sidecar next to the
    preprocessed tensors if you want them annotated::

        cp filelists_2/vctk_linux_speakers.json data/vctk_stage2/

    Without it the ids are mapped exactly the same way, the manifest just has
    ``speaker: null``.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

TRAIN = "train_stage2.txt"
VAL = "val_stage2.txt"


def read_filelist(path: str) -> List[str]:
    """Read a stage-2 filelist, dropping blank lines."""
    if not os.path.exists(path):
        raise SystemExit(f"missing filelist: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def split_line(line: str, source: str) -> Tuple[Tuple[str, str, str], int]:
    """Split ``p1|p2|p3|sid`` into ``((p1, p2, p3), sid)``."""
    parts = line.split("|")
    if len(parts) != 4:
        raise SystemExit(
            f"{source}: expected 4 '|'-separated fields, got {len(parts)}: "
            f"{line[:120]}")
    try:
        sid = int(parts[3])
    except ValueError:
        raise SystemExit(f"{source}: non-integer sid {parts[3]!r} in {line[:120]}")
    return (parts[0], parts[1], parts[2]), sid


def remap(line: str, new_sid: int) -> str:
    """Replace the sid field of a stage-2 line."""
    parts = line.split("|")
    return "|".join(parts[:3] + [str(new_sid)])


def subsample(lines: List[str], cap: int) -> List[str]:
    """Deterministically keep ``cap`` of ``lines`` (hash order, not RNG).

    Hashing the line itself means the same corpus always yields the same
    subset, so a re-run of this script cannot silently change the mix.
    """
    if cap <= 0:
        return []
    if len(lines) <= cap:
        return lines
    ordered = sorted(lines, key=lambda ln: hashlib.md5(ln.encode()).hexdigest())
    return ordered[:cap]


def allocate(counts: Dict[int, int], cap: int) -> Dict[int, int]:
    """Split ``cap`` clips across speakers proportionally (largest remainder).

    Exact, unlike ``round(fraction * n)`` per speaker: the allocation sums to
    ``min(cap, sum(counts))``, so ``--cap_multiplier 1.0`` really means 1.0x and
    the speaker balance is preserved (a corpus where one speaker has 70% of the
    clips keeps roughly 70% after capping).  A speaker whose share rounds to
    zero gets no *train* clips; it keeps its val lines and still gets an id.
    """
    total = sum(counts.values())
    if cap <= 0 or total <= cap:
        return dict(counts)
    quotas = {k: v * cap / total for k, v in counts.items()}
    alloc = {k: min(int(q), counts[k]) for k, q in quotas.items()}
    rem = cap - sum(alloc.values())
    # Largest fractional part first; ties broken by sid so the result is stable.
    order = sorted(counts, key=lambda k: (-(quotas[k] - int(quotas[k])), k))
    i = 0
    while rem > 0 and order:
        k = order[i % len(order)]
        if alloc[k] < counts[k]:
            alloc[k] += 1
            rem -= 1
        i += 1
    return alloc



def group_by_sid(lines: Sequence[str], source: str) -> Dict[int, List[str]]:
    """Group filelist lines by their raw sid."""
    groups: Dict[int, List[str]] = {}
    for ln in lines:
        _, sid = split_line(ln, source)
        groups.setdefault(sid, []).append(ln)
    return groups


def load_sidecar(extra_dir: str) -> Optional[dict]:
    """Load the ``*_speakers.json`` written next to an extra corpus, if any.

    Used only to annotate the manifest with speaker names/genders, so you can
    tell which new id is which voice.  Mapping ids does not depend on it.
    """
    hits = sorted(glob.glob(os.path.join(extra_dir, "*_speakers.json")))
    if not hits:
        return None
    if len(hits) > 1:
        print(f"WARNING: {extra_dir} has {len(hits)} *_speakers.json files; "
              f"using {os.path.basename(hits[0])}")
    with open(hits[0], "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def count_train_lines(extra_dir: str) -> int:
    """Number of train clips in one preprocessed corpus (for cap budgeting)."""
    return len(read_filelist(os.path.join(extra_dir, TRAIN)))


def plan_extra(extra_dir: str, next_sid: int, cap: int,
               include_val: bool) -> dict:
    """Assign fresh ids to one extra corpus and return its plan.

    A speaker's id is decided from the **union of its train and val lines**, so
    a speaker that happens to land entirely in val still gets the same id it
    would have had in train (no id drift between the two filelists).

    Args:
        extra_dir: preprocessed corpus directory.
        next_sid: first id to hand out to this corpus's speakers.
        cap: absolute clip budget for the train split (0 = no cap).
        include_val: accepted for symmetry with the driver; the val lines are
            always returned and the driver decides whether to use them.
    """
    train_path = os.path.join(extra_dir, TRAIN)
    val_path = os.path.join(extra_dir, VAL)
    train_lines = read_filelist(train_path)
    val_lines = read_filelist(val_path) if os.path.exists(val_path) else []
    if not val_lines:
        print(f"NOTE: {extra_dir} has no usable {VAL}; using base val only")

    train_groups = group_by_sid(train_lines, train_path)
    val_groups = group_by_sid(val_lines, val_path) if val_lines else {}
    all_sids = sorted(set(train_groups) | set(val_groups))

    sid_map = {raw: next_sid + i for i, raw in enumerate(all_sids)}
    next_sid += len(all_sids)

    # Cap the corpus size (proportionally per speaker) so it cannot drown the
    # base corpus; CodeDataset has no sampling weights, so the line ratio *is*
    # the loss ratio.  `cap` is already an absolute clip count: the driver folds
    # `--cap_multiplier` / `--cap_total_multiplier` into it.
    total = len(train_lines)
    counts = {raw: len(train_groups.get(raw, [])) for raw in all_sids}
    alloc = allocate(counts, cap) if cap and total > cap else counts

    kept_train: List[str] = []
    kept_per_sid: Dict[int, int] = {}
    for raw in all_sids:
        items = subsample(train_groups.get(raw, []), alloc.get(raw, 0))
        kept_per_sid[raw] = len(items)
        kept_train.extend(remap(ln, sid_map[raw]) for ln in items)

    kept_val = [remap(ln, sid_map[raw]) for raw in all_sids
                for ln in val_groups.get(raw, [])]

    sidecar = load_sidecar(extra_dir)
    side_speakers = (sidecar or {}).get("speakers") or {}
    speakers = {}
    for raw in all_sids:
        info = side_speakers.get(str(raw)) or {}
        speakers[raw] = {
            "new_sid": sid_map[raw],
            "speaker": info.get("speaker"),
            "gender": info.get("gender"),
            "raw_lines": len(train_groups.get(raw, [])),
            "kept_lines": kept_per_sid[raw],
        }
    return {
        "dir": extra_dir,
        "name": os.path.basename(os.path.normpath(extra_dir)),
        "train": kept_train,
        "val": kept_val,
        "sid_map": sid_map,
        "speakers": speakers,
        "n_train_raw": total,
        "n_train_kept": len(kept_train),
        "n_val": len(kept_val),
        "capped": bool(cap) and total > cap,
        "used_sidecar": sidecar is not None,
    }


def check_paths(lines: Sequence[str], label: str, strict: bool) -> int:
    """Verify the three tensors of every line exist; return the missing count."""
    missing = 0
    shown = 0
    for ln in lines:
        paths, _ = split_line(ln, label)
        gone = [p for p in paths if not os.path.exists(p)]
        if gone:
            missing += 1
            if shown < 5:
                print(f"  MISSING ({label}): {gone[0]}")
                shown += 1
    if missing and strict:
        raise SystemExit(
            f"{label}: {missing} line(s) reference tensors that do not exist. "
            f"Run this from the repo root (paths in the filelists are relative "
            f"to it) and make sure the extra corpora are still preprocessed.")
    return missing



# ---------------------------------------------------------------------- #
# Driver
# ---------------------------------------------------------------------- #
def write_lines(path: str, lines: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def check_duplicates(sections: Sequence[Tuple[str, Sequence[str]]]) -> int:
    """Count clips whose tensor triple appears more than once across sections."""
    seen: Dict[Tuple[str, str, str], str] = {}
    dupes = 0
    for label, lines in sections:
        for ln in lines:
            paths, _ = split_line(ln, label)
            if paths in seen:
                dupes += 1
                if dupes <= 5:
                    print(f"  DUPLICATE ({label} vs {seen[paths]}): {paths[0]}")
            else:
                seen[paths] = label
    return dupes


def corpus_caps(args, base_train_n: int) -> List[int]:
    """Per-corpus train-clip budgets, from the two cap knobs.

    ``--cap_multiplier`` caps *each* corpus independently; it cannot control the
    total when several corpora are mixed (three corpora at 0.9x each are 2.7x the
    base).  ``--cap_total_multiplier`` budgets the *sum* of the extras and splits
    it proportionally across them, which is the knob to reach for when adding
    several corpora at once.  Both are expressed against the base train length,
    and both subsample per speaker (see :func:`allocate`).
    """
    n = len(args.extra)
    if n == 0:
        return []
    if args.cap_multiplier and args.cap_multiplier > 0:
        per = int(round(args.cap_multiplier * base_train_n))
        return [per] * n
    if args.cap_total_multiplier and args.cap_total_multiplier > 0:
        sizes = [count_train_lines(e) for e in args.extra]
        total_cap = int(round(args.cap_total_multiplier * base_train_n))
        if sum(sizes) <= total_cap:
            return [0] * n
        alloc = allocate(dict(enumerate(sizes)), total_cap)
        return [alloc[i] for i in range(n)]
    return [0] * n


def build(args) -> int:
    base_dir = args.base
    base_train_path = os.path.join(base_dir, TRAIN)
    base_val_path = os.path.join(base_dir, VAL)
    base_train = read_filelist(base_train_path)
    base_val = read_filelist(base_val_path) if os.path.exists(base_val_path) else []
    if not base_val:
        print(f"WARNING: {base_dir} has no {VAL}; the combined val set will "
              f"come from the extra corpora only")

    out_dir = args.output_dir
    for path in [base_dir, *args.extra]:
        if os.path.normpath(os.path.abspath(path)) == \
                os.path.normpath(os.path.abspath(out_dir)):
            raise SystemExit(
                f"--output_dir must not be one of the inputs ({out_dir}): this "
                f"script would overwrite its own source filelists")

    base_sids = sorted({split_line(ln, base_train_path)[1] for ln in base_train}
                       | {split_line(ln, base_val_path)[1] for ln in base_val})
    base_max = max(base_sids) if base_sids else -1
    next_sid = args.sid_start if args.sid_start is not None else base_max + 1
    if next_sid <= base_max:
        raise SystemExit(
            f"--sid_start {next_sid} collides with the base corpus, whose ids "
            f"go up to {base_max}; use {base_max + 1} or higher")

    print(f"base {base_dir}: {len(base_train)} train / {len(base_val)} val, "
          f"{len(base_sids)} speaker ids in "
          f"[{min(base_sids) if base_sids else 0}, {base_max}]")
    caps = corpus_caps(args, len(base_train))
    plans = []
    for extra, cap in zip(args.extra, caps):
        plan = plan_extra(extra, next_sid, cap, args.include_extra_val)
        next_sid = (max(plan["sid_map"].values()) + 1
                    if plan["sid_map"] else next_sid)
        plans.append(plan)
        lo = min(plan["sid_map"].values())
        hi = max(plan["sid_map"].values())
        print(f"extra {plan['dir']}: {plan['n_train_raw']} -> "
              f"{plan['n_train_kept']} train clips"
              + (" (capped)" if plan["capped"] else "")
              + f", {plan['n_val']} val, {len(plan['sid_map'])} speakers "
              f"-> sids [{lo}, {hi}]"
              + ("" if plan["used_sidecar"] else "  (no *_speakers.json)"))

    train_lines = list(base_train) + [ln for p in plans for ln in p["train"]]
    val_lines = list(base_val)
    if args.include_extra_val:
        val_lines += [ln for p in plans for ln in p["val"]]

    # --- checks ---------------------------------------------------------- #
    max_sid = max((split_line(ln, "combined")[1]
                   for ln in train_lines + val_lines), default=-1)
    ok = True
    if max_sid >= args.n_speakers:
        print(f"ERROR: the combined corpus needs speaker id {max_sid}, but "
              f"transformer.n_speakers={args.n_speakers} only covers "
              f"[0, {args.n_speakers - 1}].  Either raise "
              f"transformer.n_speakers (that reshapes BOTH speaker tables, "
              f"and commons.load_state_dict_mismatch then re-initialises "
              f"them wholesale -- see AGENTS.md), cluster the extra speakers "
              f"onto fewer ids, or drop a corpus.")
        ok = False
    dupes = check_duplicates([("base", base_train)]
                             + [(p["name"], p["train"]) for p in plans])
    if dupes:
        print(f"ERROR: {dupes} clip(s) appear in more than one corpus -- the "
              f"same audio would be trained twice, under two speaker ids.")
        ok = False
    n_extras = len(train_lines) - len(base_train)
    if base_train and n_extras > 3 * len(base_train) and not args.cap_multiplier:
        print(f"WARNING: the extra corpora add {n_extras} clips to "
              f"{len(base_train)} base clips (>3x).  The line ratio is the "
              f"loss ratio -- consider --cap_multiplier 1.0")
    if not args.no_check_files:
        if check_paths(train_lines, "train", strict=True):
            ok = False
        if args.include_extra_val and check_paths(val_lines, "val", strict=True):
            ok = False
    if not ok:
        raise SystemExit("aborted; nothing written")

    print(f"combined: {len(train_lines)} train / {len(val_lines)} val clips, "
          f"sids [{min(base_sids) if base_sids else 0}, {max_sid}] "
          f"-> transformer.n_speakers must be >= {max_sid + 1}")
    if args.dry_run:
        print("--dry_run: nothing written")
        return 0


    # --- write ----------------------------------------------------------- #
    os.makedirs(out_dir, exist_ok=True)
    write_lines(os.path.join(out_dir, TRAIN), train_lines)
    write_lines(os.path.join(out_dir, VAL), val_lines)

    extras_meta = []
    for plan in plans:
        extras_meta.append({
            "dir": plan["dir"],
            "name": plan["name"],
            "n_train_raw": plan["n_train_raw"],
            "n_train_kept": plan["n_train_kept"],
            "n_val": plan["n_val"],
            "capped": plan["capped"],
            "used_sidecar": plan["used_sidecar"],
            "sid_map": {str(k): v for k, v in sorted(plan["sid_map"].items())},
            "speakers": {str(k): v
                         for k, v in sorted(plan["speakers"].items())},
        })

    manifest = {
        "generated_by": "2_combine_corpora.py",
        "base": {"dir": base_dir, "n_train": len(base_train),
                 "n_val": len(base_val), "n_sids": len(base_sids),
                 "sid_min": min(base_sids) if base_sids else None,
                 "sid_max": base_max},
        "extras": extras_meta,
        "totals": {"n_train": len(train_lines), "n_val": len(val_lines),
                   "n_sids": len({split_line(ln, "combined")[1]
                                  for ln in train_lines + val_lines}),
                   "max_sid": max_sid},
        "n_speakers_required": max_sid + 1,
        "cap_multiplier": args.cap_multiplier,
        "cap_total_multiplier": args.cap_total_multiplier,
        "include_extra_val": bool(args.include_extra_val),
    }
    with open(os.path.join(out_dir, "speaker_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    n_new = sum(len(p["sid_map"]) for p in plans)
    print(f"wrote {out_dir}/{TRAIN}, {out_dir}/{VAL} and speaker_manifest.json "
          f"({n_new} new speaker ids)")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Combine preprocessed stage-2 corpora, appending new "
                    "speaker ids on top of a base corpus")
    p.add_argument("--base", default="data/genshin_full_stage2",
                   help="preprocessed directory treated as the base corpus; "
                        "its speaker ids are kept verbatim")
    p.add_argument("--extra", action="append", default=[],
                   help="additional preprocessed directory (repeatable); its "
                        "speakers get fresh ids above the base range")
    p.add_argument("--output_dir", required=True,
                   help="where the combined filelists + manifests are written")
    p.add_argument("--sid_start", type=int, default=None,
                   help="first id handed to the extra corpora "
                        "(default: max base id + 1)")
    p.add_argument("--n_speakers", type=int, default=4096,
                   help="transformer.n_speakers: the combined ids must stay "
                        "below it (default %(default)s, see "
                        "configs/transformer.yaml)")
    p.add_argument("--cap_multiplier", type=float, default=0.0,
                   help="cap EACH extra corpus at this multiple of the base "
                        "train length, subsampled per speaker (0 = no cap)")
    p.add_argument("--cap_total_multiplier", type=float, default=0.0,
                   help="cap the SUM of the extra corpora at this multiple of "
                        "the base train length, split across them in proportion "
                        "to their size (0 = no cap). Use this when mixing "
                        "several corpora; it wins over --cap_multiplier.")
    p.add_argument("--include_extra_val", action="store_true",
                   help="also append the extra corpora's val lines; the "
                        "default (base val only) keeps val_loss comparable")
    p.add_argument("--no_check_files", action="store_true",
                   help="skip the existence check on the referenced tensors")
    p.add_argument("--dry_run", action="store_true",
                   help="report the plan and the id ranges, write nothing")
    return build(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

