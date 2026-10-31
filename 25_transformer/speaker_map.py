"""Speaker-id canonicalisation between stage 1 and stage 2.

Stage 1 (``Generator``) owns a *frozen* speaker-embedding table, so the
only ids that synthesize the correct voice are **stage-1 speaker ids**.
The stage-2 filelists, however, are assembled separately and may use a
completely different id space (e.g. ``Pinkie`` is ``4`` in the stage-1
filelist but ``1`` in a stage-2 filelist).

To make a single id usable for both the transformer conditioning and
``Generator.infer_from_codes`` we canonicalise stage-2 ids **into the
stage-1 space** with a small hand-editable JSON map::

    {
      "rules": [
        {"match": "Twilight", "sid": 0},
        {"match": "Pinkie",   "sid": 4},
        {"match": "glob:/data/singing/m5_*", "sid": 45}
      ],
      "sid_map": {"1": 4, "10": 0},
      "unmapped": "skip"
    }

* ``rules`` are tried in order against the audio path.  A ``match`` may be:
  * a plain substring (``"Twilight"``),
  * a glob, either written directly (``"*m5_*"``) or with an explicit
    ``glob:`` prefix (``"glob:/data/singing/m5_*"``), matched against the
    full path and the basename, or
  * an exact path with a ``path:`` prefix (``"path:/a/b/c.flac"``).
* ``sid_map`` remaps a *raw* stage-2 id directly.  It is used as a
  fallback when no path rule matches, which lets you fix an already
  preprocessed filelist without re-running ``2_preprocess``.
* ``unmapped`` controls what happens when nothing matches: ``"skip"``
  (drop the sample), ``"keep"`` (leave the original id) or an integer id.

``2_preprocess.py`` applies the path rules while the raw audio path is
still available.  ``dataset_codes.py`` applies ``sid_map`` (path-based
rules cannot be used there because preprocessed filelists no longer store
the audio path).

Bootstrap a starter map from the stage-1 assignment with::

    python speaker_map.py --filelist filelists_2/base.txt \\
        --stage1-filelist filelists_1/base_linux.txt --out speaker_map.json

The bootstrap canonicalises **by exact file first**: every stage-2 sample is
looked up in the stage-1 filelist (normalised path, then exact basename when
only the root differs, e.g. Windows vs Linux filelists), and the matched
stage-1 id is what the emitted rules are derived from.  The speaker-name
heuristic (``00_48_04_Sunset Shimmer_...flac`` -> ``Sunset Shimmer``) is only a
fallback for files that are absent from the stage-1 filelist.  When exact
matching disagrees with the name-level rule (or the filename is unparseable) a
``path:`` rule is emitted so the exact id wins at resolution time.
"""

from __future__ import annotations

import argparse
import collections
import fnmatch
import glob
import json
import os
import re
from typing import Optional

# ``00_48_04_Sunset Shimmer_Neutral___I'm sorry..flac`` -> ``Sunset Shimmer``
_SPEAKER_RE = re.compile(r"^\d{2}_\d{2}_\d{2}_(.+?)_")


def _norm_path(path: str) -> str:
    """Normalise separators so Windows and POSIX filelist entries compare."""
    return path.replace("\\", "/")


def speaker_name_from_path(path: str) -> Optional[str]:
    """Best-effort speaker/character name from a sliced-dialogue filename."""
    m = _SPEAKER_RE.match(os.path.basename(_norm_path(path)))
    return m.group(1) if m else None


def load_speaker_map(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _rule_matches(pattern: str, path: str) -> bool:
    if pattern.startswith("glob:"):
        pattern = pattern[len("glob:"):]
        return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(
            os.path.basename(path), pattern
        )
    if pattern.startswith("path:"):
        return _norm_path(path) == _norm_path(pattern[len("path:"):])
    if any(c in pattern for c in "*?["):
        return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(
            os.path.basename(path), pattern
        )
    return pattern in path


def resolve_sid(
    speaker_map: dict,
    path: Optional[str] = None,
    raw_sid: Optional[int] = None,
) -> Optional[int]:
    """Resolve a stage-1 speaker id using ``speaker_map``.

    Returns ``None`` when the sample should be skipped (``unmapped:
    "skip"``).
    """
    if speaker_map is None:
        return raw_sid
    for rule in speaker_map.get("rules", []) or []:
        pattern = rule.get("match")
        if path is not None and pattern is not None and _rule_matches(pattern, path):
            return int(rule["sid"])
    if raw_sid is not None:
        sid_map = speaker_map.get("sid_map") or {}
        key = str(int(raw_sid))
        if key in sid_map:
            return int(sid_map[key])
    unmapped = speaker_map.get("unmapped", "skip")
    if unmapped == "skip":
        return None
    if unmapped == "keep":
        return raw_sid
    return int(unmapped)


# ---------------------------------------------------------------------- #
# Bootstrapping helpers
# ---------------------------------------------------------------------- #
def build_from_stage1(stage1_filelist: str) -> dict:
    """Map speaker name -> stage-1 id (most frequent id wins)."""
    counts: dict = collections.defaultdict(collections.Counter)
    with open(stage1_filelist, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            path, _, raw = line.rpartition("|")
            try:
                sid = int(raw)
            except ValueError:
                continue
            name = speaker_name_from_path(path)
            counts[name][sid] += 1
    out = {}
    for name, c in counts.items():
        if name is None:
            continue

        # most frequent id; ties broken by the smallest id
        out[name] = min(c.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return out


def build_stage1_index(
    stage1_filelist: str,
) -> tuple[dict, dict, set]:
    """Index the stage-1 filelist for exact-file canonicalisation.

    Returns ``(path_sids, basename_sids, ambiguous_basenames)`` where

    * ``path_sids`` maps a separator-normalised full path to its stage-1 id
      (most frequent id wins when a path is listed more than once),
    * ``basename_sids`` maps a basename to its stage-1 id, but **only** when
      every stage-1 file with that basename agrees on the id,
    * ``ambiguous_basenames`` is the set of basenames that map to more than
      one stage-1 id and must therefore not be matched by basename alone.

    A basename index is needed because filelists from different platforms
    store the same corpus under different roots (e.g. ``D:\\MLP_Samples\\...``
    vs ``/mnt/data/MLP_Samples/...``), so full paths do not always match.
    """
    votes: dict = collections.defaultdict(collections.Counter)
    with open(stage1_filelist, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            path, _, raw = line.rpartition("|")
            try:
                sid = int(raw)
            except ValueError:
                continue
            votes[_norm_path(path)][sid] += 1

    path_sids = {
        path: min(c.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        for path, c in votes.items()
    }

    basenames: dict = collections.defaultdict(set)
    for path, sid in path_sids.items():
        basenames[os.path.basename(path)].add(sid)

    ambiguous = {b for b, sids in basenames.items() if len(sids) > 1}
    basename_sids = {
        b: next(iter(sids)) for b, sids in basenames.items() if len(sids) == 1
    }
    return path_sids, basename_sids, ambiguous


def make_template(filelist: str, stage1_filelist: str, out_path: str,
                  unmapped: str = "skip") -> dict:
    """Write a starter speaker map and report entries needing manual work.

    Each stage-2 file is first matched exactly against the stage-1 filelist
    (full path, then unambiguous basename).  Only files with no exact match
    fall back to the speaker-name heuristic, so the generated ids come from
    stage 1 whenever the sample is present in both corpora.  Exact matches
    that contradict the name-level rule (and matches for unparseable
    filenames) are emitted as ``path:`` rules ahead of the name rules.
    """
    stage1 = build_from_stage1(stage1_filelist)
    path_sids, basename_sids, ambiguous = build_stage1_index(stage1_filelist)

    seen = collections.Counter()
    unknown = collections.Counter()
    name_votes: dict = collections.defaultdict(collections.Counter)
    exact_hits: list[tuple[str, int, Optional[str]]] = []
    stats = collections.Counter()

    with open(filelist, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            path = line.split("|")[0]
            name = speaker_name_from_path(path)

            norm = _norm_path(path)
            basename = os.path.basename(norm)
            sid = path_sids.get(norm)
            if sid is not None:
                stats["exact_path"] += 1
            else:
                sid = basename_sids.get(basename)
                if sid is not None:
                    stats["exact_basename"] += 1
                else:
                    if basename in ambiguous:
                        stats["ambiguous_basename"] += 1
                    if name is not None and name in stage1:
                        sid = stage1[name]
                        stats["name"] += 1

            if sid is None:
                stats["unresolved"] += 1
                if name is None:
                    unknown[path] += 1
            else:
                exact_hits.append((path, sid, name))

            if name is not None:
                seen[name] += 1
                if sid is not None:
                    name_votes[name][sid] += 1

    # Speaker-name rules: prefer ids proven by exact file matches, then fall
    # back to the stage-1 name assignment.
    name_sid = {}
    for name in sorted(seen):
        votes = name_votes.get(name)
        if votes:
            # most frequent id; ties broken by the smallest id
            name_sid[name] = min(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        else:
            sid = stage1.get(name)
            if sid is not None:
                name_sid[name] = sid
            else:
                print(f"  [warn] no stage-1 id for speaker {name!r} "
                      f"({seen[name]} files)")

    # Name rules are emitted as delimited globs (``*_Name_*``) rather than bare
    # substrings, because a bare name also occurs inside dialogue text, e.g.
    # ``00_06_23_Rainbow_Annoyed_Noisy_all because of Pinkie's playing..flac``
    # would otherwise hit a ``Pinkie`` rule.  Speaker names always come from
    # the ``_<Name>_`` slot, so the delimiters are what make them unique.
    name_rules = [
        {"match": f"glob:*_{glob.escape(name)}_*", "sid": sid}
        for name, sid in name_sid.items()
    ]

    def _rule_sid(path: str) -> Optional[int]:
        for rule in name_rules:
            if _rule_matches(rule["match"], path):
                return int(rule["sid"])
        return None

    # Exact-path overrides, emitted only where the name rules would not already
    # reproduce the exact stage-1 id.
    override_rules = []
    seen_overrides = set()
    for path, sid, name in exact_hits:
        if _rule_sid(path) == sid:
            continue
        key = (path, sid)
        if key in seen_overrides:
            continue
        seen_overrides.add(key)
        override_rules.append({"match": f"path:{path}", "sid": sid})

    rules = override_rules + name_rules

    speaker_map = {"rules": rules, "sid_map": {}, "unmapped": unmapped}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(speaker_map, f, indent=2)

    print(f"Wrote starter map ({len(name_sid)} name rules, "
          f"{len(override_rules)} exact-path overrides) to {out_path}")
    print(f"  matched by exact path:     {stats['exact_path']}")
    print(f"  matched by exact basename: {stats['exact_basename']}")
    print(f"  matched by speaker name:   {stats['name']}")
    print(f"  unresolved:                {stats['unresolved']}")
    if stats["ambiguous_basename"]:
        print(f"  [warn] {stats['ambiguous_basename']} file(s) had a basename "
              f"shared by multiple stage-1 ids; fell back to speaker name")
    if unknown:
        print(f"  [warn] {len(unknown)} unresolved path(s) had no recognisable "
              "speaker name; add explicit rules for them, e.g. "
              '{"match": "glob:*m5_*", "sid": 45}')
    return speaker_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filelist", required=True,
                        help="stage-2 source filelist to bootstrap from")
    parser.add_argument("--stage1-filelist", required=True,
                        help="stage-1 filelist (path|sid) defining the canonical ids")
    parser.add_argument("--out", required=True, help="output speaker_map.json")
    parser.add_argument("--unmapped", default="skip",
                        help='unmapped policy: "skip", "keep" or an int id')
    args = parser.parse_args()
    make_template(args.filelist, args.stage1_filelist, args.out, args.unmapped)
