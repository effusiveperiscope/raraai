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
"""

from __future__ import annotations

import argparse
import collections
import fnmatch
import json
import os
import re
from typing import Optional

# ``00_48_04_Sunset Shimmer_Neutral___I'm sorry..flac`` -> ``Sunset Shimmer``
_SPEAKER_RE = re.compile(r"^\d{2}_\d{2}_\d{2}_(.+?)_")


def speaker_name_from_path(path: str) -> Optional[str]:
    """Best-effort speaker/character name from a sliced-dialogue filename."""
    m = _SPEAKER_RE.match(os.path.basename(path))
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
        return path == pattern[len("path:"):]
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


def make_template(filelist: str, stage1_filelist: str, out_path: str,
                  unmapped: str = "skip") -> dict:
    """Write a starter speaker map and report entries needing manual work."""
    stage1 = build_from_stage1(stage1_filelist)
    seen = collections.Counter()
    unknown = collections.Counter()
    with open(filelist, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            path = line.split("|")[0]
            name = speaker_name_from_path(path)
            if name is None:
                unknown[path] += 1
            else:
                seen[name] += 1

    rules = []
    for name in sorted(seen):
        sid = stage1.get(name)
        if sid is not None:
            rules.append({"match": name, "sid": sid})
        else:
            print(f"  [warn] no stage-1 id for speaker {name!r} ({seen[name]} files)")

    speaker_map = {"rules": rules, "sid_map": {}, "unmapped": unmapped}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(speaker_map, f, indent=2)

    print(f"Wrote starter map ({len(rules)} rules) to {out_path}")
    if unknown:
        print(f"  [warn] {len(unknown)} path(s) had no recognisable speaker name; "
              "add explicit rules for them, e.g. "
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
