#!/usr/bin/env python3
"""
speaker_histogram.py — visualize speaker distribution from a filelist file.

Usage:
    python speaker_histogram.py <filelist_file>
"""

import sys
import os
from collections import Counter
from pathlib import Path


def parse_filelist(path: str) -> list[int]:
    speaker_ids = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if "|" not in line:
                print(f"Warning: line {lineno} has no '|' separator, skipping: {line!r}")
                continue
            _, _, raw_id = line.rpartition("|")
            try:
                speaker_ids.append(int(raw_id.strip()))
            except ValueError:
                print(f"Warning: line {lineno} has non-integer speaker ID {raw_id!r}, skipping.")
    return speaker_ids


def extract_speaker_name(filepath: str) -> str | None:
    """
    Try to extract a speaker name from the filename portion of the path.
    Expected filename format: HH_MM_SS_<SpeakerName>_<emotion>_...<rest>.flac
    e.g. 00_06_05_Fluttershy_Happy Singing_Noisy__To turn it up..flac
    """
    filename = Path(filepath).stem  # drop .flac
    parts = filename.split("_")
    # First 3 parts are time (HH, MM, SS), 4th is speaker name
    if len(parts) >= 4:
        return parts[3]
    return None


def build_speaker_labels(path: str) -> dict[int, str]:
    """Build a mapping of speaker_id -> name by scanning the filelist."""
    labels: dict[int, set[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "|" not in line:
                continue
            filepath, _, raw_id = line.rpartition("|")
            try:
                sid = int(raw_id.strip())
            except ValueError:
                continue
            name = extract_speaker_name(filepath.strip())
            if name:
                labels.setdefault(sid, set()).add(name)

    # Collapse sets to display strings; if multiple names map to same ID, join them
    return {
        sid: " / ".join(sorted(names))
        for sid, names in labels.items()
    }


def print_histogram(counts: Counter, labels: dict[int, str]) -> None:
    total = sum(counts.values())
    if total == 0:
        print("No entries found.")
        return

    sorted_items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))

    max_count = sorted_items[0][1]
    bar_width = 40  # characters

    # Column widths
    id_w = max(len("ID"), max(len(str(sid)) for sid, _ in sorted_items))
    label_w = max(
        len("Speaker"),
        max((len(labels.get(sid, "")) for sid, _ in sorted_items), default=0),
    )
    count_w = max(len("Count"), len(str(max_count)))
    pct_w = 7  # "100.0 %"

    header = (
        f"{'ID':<{id_w}}  "
        f"{'Speaker':<{label_w}}  "
        f"{'Count':>{count_w}}  "
        f"{'  %':>{pct_w}}  "
        f"Bar"
    )
    sep = "-" * (len(header) + bar_width)

    print()
    print(f"  Speaker Distribution  (total clips: {total:,})")
    print(sep)
    print(header)
    print(sep)

    for sid, count in sorted_items:
        pct = count / total * 100
        bar_len = round(count / max_count * bar_width)
        bar = "█" * bar_len
        label = labels.get(sid, "")
        print(
            f"{sid:<{id_w}}  "
            f"{label:<{label_w}}  "
            f"{count:>{count_w},}  "
            f"{pct:>{pct_w - 2}.1f} %  "
            f"{bar}"
        )

    print(sep)
    print(f"{'Total':<{id_w + 2 + label_w}}  {total:>{count_w},}  100.0 %")
    print()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python speaker_histogram.py <filelist_file>")
        sys.exit(1)

    filelist_path = sys.argv[1]

    if not os.path.isfile(filelist_path):
        print(f"Error: file not found: {filelist_path!r}")
        sys.exit(1)

    speaker_ids = parse_filelist(filelist_path)
    if not speaker_ids:
        print("No valid entries found in the filelist.")
        sys.exit(1)

    counts = Counter(speaker_ids)
    labels = build_speaker_labels(filelist_path)
    print_histogram(counts, labels)


if __name__ == "__main__":
    main()