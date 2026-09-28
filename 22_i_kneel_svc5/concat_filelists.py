#!/usr/bin/env python3
"""
Concatenate multiple filelists with speaker ID offsetting.

Each line in a filelist has the format:
    /path/to/file.wav|speaker_id

IDs in each subsequent filelist are offset by the max ID seen so far,
so IDs remain globally unique across all filelists.

Usage:
    python concat_filelists.py filelist1.txt filelist2.txt [...] -o output.txt
"""

import argparse
from pathlib import Path


def read_filelist(path: str) -> list[tuple[str, int]]:
    """Read a filelist and return a list of (filepath, speaker_id) tuples."""
    entries = []
    with open(path, "r") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit("|", 1)
            if len(parts) != 2:
                raise ValueError(
                    f"{path}:{lineno}: expected 'filepath|id', got: {line!r}"
                )
            filepath, raw_id = parts
            try:
                speaker_id = int(raw_id)
            except ValueError:
                raise ValueError(
                    f"{path}:{lineno}: speaker ID {raw_id!r} is not an integer"
                )
            entries.append((filepath, speaker_id))
    return entries


def concat_filelists(input_paths: list[str], output_path: str) -> None:
    all_lines = []
    id_offset = 0

    for filelist_path in input_paths:
        entries = read_filelist(filelist_path)
        if not entries:
            print(f"  [warn] {filelist_path}: empty filelist, skipping")
            continue

        local_ids = {sid for _, sid in entries}
        max_local_id = max(local_ids)
        num_speakers = len(local_ids)

        print(
            f"  {filelist_path}: {len(entries)} lines, "
            f"{num_speakers} unique speaker(s), "
            f"local IDs {min(local_ids)}–{max_local_id}, "
            f"offset → +{id_offset}"
        )

        for filepath, speaker_id in entries:
            all_lines.append(f"{filepath}|{speaker_id + id_offset}")

        id_offset += max_local_id + 1  # next filelist starts after this one's max

    with open(output_path, "w") as f:
        f.write("\n".join(all_lines) + "\n")

    # Summary
    final_ids = {int(line.rsplit("|", 1)[1]) for line in all_lines}
    print(
        f"\nWrote {len(all_lines)} lines with "
        f"{len(final_ids)} unique speaker IDs "
        f"(0–{max(final_ids)}) → {output_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate filelists with globally unique speaker IDs."
    )
    parser.add_argument(
        "filelists",
        nargs="+",
        metavar="FILELIST",
        help="Input filelist files (processed in order).",
    )
    parser.add_argument(
        "-o", "--output",
        default="combined_filelist.txt",
        metavar="OUTPUT",
        help="Output filelist path (default: combined_filelist.txt).",
    )
    args = parser.parse_args()

    # Validate inputs exist
    for p in args.filelists:
        if not Path(p).is_file():
            parser.error(f"File not found: {p}")

    print(f"Concatenating {len(args.filelists)} filelist(s):")
    concat_filelists(args.filelists, args.output)


if __name__ == "__main__":
    main()