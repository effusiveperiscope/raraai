#!/usr/bin/env python3
import argparse
import os
import torch

def main():
    parser = argparse.ArgumentParser(
        description="Merge a variable number of sid_avgs.pt files, sequentially offsetting speaker IDs."
    )

    # Positional argument: takes 2 or more files to merge
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Paths to the sid_avgs.pt files to merge (in order)",
    )

    # Optional argument for output path
    parser.add_argument(
        "-o",
        "--output",
        default="data/sid_avgs.pt",
        help="Path to save the merged output (default: data/sid_avgs.pt)",
    )

    args = parser.parse_args()

    # Enforce at least 2 files
    if len(args.input_files) < 2:
        parser.error("You must provide at least two input files to merge.")

    merged_dict = {}
    current_offset = 0

    print("Starting merge process...")

    for i, file_path in enumerate(args.input_files):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        print(f"Processing [{i+1}/{len(args.input_files)}]: {file_path}")

        # Load the current dictionary
        # Weights are usually saved as CPU/GPU tensors; map_location='cpu' is safer for general CLI usage
        current_dict = torch.load(file_path, map_location="cpu")

        # Apply offset to keys and merge
        for sid, emb in current_dict.items():
            new_sid = str(int(sid) + current_offset)
            merged_dict[new_sid] = emb

        # Update the offset by the number of unique IDs in the file just processed
        current_offset += len(current_dict)

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the combined dictionary
    torch.save(merged_dict, args.output)

    print("-" * 40)
    print(f"Successfully merged {len(args.input_files)} files!")
    print(f"Total keys in merged dictionary: {len(merged_dict)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()