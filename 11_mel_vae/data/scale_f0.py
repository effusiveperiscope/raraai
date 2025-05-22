import os
import torch
import argparse
import sys

# Define the scaling factor
# Use floating-point division
SCALING_FACTOR = 44100.0 / 16000.0

def process_f0_file(filepath):
    """
    Loads an .f0 file saved with torch.save, scales the data,
    and saves it back to the original file.
    """
    print(f"Processing: {filepath}...")
    try:
        # Load the data - assuming it's a PyTorch tensor
        original_data = torch.load(filepath)

        if not isinstance(original_data, torch.Tensor):
            print(f"  WARNING: File {filepath} does not contain a PyTorch Tensor. Skipping.")
            return False

        # Ensure data is float for multiplication
        if not torch.is_floating_point(original_data):
             print(f"  INFO: Casting data in {filepath} to float.")
             original_data = original_data.float()

        # Scale the data
        scaled_data = original_data * SCALING_FACTOR

        # Save the scaled data back to the original file
        torch.save(scaled_data, filepath)
        print(f"  Successfully scaled and saved: {filepath}")
        return True

    except FileNotFoundError:
        print(f"  ERROR: File not found: {filepath}. Skipping.")
        return False
    except Exception as e:
        print(f"  ERROR: Failed to process file {filepath}: {e}. Skipping.")
        return False

def main():
    """
    Main function to parse arguments and traverse the directory.
    """
    parser = argparse.ArgumentParser(
        description="Recursively find *.f0 files (saved with torch.save), "
                    "scale their tensor data by 44100/16000, and save back."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="The root directory to search for *.f0 files."
    )

    args = parser.parse_args()
    target_directory = args.directory

    if not os.path.isdir(target_directory):
        print(f"Error: Directory not found: {target_directory}", file=sys.stderr)
        sys.exit(1)

    print(f"Starting f0 scaling process in directory: {target_directory}")
    print(f"Scaling factor: {SCALING_FACTOR:.4f}")
    print("-" * 30)

    processed_count = 0
    skipped_count = 0

    # Recursively walk through the directory
    for root, dirs, files in os.walk(target_directory):
        for filename in files:
            if filename.endswith(".f0"):
                filepath = os.path.join(root, filename)
                if process_f0_file(filepath):
                    processed_count += 1
                else:
                    skipped_count += 1

    print("-" * 30)
    print("Processing complete.")
    print(f"Successfully processed files: {processed_count}")
    print(f"Skipped/Errored files:      {skipped_count}")

if __name__ == "__main__":
    # Ensure PyTorch is installed
    try:
        import torch
    except ImportError:
        print("Error: PyTorch is not installed. Please install it using 'pip install torch'", file=sys.stderr)
        sys.exit(1)

    main()