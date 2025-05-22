import torch
import os
import argparse
from tqdm import tqdm
from nsfhifigan.wav2mel import PitchAdjustableMelSpectrogram
import math

def calculate_stats(mel_dir):
    """
    Calculates the global mean and standard deviation of logs of linear mel spectrograms
    stored in .mel files within a directory.

    Args:
        mel_dir (str): The path to the directory containing .mel files.

    Returns:
        tuple: A tuple containing (global_mean, global_std), or (None, None)
               if no valid .mel files are found or an error occurs.
    """
    if not os.path.isdir(mel_dir):
        print(f"Error: Directory not found: {mel_dir}")
        return None, None

    mel_files = [f for f in os.listdir(mel_dir) if f.lower().endswith('.mel')]

    if not mel_files:
        print(f"Error: No '.mel' files found in directory: {mel_dir}")
        return None, None

    print(f"Found {len(mel_files)} '.mel' files. Processing...")

    total_sum = 0.0
    total_sum_sq = 0.0
    total_elements = 0

    # Use float64 for accumulators to maintain precision
    dtype = torch.float64

    for filename in tqdm(mel_files, desc="Calculating Stats"):
        filepath = os.path.join(mel_dir, filename)
        try:
            # Load the tensor. Assuming the file contains *only* the tensor.
            # If it's a dict, adjust accordingly (e.g., data = torch.load(filepath)['mel_tensor'])
            mel_tensor = torch.load(filepath)

            if not isinstance(mel_tensor, torch.Tensor):
                print(f"Warning: Skipping non-tensor file: {filename}")
                continue

            # Ensure tensor is float for calculations
            mel_tensor = mel_tensor.to(dtype)

            num_elements = mel_tensor.numel()
            if num_elements == 0:
                print(f"Warning: Skipping empty tensor in file: {filename}")
                continue
        
            log_mel_tensor = PitchAdjustableMelSpectrogram.dynamic_range_compression_torch(
                mel_tensor)

            # Accumulate sum and sum of squares
            # Use .item() to get Python floats and avoid holding tensors in memory
            total_sum += torch.sum(log_mel_tensor).item()
            total_sum_sq += torch.sum(log_mel_tensor ** 2).item()
            total_elements += num_elements

        except Exception as e:
            print(f"Warning: Could not load or process file {filename}. Error: {e}")
            continue

    if total_elements == 0:
        print("Error: No valid data points found in any '.mel' files.")
        return None, None, None

    # Calculate mean
    global_mean = total_sum / total_elements

    # Calculate variance and standard deviation
    # Var(X) = E[X^2] - (E[X])^2
    variance = (total_sum_sq / total_elements) - (global_mean ** 2)

    # Handle potential floating point inaccuracies -> variance slightly negative
    if variance < 0:
        print(f"Warning: Calculated variance is slightly negative ({variance}). Clamping to 0.")
        variance = 0.0

    global_std = math.sqrt(variance)

    return global_mean, global_std, total_elements

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate global mean and standard deviation for .mel files in a directory."
    )
    parser.add_argument(
        "mel_directory",
        type=str,
        help="Path to the directory containing the .mel files (saved using torch.save)."
    )

    args = parser.parse_args()

    mean, std, total_elements = calculate_stats(args.mel_directory)

    if mean is not None and std is not None:
        print("\n--- Global Statistics ---")
        print(f"Total elements processed: {total_elements}") # Making total_elements accessible for print
        print(f"Global Mean: {mean:.6f}")
        print(f"Global Std Dev: {std:.6f}")
        print("\nUse these values to normalize your log mel spectrograms for training:")
        print(f"normalized_mel = (mel - {mean:.6f}) / {std:.6f}")