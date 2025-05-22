import torch
import os
import glob
import random
import argparse
from tqdm import tqdm  # Optional progress bar

# Small epsilon to avoid log(0) - should match your model's value
LOG_EPSILON = 1e-5

def calculate_log_pitch_stats(data_dir, sample_size=500, epsilon=LOG_EPSILON):
    """
    Calculates the mean and standard deviation of log-transformed voiced pitch
    from a random sample of .f0 files in a directory.

    Args:
        data_dir (str): Path to the directory containing .f0 files.
        sample_size (int): Number of files to randomly sample.
        epsilon (float): Small value added before taking the logarithm.

    Returns:
        tuple: (mean, std) of log-voiced-pitch, or (None, None) if no voiced
               pitch is found in the sample.
    """
    f0_files = glob.glob(os.path.join(data_dir, '*.f0'))

    if not f0_files:
        print(f"Error: No '.f0' files found in directory: {data_dir}")
        return None, None

    print(f"Found {len(f0_files)} '.f0' files.")

    # Determine the actual number of files to sample
    num_to_sample = min(sample_size, len(f0_files))
    if num_to_sample < len(f0_files):
        print(f"Randomly selecting {num_to_sample} files for analysis...")
        sampled_files = random.sample(f0_files, num_to_sample)
    else:
        print("Analyzing all found files...")
        sampled_files = f0_files

    all_log_voiced_pitches = []
    total_voiced_frames = 0

    print("Loading files and extracting voiced log-pitch values...")
    for f0_file in tqdm(sampled_files, unit="file"):
        try:
            # Load the tensor, ensuring it's on the CPU
            pitch_tensor = torch.load(f0_file, map_location=torch.device('cpu'))

            if not isinstance(pitch_tensor, torch.Tensor):
                print(f"Warning: Skipping file {f0_file} - content is not a torch.Tensor.")
                continue

            # Ensure it's a float tensor for log operation
            pitch_tensor = pitch_tensor.float()

            # Identify voiced frames (pitch > 0)
            voiced_mask = pitch_tensor > 0
            voiced_pitches = pitch_tensor[voiced_mask]

            if voiced_pitches.numel() > 0:
                # Apply log transformation only to voiced parts
                log_voiced_pitches = torch.log(voiced_pitches + epsilon)
                all_log_voiced_pitches.append(log_voiced_pitches)
                total_voiced_frames += voiced_pitches.numel()
            # else: file might contain only unvoiced frames

        except Exception as e:
            print(f"Warning: Skipping file {f0_file} due to loading error: {e}")
            continue

    if not all_log_voiced_pitches:
        print("Error: No voiced pitch frames found in the sampled files.")
        return None, None

    # Concatenate all log-voiced-pitch values into a single tensor
    print(f"\nConcatenating {total_voiced_frames} voiced log-pitch values...")
    all_log_pitches_tensor = torch.cat(all_log_voiced_pitches)

    # Calculate mean and standard deviation
    mean_log_pitch = torch.mean(all_log_pitches_tensor)
    std_log_pitch = torch.std(all_log_pitches_tensor)

    return mean_log_pitch.item(), std_log_pitch.item()

def main():
    parser = argparse.ArgumentParser(
        description="Calculate mean and std of log-voiced-pitch from a sample of .f0 files."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the .f0 pitch files."
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=500,
        help="Number of .f0 files to randomly sample for analysis."
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=LOG_EPSILON,
        help="Small epsilon value added before log transformation (should match model)."
    )
    args = parser.parse_args()

    mean_val, std_val = calculate_log_pitch_stats(
        args.data_dir, args.sample_size, args.epsilon
    )

    if mean_val is not None and std_val is not None:
        print("\n--- Results ---")
        print(f"Approximate Mean Log-Voiced-Pitch: {mean_val:.4f}")
        print(f"Approximate Std Dev Log-Voiced-Pitch: {std_val:.4f}")
        print("-" * 15)
        suggested_noise_std_min = 0.1 * std_val
        suggested_noise_std_max = 0.25 * std_val # Increased upper bound slightly
        print(f"\nRecommendation for config.model.decoder.pitch_noise_std:")
        print(f"Start experimenting in the range: {suggested_noise_std_min:.4f} to {suggested_noise_std_max:.4f}")
        print(f"(These are ~10% to 25% of the calculated standard deviation)")
        print("Adjust based on validation performance and listening tests.")

if __name__ == "__main__":
    main()