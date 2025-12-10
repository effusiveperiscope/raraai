import argparse
import json
import os
from pathlib import Path
import soundfile as sf
import torch
import librosa
import numpy as np
from tqdm import tqdm
import random


def resample_audio(audio, orig_sr, target_sr=16000):
    """Resample audio to target sample rate using librosa."""
    if orig_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr,
            res_type='fft')
    return audio


def process_audio_file(audio_path, output_dir, target_sr=16000):
    """
    Load audio file, convert to tensor, resample to target sample rate, and save.
    
    Args:
        audio_path: Path to input audio file
        output_dir: Directory to save processed tensor
        target_sr: Target sample rate (default: 16000 Hz)
    
    Returns:
        Path to saved tensor file
    """
    # Read audio file
    audio, orig_sr = sf.read(audio_path)
    
    # Convert to float32 if needed
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    # Resample to target sample rate
    audio = resample_audio(audio, orig_sr, target_sr)
    
    # Convert to torch tensor
    audio_tensor = torch.from_numpy(audio)
    
    # Create output filename based on input filename
    input_filename = Path(audio_path).stem
    output_filename = f"{input_filename}.pt"
    output_path = os.path.join(output_dir, output_filename)
    
    # Save tensor
    torch.save(audio_tensor, output_path)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Process audio files and create filelist')
    parser.add_argument('--annotation_json', type=str, required=True,
                        help='Path to JSON file containing audio annotations')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save processed audio tensors and filelist')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Target sample rate (default: 16000 Hz)')
    parser.add_argument('--val_fraction', type=float, default=0.1,
                        help='Fraction of data to use for validation (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for train/val split (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load annotation JSON
    print(f"Loading annotations from {args.annotation_json}")
    with open(args.annotation_json, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # Process each audio file
    filelist_entries = []
    print(f"\nProcessing {len(annotations)} audio files...")
    
    for audio_path, metadata in tqdm(annotations.items()):
        try:
            # Process audio file
            output_path = process_audio_file(audio_path, args.output_dir, args.sample_rate)
            
            # Get intensity value
            intensity = metadata.get('intensity', 0)
            
            # Create filelist entry
            filelist_entries.append(f"{output_path}|{intensity}")
            
        except Exception as e:
            print(f"\nError processing {audio_path}: {str(e)}")
            continue
    
    # Shuffle and split into train/val
    random.shuffle(filelist_entries)
    num_val = int(len(filelist_entries) * args.val_fraction)
    val_entries = filelist_entries[:num_val]
    train_entries = filelist_entries[num_val:]
    
    # Write train filelist
    train_filelist_path = os.path.join(args.output_dir, 'train_filelist.txt')
    print(f"\nWriting train filelist to {train_filelist_path}")
    with open(train_filelist_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_entries))
    
    # Write val filelist
    val_filelist_path = os.path.join(args.output_dir, 'val_filelist.txt')
    print(f"Writing val filelist to {val_filelist_path}")
    with open(val_filelist_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_entries))
    
    print(f"\nProcessing complete!")
    print(f"Total processed: {len(filelist_entries)} files")
    print(f"Train set: {len(train_entries)} files")
    print(f"Val set: {len(val_entries)} files")
    print(f"Output directory: {args.output_dir}")
    print(f"Train filelist: {train_filelist_path}")
    print(f"Val filelist: {val_filelist_path}")


if __name__ == "__main__":
    main()