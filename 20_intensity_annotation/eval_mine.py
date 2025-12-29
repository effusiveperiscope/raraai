import torch
import argparse
import librosa
import numpy as np
import csv
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from omegaconf import OmegaConf

# Imports assuming the same project structure as the example
from dataset import WhisperContext
from model import IntensityModel
from commons import load_submodule_prefix, sequence_mask

def process_audio_file(audio_path, context, model, device='cuda'):
    """
    Process a single audio file and return total subjective intensity.
    Retains logic from the original example.
    """
    # Load audio
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None
    
    # Process through model
    # Note: extract_features_batched expects a list of audio arrays
    feats, feat_lens = context.extract_features_batched([audio])
    
    # Move to device and adjust dimensions
    feats = feats.half().to(device).unsqueeze(0)
    interp_feats = context.interp2(feats)
    feat_mask = sequence_mask(feat_lens).to(torch.long).to(device)
    
    # Trim to max length
    interp_feats = interp_feats[:, :feat_lens.max(), :]
    
    with torch.no_grad():
        intensity_pred, attn = model(interp_feats, feat_mask)
    
    # Calculate score based on attention weights
    attn_feats = (intensity_pred * attn).cpu().detach().numpy()
    # Formula derived from provided snippet
    total_pred = float((attn_feats.sum(axis=1)) * 8 + 1)
    
    return total_pred

def main():
    parser = argparse.ArgumentParser(description="Calculate intensity statistics per speaker.")
    parser.add_argument('filelist', type=str, help='Path to text file with format: path/to/audio|speaker_id')
    parser.add_argument('--output_csv', type=str, default='speaker_intensity_stats.csv', help='Path to save output CSV')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/base.yaml', help='Path to model config')
    parser.add_argument('--min_samples', type=int, default=1, help='Omit speakers with fewer samples than this count')
    
    args = parser.parse_args()
    
    # ---------------------------------------------------------
    # 1. Setup Model and Context
    # ---------------------------------------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model on {device}...")
    
    context = WhisperContext()
    config = OmegaConf.load(args.config)
    
    model = IntensityModel(**config.model).to(device).half()
    state = torch.load(args.ckpt, map_location='cpu', weights_only=False)['state_dict']
    load_submodule_prefix(model, 'model.', state)
    model.eval()

    # ---------------------------------------------------------
    # 2. Parse Filelist
    # ---------------------------------------------------------
    print(f"Reading filelist: {args.filelist}")
    dataset = []
    try:
        with open(args.filelist, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 2:
                    audio_path = parts[0].strip()
                    speaker_id = parts[1].strip()
                    dataset.append((audio_path, speaker_id))
    except FileNotFoundError:
        print(f"Error: Filelist {args.filelist} not found.")
        exit(1)

    print(f"Found {len(dataset)} items in filelist.")

    # ---------------------------------------------------------
    # 3. Process Audio and Accumulate Scores
    # ---------------------------------------------------------
    speaker_scores = defaultdict(list)
    
    print("Processing audio files...")
    for audio_path, speaker_id in tqdm(dataset):
        score = process_audio_file(audio_path, context, model, device=device)
        if score is not None:
            speaker_scores[speaker_id].append(score)

    # ---------------------------------------------------------
    # 4. Calculate Statistics and Filter
    # ---------------------------------------------------------
    output_rows = []
    skipped_speakers = 0

    print("\nCalculating statistics...")
    for speaker_id, scores in speaker_scores.items():
        count = len(scores)
        
        # Filter based on min_samples argument
        if count < args.min_samples:
            skipped_speakers += 1
            continue
            
        # Calculate Standard Deviation (Population or Sample? Defaulting to Sample ddof=1)
        if count > 1:
            std_dev = np.std(scores, ddof=1) 
        else:
            std_dev = 0.0 # STD is 0 if only 1 sample
            
        output_rows.append({
            'Speaker ID': speaker_id,
            'Sample Count': count,
            'Intensity STD': f"{std_dev:.4f}",
            'Mean Intensity': f"{np.mean(scores):.4f}" # Added Mean for extra utility
        })

    # Sort by Speaker ID for cleanliness
    output_rows.sort(key=lambda x: x['Speaker ID'])

    # ---------------------------------------------------------
    # 5. Write to CSV
    # ---------------------------------------------------------
    if output_rows:
        headers = ['Speaker ID', 'Sample Count', 'Intensity STD', 'Mean Intensity']
        try:
            with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                writer.writerows(output_rows)
            print(f"\nSuccess! Statistics written to: {args.output_csv}")
            print(f"Speakers processed: {len(output_rows)}")
            print(f"Speakers omitted (< {args.min_samples} samples): {skipped_speakers}")
        except Exception as e:
            print(f"Error writing CSV: {e}")
    else:
        print("\nNo data generated (check your file paths or threshold settings).")

if __name__ == '__main__':
    main()