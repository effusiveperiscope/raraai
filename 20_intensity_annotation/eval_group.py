import torch
from dataset import WhisperContext
from model import IntensityModel
from omegaconf import OmegaConf
import argparse
import librosa
import numpy as np
from commons import load_submodule_prefix, sequence_mask
from pathlib import Path
from tqdm import tqdm

def process_audio_file(audio_path, context, model):
    """Process a single audio file and return total subjective intensity."""
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Process through model
    feats, feat_lens = context.extract_features_batched([audio])
    feats = feats.half().to('cuda').unsqueeze(0)
    interp_feats = context.interp2(feats)
    feat_mask = sequence_mask(feat_lens).to(torch.long).to('cuda')
    interp_feats = interp_feats[:, :feat_lens.max(), :]
    
    with torch.no_grad():
        intensity_pred, attn = model(interp_feats, feat_mask)
    
    attn_feats = (intensity_pred * attn).cpu().detach().numpy()
    total_pred = float((attn_feats.sum(axis=1)) * 8 + 1)
    
    return total_pred

if __name__ == '__main__':
    context = WhisperContext()
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_dir', type=str, help='Directory containing audio files')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--config', type=str, default='config/base.yaml')
    parser.add_argument('--extensions', type=str, nargs='+', 
                        default=['.wav', '.mp3', '.flac', '.ogg', '.m4a'],
                        help='Audio file extensions to process')
    args = parser.parse_args()
    
    # Load model once
    config = OmegaConf.load(args.config)
    model = IntensityModel(**config.model).to('cuda').half()
    state = torch.load(args.ckpt, map_location='cpu', weights_only=False)['state_dict']
    load_submodule_prefix(model, 'model.', state)
    model.eval()
    
    # Find all audio files
    audio_dir = Path(args.audio_dir)
    audio_files = []
    for ext in args.extensions:
        audio_files.extend(audio_dir.glob(f'*{ext}'))
    audio_files = sorted(audio_files)
    
    if len(audio_files) == 0:
        print(f"No audio files found in {audio_dir}")
        exit(1)
    
    print(f"Found {len(audio_files)} audio files")
    print()
    
    # Process all files
    results = []
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        try:
            total_intensity = process_audio_file(audio_file, context, model)
            results.append((audio_file.name, total_intensity))
        except Exception as e:
            print(f"\nError processing {audio_file.name}: {e}")
            results.append((audio_file.name, "ERROR"))
    
    # Print results table
    print("\n" + "="*70)
    print(f"{'Audio File':<40} {'Total Subjective Intensity':>25}")
    print("="*70)
    
    for filename, intensity in results:
        if isinstance(intensity, str):  # Error case
            print(f"{filename:<40} {intensity:>25}")
        else:
            print(f"{filename:<40} {intensity:>25.4f}")
    
    print("="*70)
    
    # Print statistics
    valid_intensities = [i for _, i in results if isinstance(i, float)]