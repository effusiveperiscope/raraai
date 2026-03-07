import argparse
import os
import torch
from tqdm import tqdm

def win_longpath(path):
    if os.name != 'nt':
        return path
    if path.startswith('\\\\?\\'):
        return path
    return '\\\\?\\' + os.path.abspath(path)

def filter_filelist(filelist_path, output_path=None, min_frames=16):
    """
    Filter filelist to remove items with whisper features shorter than min_frames.
    
    Args:
        filelist_path: Path to the filelist to filter (train.txt or val.txt)
        output_path: Path to save filtered filelist (defaults to filelist_path + '.filtered')
        min_frames: Minimum number of frames required (default: 16)
    """
    if output_path is None:
        output_path = filelist_path + '.filtered'
    
    with open(filelist_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
    
    valid_lines = []
    removed_count = 0
    
    for line in tqdm(lines, desc=f'Filtering {os.path.basename(filelist_path)}'):
        if not line:
            continue
            
        # Parse the line to get the whisper feature path
        parts = line.split('|')
        if len(parts) < 9:
            print(f"Skipping malformed line: {line}")
            continue
            
        whisper_path = parts[0]
        whisper_path = win_longpath(whisper_path)
        
        # Check if whisper file exists
        if not os.path.exists(whisper_path):
            print(f"Whisper file not found: {whisper_path}")
            removed_count += 1
            continue
        
        # Load whisper features and check shape
        try:
            whisper_feats = torch.load(whisper_path, map_location='cpu')
            # Shape is [N, 512], we check the first dimension
            if whisper_feats.shape[0] >= min_frames:
                valid_lines.append(line)
            else:
                print(f"Removing short file ({whisper_feats.shape[0]} frames): {whisper_path}")
                removed_count += 1
        except Exception as e:
            print(f"Error loading {whisper_path}: {e}")
            removed_count += 1
            continue
    
    # Save filtered filelist
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(valid_lines))
    
    print(f"\n=== Filtering Summary ===")
    print(f"Original lines: {len(lines)}")
    print(f"Valid lines: {len(valid_lines)}")
    print(f"Removed lines: {removed_count}")
    print(f"Filtered filelist saved to: {output_path}")
    
    return valid_lines

def filter_train_and_val(output_dir, min_frames=16, backup=True):
    """
    Filter both train.txt and val.txt in the output directory.
    
    Args:
        output_dir: Directory containing train.txt and val.txt
        min_frames: Minimum number of frames required (default: 16)
        backup: Whether to backup original files (default: True)
    """
    train_path = os.path.join(output_dir, 'train.txt')
    val_path = os.path.join(output_dir, 'val.txt')
    
    # Backup original files if requested
    if backup:
        if os.path.exists(train_path):
            backup_path = train_path + '.backup'
            if not os.path.exists(backup_path):
                os.rename(train_path, backup_path)
                print(f"Backed up {train_path} to {backup_path}")
        if os.path.exists(val_path):
            backup_path = val_path + '.backup'
            if not os.path.exists(backup_path):
                os.rename(val_path, backup_path)
                print(f"Backed up {val_path} to {backup_path}")
    
    # Filter train.txt
    if os.path.exists(train_path + '.backup'):
        print("\n=== Filtering train.txt ===")
        filter_filelist(train_path + '.backup', train_path, min_frames)
    elif os.path.exists(train_path):
        print("\n=== Filtering train.txt ===")
        filter_filelist(train_path, train_path + '.tmp', min_frames)
        os.rename(train_path + '.tmp', train_path)
    
    # Filter val.txt
    if os.path.exists(val_path + '.backup'):
        print("\n=== Filtering val.txt ===")
        filter_filelist(val_path + '.backup', val_path, min_frames)
    elif os.path.exists(val_path):
        print("\n=== Filtering val.txt ===")
        filter_filelist(val_path, val_path + '.tmp', min_frames)
        os.rename(val_path + '.tmp', val_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Filter filelist to remove items with short whisper features'
    )
    parser.add_argument('--filelist', type=str, 
                        help='Path to a single filelist to filter')
    parser.add_argument('--output_dir', type=str,
                        help='Directory containing train.txt and val.txt to filter both')
    parser.add_argument('--output', type=str,
                        help='Output path for filtered filelist (only with --filelist)')
    parser.add_argument('--min_frames', type=int, default=16,
                        help='Minimum number of whisper frames required (default: 16)')
    parser.add_argument('--no_backup', action='store_true',
                        help='Do not backup original files')
    
    args = parser.parse_args()
    
    if args.output_dir:
        # Filter both train.txt and val.txt
        filter_train_and_val(args.output_dir, args.min_frames, not args.no_backup)
    elif args.filelist:
        # Filter a single filelist
        filter_filelist(args.filelist, args.output, args.min_frames)
    else:
        parser.error("Either --filelist or --output_dir must be specified")