# Input filelist:
# audio_file_path|speaker_name

# Output filelist:
# feature_file_path|speaker_id

INPUT_FILELIST = r'D:\DataAugmentation\TestMulti2.list'
OUTPUT_FEATURES_DIR = 'TestMultiFeatures2'
OUTPUT_FILELIST = 'TestMultiFeatures2.list'
SPEAKER_MAP = 'speaker_map.txt'

from svc_helper.sfeatures.models import RVCHubertModel
import librosa
import numpy as np
import os
import torch

from tqdm import tqdm
def main():
    """Main entry point"""
    speaker_ids = {}
    if not os.path.exists(OUTPUT_FEATURES_DIR):
        os.makedirs(OUTPUT_FEATURES_DIR)
    rvc_model = RVCHubertModel()
    with open(OUTPUT_FILELIST, 'w') as f2:
        with open (INPUT_FILELIST, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines, total=len(lines), desc='Generating features'):
                line = line.strip()
                file_path = line.split('|')[0]
                speaker_name = line.split('|')[1]

                # Map unique speaker names to unique IDs
                if speaker_name not in speaker_ids:
                    speaker_ids[speaker_name] = len(speaker_ids)

                # Load audio file and extract features
                out_path = os.path.join(
                    OUTPUT_FEATURES_DIR,
                    os.path.basename(file_path).replace('.wav', '.npy'))
                if os.path.exists(out_path):
                    continue
                
                data, rate = librosa.load(file_path, sr=RVCHubertModel.expected_sample_rate)
                padded_data = rvc_model.pad_audio(data)
                feat = rvc_model.extract_features(torch.from_numpy(data))

                # Save features to numpy file
                with open(out_path, 'wb') as f:
                    np.save(f, feat)
                
                # Write output filelist
                f2.write(f'{out_path}|{speaker_ids[speaker_name]}\n')

    # Write speaker map
    with open(SPEAKER_MAP, 'w') as f:
        for k, v in speaker_ids.items():
            f.write(f'{k}|{v}\n')
    

def load_data(filelist=OUTPUT_FILELIST):
    """Load features and speaker IDs from file"""
    features_list = []
    speaker_ids = []
    speaker_ids_set = set()
    with open(filelist, 'r') as f:
        for line in f:
            line = line.strip()
            feat_path, speaker_id = line.split('|')
            features_list.append(np.load(feat_path))
            speaker_ids.append(int(speaker_id))
            speaker_ids_set.add(int(speaker_id))
    return features_list, speaker_ids, len(speaker_ids)

if __name__ == '__main__':
    main()

