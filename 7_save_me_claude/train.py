import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model import create_model, VoiceConversionTrainer

# Example custom dataset for voice conversion
class VoiceConversionDataset(Dataset):
    def __init__(self, features_list, speaker_ids, max_len=None):
        """
        Args:
            features_list: List of speech features arrays
            speaker_ids: List of speaker IDs corresponding to features
            max_len: Optional max sequence length for padding/truncation
        """
        self.features = features_list
        self.speaker_ids = speaker_ids
        self.max_len = max_len
        
        # Create speaker-wise indices for reference speech selection
        self.speaker_to_indices = {}
        for i, spk_id in enumerate(speaker_ids):
            if spk_id not in self.speaker_to_indices:
                self.speaker_to_indices[spk_id] = []
            self.speaker_to_indices[spk_id].append(i)

        self.speaker_ids_set = set(speaker_ids)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Get current speech features and speaker
        speech = self.features[idx]
        speaker_id = self.speaker_ids[idx]

        if speech.ndim == 3: # [batch = 1, seq_len, feature_dim]
            speech = speech.squeeze(0)  # [seq_len, feature_dim]
        
        # Optionally handle sequence length
        if self.max_len is not None:
            if speech.shape[0] > self.max_len:
                # Randomly crop
                start = np.random.randint(0, speech.shape[0] - self.max_len)
                speech = speech[start:start + self.max_len]
            elif speech.shape[0] < self.max_len:
                # Pad with zeros
                padding = np.zeros((self.max_len - speech.shape[0], speech.shape[1]))
                speech = np.concatenate([speech, padding], axis=0)
        
        # Get reference speech from same speaker (different utterance)
        speaker_indices = self.speaker_to_indices[speaker_id].copy()
        if len(speaker_indices) > 1:  # If more than one utterance from this speaker
            speaker_indices.remove(idx)  # Remove current utterance
            ref_idx = np.random.choice(speaker_indices)  # Choose random utterance from same speaker
            reference_speech = self.features[ref_idx]

            if reference_speech.ndim == 3: # [batch = 1, seq_len, feature_dim]
                reference_speech = reference_speech.squeeze(0)  # [seq_len, feature_dim]
            
            # Handle sequence length for reference speech
            if self.max_len is not None:
                if reference_speech.shape[0] > self.max_len:
                    start = np.random.randint(0, reference_speech.shape[0] - self.max_len)
                    reference_speech = reference_speech[start:start + self.max_len]
                elif reference_speech.shape[0] < self.max_len:
                    padding = np.zeros((self.max_len - reference_speech.shape[0], reference_speech.shape[1]))
                    reference_speech = np.concatenate([reference_speech, padding], axis=0)
        else:
            # If only one utterance, use it as its own reference (not ideal but a fallback)
            reference_speech = speech.copy()
        
        return {
            "speech": torch.FloatTensor(speech),
            "speaker_id": torch.LongTensor([speaker_id]).squeeze(),
            "reference_speech": torch.FloatTensor(reference_speech),
        }


# Example training loop
def train_model(model, train_dataset, val_dataset, batch_size=32, epochs=50, stage=1,
    config=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    trainer = VoiceConversionTrainer(model, learning_rate=0.0001, device=device,
        config=config)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        train_loss, out = trainer.train_epoch(train_loader, use_speaker_id=True,
             stage=stage, speaker_ids_set=train_dataset.speaker_ids_set)
        
        # Validation phase
        val_loss, val_out = trainer.validate(val_loader, use_speaker_id=True, stage=stage,
             speaker_ids_set=val_dataset.speaker_ids_set)
        
        if stage == 1:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        elif stage == 2:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                  f", Cyclic Loss: {out['cyclic_loss']:.6f}, Val Cyclic Loss: {val_out['cyclic_loss']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_voice_conversion_model2.pt")
            print(f"Model saved with validation loss: {val_loss:.6f}")
    
    print("Training completed!")
    return model


# Example inference function
def convert_voice(model, source_features, target_features):
    """
    Convert source speech to target speaker's voice
    
    Args:
        model: Trained VoiceConversionModel
        source_features: Source speech features [seq_len, feature_dim]
        target_features: Target speaker reference features [ref_seq_len, feature_dim]
    
    Returns:
        converted_features: Converted speech features [seq_len, feature_dim]
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Prepare inputs
    source = torch.FloatTensor(source_features).unsqueeze(0).to(device)  # Add batch dimension
    target = torch.FloatTensor(target_features).unsqueeze(0).to(device)  # Add batch dimension
    
    # Convert
    with torch.no_grad():
        converted = model.convert_voice(source, target)
    
    # Return as numpy array
    return converted.squeeze(0).cpu().numpy()


# Example training and inference workflow
def main():

    # Load config for create_model from TestMultiFeatures2.yaml
    import yaml
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="config file for create_model")
    parser.add_argument("--filelist", help="training data filelist",
        default="TestMultiFeatures2.list")
    parser.add_argument("--pretrained", help="pretrained model")
    parser.add_argument("--stage", help="stage", default=1) # 1: pretrain, 2: cycle consistency
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
    
    # 1. Create model
    model = create_model(config)
    
    if os.path.exists(args.pretrained):
        model.load_state_dict(torch.load(args.pretrained))
        print(f"Model loaded from {args.pretrained}")
    
    # 2. Prepare your data
    max_seq_len = config['max_seq_len']

    from preprocess import load_data
    features_list, speaker_ids, num_speakers = load_data(args.filelist)
    max_seq_len = max(feat.shape[1] for feat in features_list)
    num_samples = len(features_list)
    
    # 3. Create datasets
    train_size = int(0.8 * num_samples)
    train_dataset = VoiceConversionDataset(
        features_list[:train_size], 
        speaker_ids[:train_size],
        max_len=max_seq_len
    )
    val_dataset = VoiceConversionDataset(
        features_list[train_size:], 
        speaker_ids[train_size:],
        max_len=max_seq_len
    )
    
    # 4. Train model
    if args.stage == 1:
        trained_model = train_model(model, train_dataset, val_dataset,
            batch_size=config['stage1_batch_size'], epochs=config['stage1_epochs'], 
            stage=1, config=config)
    else:
        trained_model = train_model(model, train_dataset, val_dataset,
            batch_size=config['stage2_batch_size'], epochs=config['stage2_epochs'], 
            stage=2, config=config)
    
    # 5. Example conversion
    # source_idx = train_size + 5  # Just an example index from validation set
    # target_idx = train_size + 10  # Different speaker for conversion target
    # 
    # converted_features = convert_voice(
        # trained_model, 
        # features_list[source_idx], 
        # features_list[target_idx]
    # )
    # 
    # print(f"Converted features shape: {converted_features.shape}")
    
    # 6. In a real application, you would use a vocoder to convert
    # the features back to audio waveforms


if __name__ == "__main__":
    main()

