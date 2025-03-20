import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random
from tqdm import tqdm

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=5000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Create cached cos and sin values for rotary embeddings
        # We use dim/2 because each dimension pair shares the same angle
        self.freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.freq = self.freq.unsqueeze(0)  # [1, dim/2]
        
        # Initialize cache for position indices
        self.register_buffer("positions", torch.arange(0, max_seq_len).float().unsqueeze(1))  # [max_seq_len, 1]
        
        # Compute cos and sin values: [max_seq_len, dim/2]
        cos_cached = torch.cos(self.positions * self.freq)
        sin_cached = torch.sin(self.positions * self.freq)
        
        # Interleave to match pair structure: [max_seq_len, dim]
        cos_cached = torch.repeat_interleave(cos_cached, 2, dim=1)
        sin_cached = torch.repeat_interleave(sin_cached, 2, dim=1)
        
        self.register_buffer("cos_cached", cos_cached)
        self.register_buffer("sin_cached", sin_cached)
        
    def forward(self, x, seq_dim=1):
        """
        Apply rotary embeddings to input tensor
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim] or [seq_len, batch_size, dim]
            seq_dim: Dimension corresponding to sequence length (default: 1)
            
        Returns:
            Tensor with rotary position embeddings applied
        """
        seq_len = x.shape[seq_dim]
        
        # Get cos and sin values for this sequence length
        cos = self.cos_cached[:seq_len]  # [seq_len, dim]
        sin = self.sin_cached[:seq_len]  # [seq_len, dim]
        
        # Reshape depending on input shape
        if seq_dim == 0:
            # [seq_len, batch_size, dim]
            cos = cos.unsqueeze(1) 
            sin = sin.unsqueeze(1)
        else:
            # [batch_size, seq_len, dim]
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        
        # Apply rotary embeddings
        # For each dimension pair (x_i, x_{i+1}), apply rotation:
        # [x_i, x_{i+1}] -> [x_i*cos - x_{i+1}*sin, x_i*sin + x_{i+1}*cos]
        
        # First, reshape x to separate even and odd dimensions
        x_shape = x.shape
        x_reshaped = x.reshape(*x_shape[:-1], -1, 2)
        
        # Get even and odd dimensions
        x_even = x_reshaped[..., 0]
        x_odd = x_reshaped[..., 1]
        
        # Reshape cos and sin to match
        cos_view = cos.reshape(*cos.shape[:-1], -1, 2)[..., 0]
        sin_view = sin.reshape(*sin.shape[:-1], -1, 2)[..., 0]
        
        # Apply rotation
        x_rotated_even = x_even * cos_view - x_odd * sin_view
        x_rotated_odd = x_even * sin_view + x_odd * cos_view
        
        # Stack even and odd dimensions
        x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
        
        # Reshape back to original shape
        x_rotated = x_rotated.reshape(*x_shape)
        
        return x_rotated

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # Feed-forward network
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)
        
        return x

class SpeechEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, bottleneck_dim, n_layers=6, n_heads=8, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = RotaryPositionalEmbedding(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.bottleneck = nn.Sequential(nn.Linear(hidden_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),)
        
    def forward(self, x, mask=None):
        # x shape: [batch_size, seq_len, input_dim]
        
        # Project input features to hidden dimension
        x = self.input_projection(x)  # [batch_size, seq_len, hidden_dim]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Rearrange dimensions for transformer: [seq_len, batch_size, hidden_dim]
        x = x.transpose(0, 1)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Revert to [batch_size, seq_len, hidden_dim]
        x = x.transpose(0, 1)

        # Bottleneck layer
        x = self.bottleneck(x)
        
        return x

class SpeechDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, bottleneck_dim, speaker_dim, n_layers=6, n_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.speaker_dim = speaker_dim

        # Bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),)
        
        # Speaker embedding projection
        self.speaker_projection = nn.Linear(speaker_dim, hidden_dim)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, bottleneck_representation, speaker_embedding, mask=None):
        # bottleneck_representation: [batch_size, seq_len, bottleneck_dim]
        # speaker_embedding: [batch_size, speaker_dim]

        # Bottleneck layer
        hidden_representation = self.bottleneck(bottleneck_representation)
        
        # Project speaker embedding and expand
        batch_size, seq_len, _ = hidden_representation.shape
        speaker_proj = self.speaker_projection(speaker_embedding)  # [batch_size, hidden_dim]
        speaker_proj = speaker_proj.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, hidden_dim]
        
        # Combine hidden representation with speaker information
        x = hidden_representation + speaker_proj  # Simple addition, could be concatenation + projection
        
        # Rearrange for transformer: [seq_len, batch_size, hidden_dim]
        x = x.transpose(0, 1)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Revert to [batch_size, seq_len, hidden_dim]
        x = x.transpose(0, 1)
        
        # Project to output dimension
        output = self.output_projection(x)  # [batch_size, seq_len, output_dim]
        
        return output

class SpeakerEncoder(nn.Module):
    def __init__(self, input_dim, speaker_dim, hidden_dim=256, n_layers=3):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True
        )
        
        self.speaker_projection = nn.Linear(hidden_dim * 2, speaker_dim)
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        x = self.input_projection(x)
        _, (hidden, _) = self.lstm(x)
        
        # Get last layer hidden state (bidirectional)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [batch_size, hidden_dim*2]
        
        # Project to speaker embedding dimension
        speaker_embedding = self.speaker_projection(hidden)  # [batch_size, speaker_dim]
        
        return speaker_embedding


class VoiceConversionModel(nn.Module):
    def __init__(
        self, config):
        super().__init__()

        input_dim=config.get("input_dim", 768)         # HuBERT/wav2vec feature dimension
        hidden_dim=config.get("hidden_dim", 512)        # Transformer hidden dimension
        bottleneck_dim=config.get("bottleneck_dim", 256)    # Bottleneck dimension
        speaker_dim=config.get("speaker_dim", 128)       # Speaker embedding dimension
        output_dim=config.get("output_dim", 768)        # Output speech feature dimension
        num_speakers=config.get("num_speakers", 100)      # Number of speakers in training data
        encoder_layers=config.get("encoder_layers", 6)
        decoder_layers=config.get("decoder_layers", 6)
        n_heads=config.get("n_heads", 8)
        dropout=config.get("dropout", 0.1)
        
        # Speech encoder
        self.encoder = SpeechEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            n_layers=encoder_layers,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Speech decoder
        self.decoder = SpeechDecoder(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            bottleneck_dim=bottleneck_dim,
            speaker_dim=speaker_dim,
            n_layers=decoder_layers,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Speaker encoder (for training and inference)
        # self.speaker_encoder = SpeakerEncoder(
        #     input_dim=input_dim,
        #     speaker_dim=speaker_dim
        # )
        
        # Alternative: Learnable speaker embeddings (for training)
        self.speaker_embeddings = nn.Embedding(num_speakers, speaker_dim)
        
    def forward(self, x, speaker_id=None, reference_speech=None):
        """
        Forward pass with either speaker ID or reference speech
        
        Args:
            x: Input speech features [batch_size, seq_len, input_dim]
            speaker_id: Integer speaker IDs [batch_size]
            reference_speech: Reference speech for target speaker [batch_size, ref_seq_len, input_dim]
            
        Returns:
            output: Converted speech features [batch_size, seq_len, output_dim]
        """

        # Encode input speech to hidden representation
        hidden = self.encoder(x)
        
        # Get speaker embedding either from ID or reference speech
        if speaker_id is not None:
            speaker_emb = self.speaker_embeddings(speaker_id)
        elif reference_speech is not None:
            speaker_emb = self.speaker_encoder(reference_speech)
        else:
            raise ValueError("Either speaker_id or reference_speech must be provided")
        
        # Decode with speaker embedding
        output = self.decoder(hidden, speaker_emb)
        
        return output

    def encode(self, x):    
        return self.encoder(x)

    def decode(self, x, speaker_id=None, reference_speech=None):    
        if speaker_id is not None:
            return self.decoder(x, self.speaker_embeddings(speaker_id))
        elif reference_speech is not None:
            return self.decoder(x, self.speaker_encoder(reference_speech))
        else:
            raise ValueError("Either speaker_id or reference_speech must be provided")
    
    def convert_voice(self, source_speech, target_speech=None, target_speaker_id=None):
        """
        Convert source speech to target speaker's voice
        
        Args:
            source_speech: Source speech features [1, seq_len, input_dim]
            target_speech: Target speaker reference speech [1, ref_seq_len, input_dim] (optional)
            target_speaker_id: Target speaker ID (optional)
            
        Returns:
            converted_speech: Speech features in target voice [1, seq_len, output_dim]
            
        Note:
            Either target_speech or target_speaker_id must be provided
        """
        if target_speech is None and target_speaker_id is None:
            raise ValueError("Either target_speech or target_speaker_id must be provided")
            
        with torch.no_grad():
            # Encode source speech
            hidden = self.encoder(source_speech)
            
            # Extract speaker embedding either from target speech or speaker ID
            if target_speech is not None:
                speaker_emb = self.speaker_encoder(target_speech)
            else:
                speaker_emb = self.speaker_embeddings(target_speaker_id)
            
            # Decode with target speaker embedding
            converted_speech = self.decoder(hidden, speaker_emb)
            
        return converted_speech

class VoiceConversionTrainer:
    def __init__(
        self,
        model,
        learning_rate=0.0001,
        device="cuda" if torch.cuda.is_available() else "cpu",
        config=None
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        self.config = config
        
    def train_step(self, batch, use_speaker_id=True, stage=1, speaker_ids_set=None):
        """
        Single training step
        
        Args:
            batch: Dictionary containing:
                - speech: [batch_size, seq_len, input_dim]
                - speaker_id: [batch_size]
                - reference_speech: [batch_size, ref_seq_len, input_dim] (optional)
        """
        speech = batch["speech"].to(self.device)
        
        if use_speaker_id:
            speaker_id = batch["speaker_id"].to(self.device)
            reference_speech = None
        else:
            speaker_id = None
            reference_speech = batch.get("reference_speech")
            if reference_speech is not None:
                reference_speech = reference_speech.to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        encoded = self.model.encode(speech)
        output = self.model.decode(encoded, speaker_id, reference_speech)
        # Reconstruction loss
        loss = F.mse_loss(output, speech)

        out = {
            "loss": loss.item(),
        }
        if stage == 2:
            cyclic_loss = self.cyclic_step(
                encoded, speech, speaker_id, use_speaker_id, speaker_ids_set)
            loss += cyclic_loss
            out["cyclic_loss"] = cyclic_loss.item()
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), out

    def cyclic_step(self, encoded, speech, speaker_id, use_speaker_id=True, speaker_ids_set=None):
        assert use_speaker_id
        assert speaker_ids_set is not None

        different_ids = []
        for spk in speaker_id:
            difference = speaker_ids_set - {spk.item()}
            if len(difference) == 0:
                different_id = spk # No other speaker to train on
            else:
                different_id = random.choice(list(difference))
            different_ids.append(different_id)
        
        different_ids = torch.tensor(different_ids).to(self.device).to(
            speaker_id.dtype)

        # Forward pass with different speaker
        output_diff = self.model.decode(encoded, different_ids, None)
        output_reconstructed = self.model(output_diff, speaker_id)

        # Cyclic consistency
        cyclic_loss = F.mse_loss(output_reconstructed, speech) * self.config['cyclic_lr_coef']
        return cyclic_loss
    
    def train_epoch(self, dataloader, use_speaker_id=True, stage=1, speaker_ids_set=None):
        self.model.train()
        epoch_loss = 0
        cyclic_loss = 0
        step_count = 0
        
        for batch in tqdm(dataloader, total=len(dataloader)):
            loss, out = self.train_step(batch, use_speaker_id, stage=stage,
                 speaker_ids_set=speaker_ids_set)
            epoch_loss += loss
            
            if stage == 2:
                cyclic_loss += out["cyclic_loss"]
            step_count += 1
        
        avg_loss = epoch_loss / len(dataloader)
        if stage == 2:
            avg_cyclic_loss = cyclic_loss / step_count
        self.scheduler.step(avg_loss)
        
        return avg_loss, {
            "loss": avg_loss,
            "cyclic_loss": avg_cyclic_loss if stage == 2 else None,}
    
    def validate(self, dataloader, use_speaker_id=True, stage=1, speaker_ids_set=None):
        self.model.eval()
        val_loss = 0
        cyclic_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader)):
                speech = batch["speech"].to(self.device)
                
                if use_speaker_id:
                    speaker_id = batch["speaker_id"].to(self.device)
                    reference_speech = None
                else:
                    speaker_id = None
                    reference_speech = batch["reference_speech"].to(self.device)
                
                encoded = self.model.encode(speech)
                output = self.model.decode(encoded, speaker_id, reference_speech)
                loss = F.mse_loss(output, speech)
                val_loss += loss.item()

                if stage == 2:
                    step_cyclic_loss = self.cyclic_step(
                        encoded,
                        speech,
                        speaker_id, use_speaker_id, speaker_ids_set).item()
                    cyclic_loss += step_cyclic_loss
                    val_loss += step_cyclic_loss
        
        return val_loss / len(dataloader), {
            "loss": val_loss / len(dataloader),
            "cyclic_loss": cyclic_loss / len(dataloader) if stage == 2 else None
        }


# Example usage
def create_model(model_config):
    model = VoiceConversionModel(
        model_config,
    )
    
    return model