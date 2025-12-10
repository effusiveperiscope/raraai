import torch
import torch.nn as nn
import torch.nn.functional as F
from mambapy.mamba import Mamba, MambaConfig

class ValenceArousalPredictor(nn.Module):
    """
    Predicts valence and arousal (both in range [1,7]) from Whisper-medium features.
    
    Architecture:
    - Temporal CNN layers for local feature extraction
    - Bidirectional LSTM for sequential modeling
    - Separate regression heads for valence and arousal
    - Normalizes targets to [-1, 1] during training, denormalizes outputs
    """
    
    def __init__(
        self,
        input_dim=512,
        hidden_dim=256,
        mamba_layers=2,
        dropout=0.3,
        num_cnn_layers=2
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Temporal CNN layers for local feature extraction
        self.cnn_layers = nn.ModuleList()
        for i in range(num_cnn_layers):
            self.cnn_layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        
        # Mamba for sequential modeling
        mamba_config = MambaConfig(d_model=hidden_dim, n_layers=mamba_layers)
        self.mamba = Mamba(mamba_config)
        
        # Attention mechanism for temporal pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Separate regression heads
        self.valence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.arousal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def normalize_targets(self, values):
        """Normalize [1,7] to [-1,1]"""
        return (values - 4.0) / 3.0
    
    def denormalize_outputs(self, values):
        """Denormalize [-1,1] to [1,7]"""
        return values * 3.0 + 4.0
    
    def forward(self, x, return_normalized=False):
        """
        Args:
            x: [Batch, Sequence, 1024] - Whisper features
            return_normalized: If True, returns values in [-1,1], else [1,7]
            
        Returns:
            valence: [Batch, 1] - Predicted valence
            arousal: [Batch, 1] - Predicted arousal
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_proj(x)  # [B, S, hidden_dim]
        
        # Apply temporal CNNs (need to transpose for Conv1d)
        x_cnn = x.transpose(1, 2)  # [B, hidden_dim, S]
        for cnn in self.cnn_layers:
            x_cnn = cnn(x_cnn) + x_cnn  # Residual connection
        x = x_cnn.transpose(1, 2)  # [B, S, hidden_dim]
        
        # Mamba processing
        mamba_out = self.mamba(x)  # [B, S, hidden_dim]
        
        # Attention-based pooling
        attn_weights = self.attention(mamba_out)  # [B, S, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * mamba_out, dim=1)  # [B, hidden_dim]
        
        # Predict valence and arousal (normalized to [-1,1])
        valence_norm = torch.tanh(self.valence_head(context))  # [B, 1]
        arousal_norm = torch.tanh(self.arousal_head(context))  # [B, 1]
        
        if return_normalized:
            return valence_norm, arousal_norm
        
        # Denormalize to [1,7] range
        valence = self.denormalize_outputs(valence_norm)
        arousal = self.denormalize_outputs(arousal_norm)
        
        # Clamp to valid range
        valence = torch.clamp(valence, 1.0, 7.0)
        arousal = torch.clamp(arousal, 1.0, 7.0)
        
        return valence, arousal


class ConcordanceCorrelationLoss(nn.Module):
    """
    Concordance Correlation Coefficient (CCC) loss for regression.
    Often used in affective computing tasks.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        """Both pred and target should be normalized to similar scales"""
        pred_mean = torch.mean(pred)
        target_mean = torch.mean(target)
        
        pred_var = torch.var(pred)
        target_var = torch.var(target)
        
        covariance = torch.mean((pred - pred_mean) * (target - target_mean))
        
        ccc = (2 * covariance) / (pred_var + target_var + (pred_mean - target_mean) ** 2 + 1e-8)
        
        return 1 - ccc  # Loss (minimize)