from omegaconf import OmegaConf
import torch
from torch import nn
from modeling.commons import ConformerEncoder, ConformerBlock, generate_causal_mask, FiLMGenerator, SinusoidalPositionalEncoding
from modeling.grl import grad_reverse
from utils import random_subsample_segments

import pdb
import sys
import traceback
def custom_excepthook(exc_type, exc_value, exc_traceback):
    """
    Custom exception hook that prints the exception information
    and then drops into a pdb debugger session.
    """
    # First, print the exception information as Python normally would.
    # We use traceback.print_exception to ensure consistent formatting.
    print("An unhandled exception occurred:")
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("\nDropping into debugger...")

    # Then, drop into the pdb debugger.
    # The post_mortem function starts the debugger at the point of the exception.
    pdb.post_mortem(exc_traceback)

# Set the custom exception hook
sys.excepthook = custom_excepthook


class Encoder(nn.Module):
    def __init__(self, config: OmegaConf):
        super(Encoder, self).__init__()
        self.in_proj = nn.Linear(
            config.model.whisper_dim, config.model.d_encoder)

        self.sipe = SinusoidalPositionalEncoding(
            config.model.d_encoder)
        self.phoneme_emb = nn.Embedding(
            config.model.n_phonemes + 3, config.model.d_encoder)
        self.phoneme_head = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.model.d_encoder,
                nhead=config.model.phoneme_decoder.num_heads,
                dim_feedforward=config.model.phoneme_decoder.d_ff,
                dropout=config.model.phoneme_decoder.dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=config.model.phoneme_decoder.num_layers
        )
        self.phoneme_proj = nn.Linear(
            config.model.d_encoder, config.model.n_phonemes + 3) # bos, pad, eos

        self.prior_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.model.d_encoder,
                nhead=config.model.prior_encoder.num_heads,
                dim_feedforward=config.model.prior_encoder.d_ff,
                dropout=config.model.prior_encoder.dropout,
                activation='gelu',
                batch_first=True
        ),
            num_layers=config.model.prior_encoder.num_layers
        )
        self.prior_proj = nn.Linear(
            config.model.d_encoder, config.model.latent_dim * 2
        )

        self.bos_token_id = config.model.bos_token_id
        self.pad_token_id = config.model.pad_token_id
        self.eos_token_id = config.model.eos_token_id

    def generate(self, x, x_mask, max_len=100): # generate using greedy decoding
        x = self.in_proj(x) * x_mask.unsqueeze(-1)
        phone_logits, phone_hidden, generated_tokens = self.generate_phonemes(x, ~x_mask, max_len=max_len)  # Fixed mask inversion
        prior = self.prior_encoder(x, src_key_padding_mask=~x_mask)
        prior = self.prior_proj(prior)
        m_p, log_var_p = prior.chunk(2, dim=-1)
        return phone_logits, m_p, log_var_p

    def forward(self, x, x_mask, tgt, tgt_mask):
        x = self.in_proj(x) * x_mask.unsqueeze(-1)
        tgt = self.phoneme_emb(tgt)
        tgt = self.sipe(tgt) # Add position information to phonemes
        phone = self.phoneme_head(tgt, x, 
            tgt_mask=generate_causal_mask(tgt.size(1)).to(tgt.device),
            tgt_key_padding_mask=~tgt_mask, memory_key_padding_mask=~x_mask)
        phone_logits = self.phoneme_proj(phone)
        prior = self.prior_encoder(x, src_key_padding_mask=~x_mask)
        prior = self.prior_proj(prior)
        m_p, log_var_p = prior.chunk(2, dim=-1)
        return phone_logits, m_p, log_var_p

    def prior_only(self, x, x_mask):
        x = self.in_proj(x) * x_mask.unsqueeze(-1)
        prior = self.prior_encoder(x, src_key_padding_mask=~x_mask)
        prior = self.prior_proj(prior)
        m_p, log_var_p = prior.chunk(2, dim=-1)
        return m_p, log_var_p

    @torch.no_grad()
    def generate_phonemes(self, memory, memory_mask, max_len):
        """Autoregressive phoneme generation"""
        batch_size = memory.size(0)
        device = memory.device
        
        # Start with BOS token - keep as token IDs initially
        generated_ids = torch.full((batch_size, 1), self.bos_token_id, 
                                dtype=torch.long, device=device)
        
        # Store all hidden states and logits
        all_hidden = []
        all_logits = []
        
        for step in range(max_len - 1):
            # Convert current sequence to embeddings
            generated_emb = self.phoneme_emb(generated_ids)
            generated_emb = self.sipe(generated_emb)
            
            # Create causal mask for current sequence
            tgt_mask = generate_causal_mask(generated_emb.size(1)).to(device)
            
            # Forward pass
            hidden = self.phoneme_head(
                generated_emb, memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_mask  # Already inverted in generate()
            )
            
            # Get logits for last token
            logits = self.phoneme_proj(hidden[:, -1:])  # [B, 1, vocab_size]
            all_logits.append(logits)
            all_hidden.append(hidden[:, -1:])
            
            # Sample next token ID (not embedding!)
            next_token_id = torch.argmax(logits, dim=-1).squeeze(-1)  # [B, 1] -> [B]
            
            # Debug print
            #print(f"Step {step}: Generated token IDs: {next_token_id}")
            
            # Append to sequence (as token IDs)
            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(1)], dim=1)
            
            # Check for EOS (now comparing token IDs correctly)
            if hasattr(self, 'eos_token_id') and (next_token_id == self.eos_token_id).all():
                print(f"EOS reached at step {step}")
                break
        
        # Concatenate all logits and hidden states
        phone_logits = torch.cat(all_logits, dim=1)  # [B, seq_len-1, vocab_size]
        phone_hidden = torch.cat(all_hidden, dim=1)  # [B, seq_len-1, hidden_size]
        
        # Add BOS logits (dummy - could be zeros or compute separately)
        bos_logits = torch.zeros(batch_size, 1, phone_logits.size(-1), device=device)
        phone_logits = torch.cat([bos_logits, phone_logits], dim=1)
        
        # Convert final token sequence to embeddings for consistency
        final_generated_emb = self.phoneme_emb(generated_ids)
        
        return phone_logits, phone_hidden, generated_ids  # Return token IDs instead of embeddings

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.in_proj = nn.Linear(
            config.model.latent_dim,
            config.model.decoder.d_model
        )

        self.sipe = SinusoidalPositionalEncoding(
            config.model.decoder.d_model)
        self.spk_emb = nn.Embedding(config.model.n_speakers, config.model.d_spk)
        self.spk_cond = FiLMGenerator(config.model.d_spk, config.model.decoder.d_model)

        self.phone_proj = nn.Linear(
            config.model.n_phonemes + 3,
            config.model.decoder.d_model
        )
        self.phone_proj2 = nn.Linear(
            config.model.decoder.d_model,
            config.model.decoder.d_model
        )
        self.pitch_film = FiLMGenerator(1, config.model.decoder.d_model)
        self.in_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.model.decoder.d_model,
                nhead=config.model.decoder.num_heads,
                dim_feedforward=config.model.decoder.d_ff,
                dropout=config.model.decoder.dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=config.model.decoder.in_layers
        )
        self.spk_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.model.decoder.d_model,
                nhead=config.model.decoder.num_heads,
                dim_feedforward=config.model.decoder.d_ff,
                dropout=config.model.decoder.dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=config.model.decoder.spk_layers
        )
        self.pitch_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.model.decoder.d_model,
                nhead=config.model.decoder.num_heads,
                dim_feedforward=config.model.decoder.d_ff,
                dropout=config.model.decoder.dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=config.model.decoder.pitch_layers
        )
        self.out_proj = nn.Linear(
            config.model.decoder.d_model,
            config.model.whisper_dim
        )

    def forward(self, z, z_mask, phone_logits, phones_mask, spk_id, pitch=None):
        z = self.in_proj(z)
        z_orig = z

        phone_feats = self.phone_proj(phone_logits.detach())
        phone_feats = self.sipe(phone_feats)
        phone_feats = self.phone_proj2(phone_feats)

        z = self.sipe(z) # Add position information to latent for decoding

        y = self.in_decoder(z, phone_feats, 
            tgt_key_padding_mask=~z_mask, memory_key_padding_mask=~phones_mask)
        # this is where you would extract features

        gamma, beta = self.spk_cond(self.spk_emb(spk_id).unsqueeze(1))
        y = y + (y * gamma + beta)

        y = self.spk_decoder(y, y, 
            tgt_key_padding_mask=~z_mask, memory_key_padding_mask=~z_mask)

        if pitch is not None:
            pitch = pitch.log()
            gamma, beta = self.pitch_film(pitch.unsqueeze(2))
            y = y + (y * gamma + beta)

        y = self.pitch_decoder(y, y, 
            tgt_key_padding_mask=~z_mask, memory_key_padding_mask=~z_mask)
        y = self.out_proj(y)
        return y 

class SpeakerClassifier(nn.Module):
    def __init__(self, config):
        super(SpeakerClassifier, self).__init__()
        self.in_proj = nn.Linear(
            config.model.latent_dim,
            config.model.spk_classifier.d_model
        )
        self.encoder = ConformerEncoder(
            encoder_layer=ConformerBlock(
                d_model=config.model.spk_classifier.d_model,
                num_heads=config.model.spk_classifier.num_heads,
                d_ff=config.model.spk_classifier.d_ff,
                conv_kernel_size=config.model.spk_classifier.conv_kernel_size,
                dropout=config.model.spk_classifier.dropout
            ),
            num_layers=config.model.spk_classifier.num_layers
        )
        self.out_proj = nn.Linear(
            config.model.spk_classifier.d_model,
            config.model.n_speakers
        )
    
    def forward(self, x, x_mask):
        x = self.in_proj(x)
        x = self.encoder(x, src_key_padding_mask=x_mask)
        x = x.mean(dim=1)
        x = self.out_proj(x)
        return x

class PASIFVAE(nn.Module):
    def __init__(self, config):
        super(PASIFVAE, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.spk_classifier = SpeakerClassifier(config)
        self.config = config

    def generate(self, x, x_mask, spk_id, pitch=None, max_len=100):
        phoneme_logits, m_p, log_var_p = self.encoder.generate(x, x_mask, max_len=max_len)
        # Sample from prior
        z = m_p + torch.randn_like(m_p) * torch.exp(log_var_p / 2)

        phones_mask = torch.full((phoneme_logits.shape[0], phoneme_logits.shape[1]), True, dtype=torch.bool, device=phoneme_logits.device)
        y = self.decoder(z, x_mask, phoneme_logits, phones_mask, spk_id, pitch=pitch)
        return y, phoneme_logits

    def force_phonemes(self, x, x_mask, phoneme_logits, phones_mask, spk_id, pitch=None, grl_lambda=0.0):
        m_p, log_var_p = self.encoder.prior_only(x, x_mask)

        # Sample from prior
        z = m_p + torch.randn_like(m_p) * torch.exp(log_var_p / 2)

        y = self.decoder(z, x_mask, phoneme_logits, phones_mask, spk_id, pitch=pitch)
        return y

    def forward(self, x, x_mask, tgt, tgt_mask, spk_id, pitch=None, grl_lambda=0.0):
        phone_logits, m_p, log_var_p = self.encoder(x, x_mask, tgt, tgt_mask)

        # Sample from prior
        z = m_p + torch.randn_like(m_p) * torch.exp(log_var_p / 2)

        y = self.decoder(z, x_mask, phone_logits, tgt_mask, spk_id, pitch=pitch)

        m_p_segments, segment_mask = random_subsample_segments(m_p, x_mask,
            min_segment_len=self.config.model.spk_classifier.min_segment_len,
            max_segment_len=self.config.model.spk_classifier.max_segment_len)
        spk_logits = self.spk_classifier(grad_reverse(m_p_segments), segment_mask)

        return y, phone_logits, spk_logits, m_p, log_var_p

if __name__ == '__main__':
    # Load configuration
    config = OmegaConf.load('../configs/test1.yaml')

    # Create dummy inputs
    batch_size = 2
    seq_len = 64
    tgt_len = 32
    feature_dim = config.model.whisper_dim

    x = torch.randn(batch_size, seq_len, feature_dim)
    x_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    tgt = torch.randint(0, config.model.n_phonemes + 3, (batch_size, tgt_len))
    tgt_mask = torch.ones(batch_size, tgt_len, dtype=torch.bool)

    spk_id = torch.randint(0, config.model.n_speakers, (batch_size,))

    # Initialize model
    model = PASIFVAE(config)

    # Forward pass
    y, phone_logits, spk_logits, m_p, log_var_p = model(x, x_mask, tgt, tgt_mask, spk_id, grl_lambda=1.0)

    # Print output shapes
    print("Decoder Output Shape:", y.shape)
    print("Phoneme Logits Shape:", phone_logits.shape)
    print("Speaker Logits Shape:", spk_logits.shape)
    print("Prior Mean Shape:", m_p.shape)
    print("Prior Log Var Shape:", log_var_p.shape)