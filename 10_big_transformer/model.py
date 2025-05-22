import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from modeling.musicgen.modeling import MyMusicgenForCausalLM
from transformers.models.musicgen.configuration_musicgen import MusicgenDecoderConfig
from modeling.common import RotaryPositionalEmbedding
from commons import create_sequence_mask
from einops import rearrange

# Really dumb encoder only model
class MyEncoderOnlyModel(torch.nn.Module):
    def __init__(self, config : OmegaConf):
        super().__init__()
        self.in_proj = torch.nn.Linear(config.in_dim, config.hidden_dim)
        self.pitch_emb = torch.nn.Linear(1, config.hidden_dim)
        self.pos_emb = RotaryPositionalEmbedding(config.hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                config.hidden_dim,
                nhead=config.n_heads,
                dim_feedforward=config.hidden_dim,
                dropout=config.dropout,
                batch_first=True
        ), config.n_layers)
        self.out_proj = torch.nn.Linear(config.hidden_dim, config.out_dim)

    def forward(self, spch, pitch, target_len):
        x = self.in_proj(spch)
        x = self.pitch_emb(pitch)
        x = self.pos_emb(x)
        x = self.encoder(x)
        x = F.interpolate(x, size=target_len, mode="linear")
        x = self.out_proj(x)
        return x

class MyEncoder(torch.nn.Module):
    def __init__(self, config : OmegaConf):
        super().__init__()
        self.in_emb = torch.nn.Embedding(config.codes, config.hidden_dim)
        self.pitch_emb = torch.nn.Linear(1, config.hidden_dim)
        self.pos_emb = RotaryPositionalEmbedding(config.hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                config.hidden_dim,
                nhead=config.n_heads,
                dim_feedforward=config.hidden_dim,
                dropout=config.dropout,
                batch_first=True
        ), config.n_layers)
        self.out_proj = torch.nn.Linear(config.hidden_dim, config.out_dim)

    def forward(self, spch, pitch, len_mask=None):
        x = self.in_emb(spch)
        x = self.pitch_emb(pitch)
        x = self.pos_emb(x)
        # SRC_KEY_PADDING_MASK SHOULD BE FALSE FOR ATTENDED POSITIONS
        x = self.encoder(x, src_key_padding_mask=~len_mask)
        x = self.out_proj(x)
        if len_mask is None:
            return x
        x = x * len_mask.unsqueeze(-1)
        return x

class MyModel(torch.nn.Module):
    def __init__(self, config : OmegaConf):
        super().__init__()
        self.encoder = MyEncoder(config.encoder)
        self.decoder = MyMusicgenForCausalLM(
        MusicgenDecoderConfig(
            vocab_size = config.decoder.vocab_size,
            num_codebooks=config.decoder.num_codebooks,
            pad_token_id=config.decoder.pad_token_id,
            bos_token_id=config.decoder.bos_token_id,
            decoder_start_token_id=config.decoder.bos_token_id,
            hidden_size=config.decoder.hidden_size,
            is_encoder_decoder=True # we're providing our own encoder
        ))
        self.config = config

    def forward(self, 
            content_tokens,
            content_interp_pitch,
            content_seq_lens,
            acoustic_codes,
            acoustic_codes_lens,
        ):
        encoder_hidden_states = self.encoder(content_tokens, content_interp_pitch, 
            len_mask=create_sequence_mask(content_seq_lens))
        input_ids = acoustic_codes

        # don't prepend BOS here - shift_tokens_right should do this automatically
        # (however it's only used with labels so at generation time we have to do it ourselves)
        labels = input_ids # (batch_size, num_sequence, num_codebooks)

        decoded = self.decoder(input_ids=None, # This allows the model to perform internal shifting of labels
            attention_mask=create_sequence_mask(acoustic_codes_lens),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=create_sequence_mask(content_seq_lens),
            labels=labels)

        # Input IDs ((b d), n)
        # Output logits ((b d), n, codebook_dim)
        return decoded

    # not sure if this works with batched input...
    def generate(self,
                content_tokens,
                content_interp_pitch,
                content_seq_lens,
                max_len, # We don't have the original acoustic codes from the utterance...
                temperature=1.0
            ):
        encoder_hidden_states = self.encoder(content_tokens, content_interp_pitch, 
            len_mask=create_sequence_mask(content_seq_lens))

        input_ids = torch.full((self.decoder.config.num_codebooks, 1),
            self.decoder.config.bos_token_id).to(content_tokens.device)
        generated = torch.Tensor([]).to(content_tokens.device)

        for _ in range(max_len):
            output = self.decoder(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids).to(content_tokens.device).bool(),
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=create_sequence_mask(content_seq_lens),)

            # output.logits ((b d), n, codebook_dim)
            logits = output.logits[:, -1, :] / temperature # ((b d), codebook_dim)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            #import pdb; pdb.set_trace()

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            generated = torch.cat([generated, next_token], dim=-1)

        return generated


    # PROBLEM - We can't use this because MusicgenForCausalLM is a decoder-only model
    # def generate(self,
    #         content_tokens,
    #         content_interp_pitch,
    #         content_seq_lens,
    #         max_len, # We don't have the original acoustic codes from the utterance...
    #         **kwargs
    #     ):
    #     encoder_hidden_states = self.encoder(
    #         content_tokens, content_interp_pitch,
    #         len_mask=create_sequence_mask(content_seq_lens))
    #     output = self.decoder.generate(
    #         # (bsz * num_codebooks, seq_len)
    #         # - the default initialization behavior of input_ids is not intended for multiple codebooks
    #         # this is probably because musicgen doesn't support unprompted inference
    #         input_ids=torch.full((self.decoder.config.num_codebooks, 1),
    #            self.decoder.config.bos_token_id).to(content_tokens.device).unsqueeze(-1), 
    #         encoder_outputs={
    #             "hidden_states": encoder_hidden_states,
    #         },
    #         # encoder_hidden_states=encoder_hidden_states,
    #         # encoder_attention_mask=create_sequence_mask(content_seq_lens),
    #         max_length=max_len,
    #         **kwargs)
    #     return output

if __name__ == '__main__':
    from features import MyFeatures
    config = OmegaConf.load("configs/common.yaml")
    config.features.want = ["pitch", "content_tokens", "acoustic_codes"]
    myfeatures = MyFeatures(config, "cuda")

    features = myfeatures.extract_features("test.wav")

    model = MyEncoder(config.encoder)
    encoded = model(
        features["content_tokens"], # Content tokens from VeVo
        features["content_interp_pitch"]) # Pitch from RMVPE interpolated to content dim
    print("Features shape:", features["content_tokens"].shape) 
    print("Encoded shape:", encoded.shape)

    decoder : MyMusicgenForCausalLM
    decoder = MyMusicgenForCausalLM(
        MusicgenDecoderConfig(
            vocab_size = config.decoder.vocab_size,
            num_codebooks=config.decoder.num_codebooks,
            pad_token_id=config.decoder.pad_token_id,
            bos_token_id=config.decoder.bos_token_id,
            hidden_size=config.decoder.hidden_size
        )
    )
    decoder.eval()
    input_ids = rearrange(features["acoustic_codes"], "b n d -> (b d) n")
    decoded = decoder(input_ids=input_ids, encoder_hidden_states=encoded)
    print("Decoded shape:", decoded.logits.shape)