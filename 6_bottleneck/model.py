import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleBottleneckModel(nn.Module):
    def __init__(self,
        in_dim = 768,
        bottleneck_dim = 32,
        asr_dim = 512,
        out_dim = 768):
        super().__init__()
        
        # We could do a sequence bottleneck with mambas too?
        self.speech_encoder = nn.Linear(in_dim, bottleneck_dim)
        self.speech_decoder = nn.Linear(bottleneck_dim, out_dim)
        self.asr_encoder = nn.Linear(asr_dim, out_dim)
        self.sequence_decoder = nn.Linear(out_dim, out_dim)

        # Sinusoidal position encoding needed?

    # We could also use ASR features as a "teacher" and use a knowledge
    # distillation approach instead of just directly using them
    # But in that case why not just use an autoencoder?
    def forward(self, speech_feats, asr_feats):
        # Pad them out..

        # Bottlenecked representation
        bottlenecked = self.speech_encoder(speech_feats)
        bottlenecked = self.speech_decoder(speech_feats)

        # ASR encode
        asr_encoder = self.asr_encoder(asr_feats)

        # TODO: We could use MultiheadAttention
        # - Q is in speech feats dim
        # - K, V in ASR feats dim
        # Then add that onto
        

        return self.sequence_decoder(bottlenecked + asr_encoder)