import torch
import torch.nn as nn

class PitchPredictorMLP(nn.Module):
    def __init__(self, input_size=768, hidden_size=256, output_size=1, n_layers=6):
        super(PitchPredictorMLP, self).__init__()
        self.activ = nn.ReLU
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            self.activ(),
            *(nn.Sequential(nn.Linear(hidden_size, hidden_size), self.activ()) for _ in range(n_layers)),
            self.activ(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)