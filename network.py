import torch
import torch.nn as nn


class CharacterClassifier(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim=18):
        super().__init__()
        # We got 18 characters in total: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ,+, -, *, /, x, y, m, =)
        channels, height, width = input_dim
        self.layers = []
        self.layers.append(nn.Conv2d(channels, height*width, (5,5)))
        height, width = (1 + (height - 5), 1 + (width - 5))
        last_idx = len(hidden_layers)-1
        for i in range(last_idx):
            self.layers.append(nn.Conv2d(hidden_layers[i], hidden_layers[i+1], (2,2)))
            height, width = (1 + (height - 2), 1 + (width - 2))
        self.layers.append(nn.Linear(hidden_layers[last_idx]*height*width, 120))
        self.layers.append(nn.Linear(120, 84))
        self.layers.append(nn.Linear(84, output_dim))

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = self.relu(layer(out))
        return self.sigmoid(out)
