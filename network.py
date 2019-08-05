import torch
import torch.nn as nn


class CharacterClassifier(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        # We got 18 characters in total: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ,+, -, *, /, x, y, m, =)
        output_dim = 18
        hidden_dim = 50

        self.layers = []
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = self.relu(layer(out))
        return self.sigmoid(out)
