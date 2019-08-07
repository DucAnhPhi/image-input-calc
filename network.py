import torch
import torch.nn as nn


class CharacterClassifier(nn.Module):

    def __init__(self, input_dim, hidden_layers, output_dim=18):
        super(CharacterClassifier, self).__init__()
        # We got 18 characters in total: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ,+, -, *, /, x, y, m, =)
        channels, height, width = input_dim
        layers = []
        layers.append(nn.Conv2d(channels, height*width, (5,5)))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(height * width, hidden_layers[0], (5, 5)))
        height, width = (1 + (height - 5), 1 + (width - 5))
        layers.append(nn.ReLU())
        height, width = (1 + (height - 5), 1 + (width - 5))
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Conv2d(hidden_layers[i-1], hidden_layers[i], (2,2)))
            layers.append(nn.ReLU())
            height, width = (1 + (height - 2), 1 + (width - 2))

        self.out_layer = nn.Linear(hidden_layers[len(hidden_layers)-1]*height*width, output_dim)
        self.layers = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        out = out.reshape(out.shape[0], out.shape[1]*out.shape[2]*out.shape[3])
        return self.sigmoid(self.out_layer(out))
