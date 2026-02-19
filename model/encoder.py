import torch
from torch import nn

from model.layer.encoder_layer import EncoderLayer

class Encoder(nn.Module):

    def __init__(self, d_model, max_len, n_hidden, n_head, n_layers, dropout_p):

        super().__init__()

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  n_hidden=n_hidden,
                                                  n_head=n_head,
                                                  dropout_p=dropout_p)
                                    for _ in range(n_layers)])

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, 1)
        )

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        x = x.mean(dim=1)

        x = self.mlp(x)

        x = torch.sigmoid(x)

        return x