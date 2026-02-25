import torch
from torch import nn

from model.layer.encoder_layer import EncoderLayer

class Transformer_Encoder_Model(nn.Module):

    def __init__(self, d_model, n_hidden=2048, n_head=8, n_layers=6, dropout_p=0.0):

        super().__init__()

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  n_hidden=n_hidden,
                                                  n_head=n_head,
                                                  dropout_p=dropout_p)
                                    for _ in range(n_layers)])

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x