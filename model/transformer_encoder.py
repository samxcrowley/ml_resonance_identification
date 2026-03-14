import torch
from torch import nn

from model.layer.encoder_layer import EncoderLayer

class Transformer_Encoder_Model(nn.Module):

    def __init__(self, d_model, n_hidden, n_head, n_layers, dropout_p):

        super().__init__()

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  n_hidden=n_hidden,
                                                  n_head=n_head,
                                                  dropout_p=dropout_p)
                                    for _ in range(n_layers)])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, pos_enc):

        for layer in self.layers:
            x = layer(x, pos_enc)

        x = self.norm(x)

        return x