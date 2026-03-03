import torch
from torch import nn

from model.layer.decoder_layer import DecoderLayer

class Transformer_Decoder_Model(nn.Module):

    def __init__(self, d_model, n_hidden, n_head, n_layers, dropout_p):

        super().__init__()

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  n_hidden=n_hidden,
                                                  n_head=n_head,
                                                  dropout_p=dropout_p)
                                    for _ in range(n_layers)])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, enc, pos_enc, query_pos):

        for layer in self.layers:

            x, cross_att = layer(x, enc, pos_enc, query_pos)

        x = self.norm(x)

        return x, cross_att