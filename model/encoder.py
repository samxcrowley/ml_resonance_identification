from torch import nn

from model.layer.encoder_layer import EncoderLayer

class Encoder(nn.Module):

    def __init__(self, d_model, max_len, n_hidden, n_head, n_layers, dropout_p, device):

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