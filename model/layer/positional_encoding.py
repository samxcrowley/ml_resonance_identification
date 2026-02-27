import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):

    def __init__(self, max_len, d_transformer):

        super().__init__()
        
        pe = torch.zeros(max_len, d_transformer)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_transformer, 2).float() * (-math.log(10000.0) / d_transformer)
        )

        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)

        self.register_buffer('pos_enc', pe)

    def forward(self, x):

        N, S, D = x.size()

        enc = self.pos_enc[:S]
        enc = enc.unsqueeze(0) # [1, S, D]
        enc = enc.expand(N, -1, -1) # [N, S, D]

        return enc