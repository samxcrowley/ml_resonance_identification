import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):

    def __init__(self, max_len, d_transformer):

        super().__init__()
        
        self.pos_enc = nn.Parameter(torch.zeros(max_len, d_transformer))

    def forward(self, x):

        N, S, D = x.size()

        enc = self.pos_enc[:S]
        enc = enc.unsqueeze(0) # [1, S, D]
        enc = enc.expand(N, -1, -1) # [N, S, D]

        return enc