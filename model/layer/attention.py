import math
from torch import nn

class Attention(nn.Module):

    def __init__(self):

        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):

        batch_size, n_head, seq_length, d_tens = q.size()

        k_t = k.transpose(2, 3)
        score = (q @ k_t) / math.sqrt(d_tens)
        
        score = self.softmax(score)

        v = score @ v

        return v, score