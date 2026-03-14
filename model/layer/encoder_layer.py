from torch import nn
import torch.nn.functional as F

class EncoderLayer(nn.Module):

    def __init__(self, d_model, n_hidden, n_head, dropout_p):

        super().__init__()

        self.attention = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout_p)

        self.linear1 = nn.Linear(d_model, n_hidden)
        self.dropout2 = nn.Dropout(dropout_p)
        self.linear2 = nn.Linear(n_hidden, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, pos_enc):

        # sublayer 1: attention

        residual = x
        x = self.norm1(x)

        q = (x + pos_enc)
        k = (x + pos_enc)
        v = x

        att_out, _ = self.attention(q, k, v)
        x = residual + self.dropout1(att_out)

        # sublayer 2: feed-forward

        residual = x
        out = self.norm2(x)

        ff_out = self.linear1(out)
        ff_out = F.relu(ff_out)
        ff_out = self.dropout2(ff_out)
        ff_out = self.linear2(ff_out)

        out = residual + ff_out

        return out