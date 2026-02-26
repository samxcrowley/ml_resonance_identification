from torch import nn
import torch.nn.functional as F

class DecoderLayer(nn.Module):

    def __init__(self, d_model, n_hidden, n_head, dropout_p):

        super().__init__()

        self.self_attention = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attention = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, n_hidden)
        self.dropout = nn.Dropout(dropout_p)
        self.linear2 = nn.Linear(n_hidden, d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc, pos_enc, query_pos):

        # sublayer 1: self-attention

        x = self.norm1(x)

        q = (x + query_pos)
        k = (x + query_pos)
        v = x

        self_att_out, _ = self.self_attention(q, k, v)
        x = x + self_att_out

        # sublayer 2: cross-attention

        x = self.norm2(x)

        q = (x + query_pos)
        k = (enc + pos_enc)
        v = enc

        cross_att_out, cross_att = self.cross_attention(q, k, v)
        x = x + cross_att_out

        # sublayer 3: feed-forward

        x = self.norm3(x)

        ff_out = self.linear1(x)
        ff_out = F.relu(ff_out)
        ff_out = self.dropout(ff_out)
        ff_out = self.linear2(ff_out)

        x = x + ff_out

        return x, cross_att