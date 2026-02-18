from torch import nn

from model.layer.attention import Attention

class MultiheadAttention(nn.Module):

    def __init__(self, d_model, n_head):

        super().__init__()

        self.n_head = n_head
        self.attention = Attention()
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):

        q = self.w_q(q)
        q = self.split(q)

        k = self.w_k(k)
        k = self.split(k)

        v = self.w_v(v)
        v = self.split(v)

        out, score = self.attention(q, k, v)
        out = self.concat(out)
        out = self.w_concat(out)

        return out

    def split(self, x):

        '''
        Splits a tensor by number of attention heads.

        Input shape: [batch_size, seq_length, d_model]
        Output shape: [batch_size, n_head, seq_length, d_tens]

        where d_tens = d_model / n_head
        '''

        batch_size, seq_length, d_model = x.size()

        d_tens = d_model // self.n_head

        x = x.view(batch_size, seq_length, self.n_head, d_tens)
        x = x.transpose(1, 2)

        return x

    def concat(self, x):

        '''
        Concatenates a tensor that has been split by attention heads.

        Input shape: [batch_size, n_head, seq_length, d_tens]
        Output shape: [batch_size, seq_length, d_model]

        where d_tens = d_model / n_head
        '''

        batch_size, n_head, seq_length, d_tens = x.size()

        d_model = n_head * d_tens

        x = x.transpose(1, 2)
        x = x.contiguous() # ensure tensor contents are contiguous after view operations
        x = x.view(batch_size, seq_length, d_model)

        return x