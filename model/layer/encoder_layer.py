from torch import nn

from model.layer.multihead_attention import MultiheadAttention
from model.layer.feed_forward import PositionwiseFeedForward
from model.layer.layer_norm import LayerNorm

class EncoderLayer(nn.Module):

    def __init__(self, d_model, n_hidden, n_head, dropout_p):

        super().__init__()

        self.attention = MultiheadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=dropout_p)

        self.ffn = PositionwiseFeedForward(d_model=d_model, n_hidden=n_hidden, dropout_p=dropout_p)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=dropout_p)

    def forward(self, x):

        '''
        Composed of two sub-layers, multihead attention and feed-forward.
        Each sub-layer is followed by a residual connection and layer
        normalisation.
        '''

        x_copy = x

        # sublayer 1
        x = self.attention(q=x, k=x, v=x)
        x = self.dropout1(x)

        x = self.norm1(x + x_copy)

        x_copy = x

        # sublayer 2
        x = self.ffn(x)
        x = self.dropout2(x)

        x = self.norm2(x + x_copy)

        return x