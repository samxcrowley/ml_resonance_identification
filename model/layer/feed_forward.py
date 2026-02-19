from torch import nn

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, n_hidden, dropout_p):

        super().__init__()

        self.linear1 = nn.Linear(d_model, n_hidden)
        self.linear2 = nn.Linear(n_hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):

        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x