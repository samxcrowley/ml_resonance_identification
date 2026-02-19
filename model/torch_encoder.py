import torch
from torch import nn

class TorchEncoder(nn.Module):

    def __init__(self, d_model, pool_kernel_size, n_head, n_layers):

        super().__init__()

        self.proj = nn.Linear(1, d_model)

        self.pool = nn.AvgPool1d(kernel_size=pool_kernel_size)

        self.layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.layer, num_layers=n_layers)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, 1)
        )

    def forward(self, x):

        if x.dim() == 3:
            x = torch.flatten(x, start_dim=1, end_dim=-1)

        x = self.pool(x)

        # embedding
        x = x.unsqueeze(-1)
        x = self.proj(x)

        x = self.encoder(x)

        x = x.mean(dim=1)

        x = self.mlp(x)

        x = torch.sigmoid(x)

        return x