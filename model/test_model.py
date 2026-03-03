import torch
import torch.nn as nn

class Test_Model(nn.Module):

    def __init__(self, n_queries=5):

        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1, n_queries * 4)

    def forward(self, x):

        N = x.size(0)
        x = self.pool(x)

        x = x.view(N, -1)

        x = self.fc(x)

        x = x.view(N, 5, 4)

        return {
            'class': x[:, :, :2],
            'energy': x[:, :, 2:3],
            'gamma_total': x[:, :, 3:4]
        }

    def get_optimiser(self, lr, weight_decay):

        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)