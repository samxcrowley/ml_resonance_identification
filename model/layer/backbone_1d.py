from torch import nn

class Backbone1D(nn.Module):

    def __init__(self, d_transformer, n_in_channels=2, n_features=8):

        super().__init__()

        # input: [NC, n_in_channels, 512]
        # processes each channel independently along the energy axis

        self.conv1 = nn.Sequential(
            nn.Conv1d(n_in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(128, d_transformer, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(d_transformer),
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool1d(n_features)

    def forward(self, x):

        # x: [NC, 2, E]
        
        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        x = self.conv4(x)

        x = self.pool(x)

        x = x.permute(0, 2, 1)

        # [NC, n_features, d_transformer]
        return x