from torch import nn
import torchvision
import torchvision.models

class Backbone(nn.Module):

    def __init__(self, d_backbone, d_transformer):

        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=(1, 2), padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, d_transformer, kernel_size=3, stride=(2, 2), padding=1),
            nn.BatchNorm2d(d_transformer),
            nn.ReLU()
        )

    def forward(self, x):

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        x = self.conv4(x)
        
        x = x.flatten(2)

        x = x.permute(0, 2, 1)

        # [N, HW, D]
        return x