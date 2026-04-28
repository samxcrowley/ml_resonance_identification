from torch import nn
import torch.nn.functional as F

class Backbone(nn.Module):

    def __init__(self, d_transformer, n_in_channels=2, norm="instance"):

        super().__init__()

        # input: [batch, 2, 512, n_channels]
        # n_channels: n_pp_combos * n_angles

        if norm == "batch":
            norm_layer = nn.BatchNorm2d
        elif norm == "instance":
            norm_layer = nn.InstanceNorm2d
        else:
            raise ValueError(f"Unknown norm type: {norm!r} (expected 'batch' or 'instance')")

        self.conv1 = nn.Sequential(
            nn.Conv2d(n_in_channels, 32, kernel_size=5, stride=(1, 3), padding=2),
            norm_layer(32),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=(1, 3), padding=2),
            norm_layer(64),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=(2, 2), padding=1),
            norm_layer(128),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, d_transformer, kernel_size=3, stride=(2, 2), padding=1),
            norm_layer(d_transformer),
            nn.ReLU()
        )

    def forward(self, x):

        mask = x[:, 1:2, :, :]
        mask = F.max_pool2d(mask, kernel_size=5, stride=(1, 3), padding=2)
        mask = F.max_pool2d(mask, kernel_size=5, stride=(1, 3), padding=2)
        mask = F.max_pool2d(mask, kernel_size=3, stride=(2, 2), padding=1)
        mask = F.max_pool2d(mask, kernel_size=3, stride=(2, 2), padding=1)
        mask = mask.flatten(1)

        key_padding_mask = (mask == 0)
        # Ensure at least one token is unmasked per sample to prevent all-inf attention
        all_masked = key_padding_mask.all(dim=1, keepdim=True)
        key_padding_mask = key_padding_mask & ~all_masked

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.flatten(2)
        x = x.permute(0, 2, 1)

        # [N, HW, D]
        return x, key_padding_mask