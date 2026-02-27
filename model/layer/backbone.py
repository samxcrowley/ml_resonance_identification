from torch import nn
import torchvision
import torchvision.models

class Backbone(nn.Module):

    def __init__(self, d_backbone, d_transformer):

        super().__init__()

        self.resnet = torchvision.models.resnet101(
            weights='DEFAULT',
            norm_layer=torchvision.ops.FrozenBatchNorm2d
        )

        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        self.proj = nn.Conv2d(
            in_channels=d_backbone,
            out_channels=d_transformer,
            kernel_size=1
        )

    def forward(self, x):

        x = self.resnet(x) # [N, C, H, W]
        x = self.proj(x) # [N, D, H, W]
        x = x.permute(0, 2, 3, 1) # [N, H, W, D]
        x = x.flatten(1, 2) # [N, HW, D]

        return x