import torch
from torch import nn
import torchvision.models

class Regression_Model(nn.Module):

    def __init__(self):

        super().__init__()

        # self.model = torchvision.models.resnet34(
        #     weights='DEFAULT'
        # )
        # fc_in = self.model.fc.in_features
        # self.model.fc = nn.Linear(fc_in, 1)

        self.model = torchvision.models.efficientnet_b0(weights='DEFAULT')
        fc_in = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(fc_in, 1)

    def forward(self, x):
        return self.model(x)

    def get_optimiser(self, lr, weight_decay):
        optimiser = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        return optimiser