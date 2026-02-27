from torch import nn
import torchvision.models

class Regression_Model(nn.Module):

    def __init__(self):

        super().__init__()

        # self.model = torchvision.models.resnet34(
        #     weights='DEFAULT'
        # )
        # fc_in = self.model.fc.in_features
        # fc_out = out_size
        # self.model.fc = nn.Linear(fc_in, fc_out)

        self.model = torchvision.models.efficientnet_b0(weights='DEFAULT')
        fc_in = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(fc_in, out_size)

    def forward(self, x):
        return self.model(x)