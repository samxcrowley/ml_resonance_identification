from torch import nn
import torchvision.models

class RESNET34_Reg_Model(nn.Module):

    def __init__(self, out_size=1):

        super().__init__()

        self.model = torchvision.models.resnet34()
        fc_in = self.model.fc.in_features
        fc_out = out_size
        self.model.fc = nn.Linear(fc_in, fc_out)

    def forward(self, x):
        return self.model(x)