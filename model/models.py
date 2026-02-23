import torch.nn as nn
import torchvision.models

def _resnet34_model():

    model = torchvision.models.resnet34()

    fc_in = model.fc.in_features
    fc_out = 1
    model.fc = nn.Linear(fc_in, fc_out)

    return model