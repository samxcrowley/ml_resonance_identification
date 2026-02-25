import torch.nn as nn
import torchvision.models

def _resnet34_single_res_model():

    _model = torchvision.models.resnet34()

    fc_in = _model.fc.in_features
    fc_out = 10
    _model.fc = nn.Linear(fc_in, fc_out)

    return _model

def _resnet34_segmentation_model():

    _model = torchvision.models.resnet34()

    # TODO: configure segmentation model
    print('# TODO: configure segmentation model')

    return _model