import torch.nn as nn
import torchvision.models
import model.encoder

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

def _encoder_model(d_model=32, pool_kernel_size=2, n_hidden=32, n_head=4, n_layers=6, dropout_p=0.0):

    _model = model.encoder.Encoder(d_model, pool_kernel_size, n_hidden, n_head, n_layers, dropout_p)

    return _model