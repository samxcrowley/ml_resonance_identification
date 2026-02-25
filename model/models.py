import torch.nn as nn
import torchvision.models
import model.encoder

# N: batch size
# C: CNN channels
# D: transformer dimensionality
# K: max resonances
# T: num prediction targets
def _detr_model():

    # CNN
    # input shape: [N, 1, E, A]
    # output shape: [N, C, H, W]

    _cnn = torchvision.models.resnet34()


    # FLATTEN
    # input shape: [N, C, H, W]
    # output shape: [N, HW, C]


    # 1x1 CONV
    # input shape: [N, HW, C]
    # output shape: [N, HW, D]


    # TRANSFORMER ENCODER
    # input shape: [N, HW, D]
    # output shape: [N, HW, D]


    # OBJECT QUERIES
    # shape: [N, K, D]


    # TRANSFORMER DECODER
    # input shape: [N, HW, D]
    # output shape: [N, HW, D]


    # PREDICTION HEADS
    

    return None

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