from enum import Enum
import torch
from torch import nn
import torchvision.models
import train
import transforms
import data
import targets
import model.models as models
import config
import header

# set training parameters and model hyperparameters
params = {
    'seed': 22,

    'data_filename': '10res_training.gz',
    'header_filename': 'o16_header.json',
    'data_is_compressed': True,

    'num_workers': 8,
    'batch_size': 64,
    'n_epochs': 100,
    'lr': 2e-4,
    'weight_decay': 1e-4
}

# set config (model + transforms)
config = config.Config.RESNET34
model = config.get_model()
transform = config.get_transform()

# input data
# header = header.Header(params['header_filename'])

# set prediction target
target = targets.Target.N_RESONANCES

# train the model and output training results
train.train(params, transform, target, model)