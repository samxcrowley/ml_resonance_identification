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

# set training parameters and model hyperparameters
params = {
    'seed': 22,

    'data_path': 'data/1000.json',
    'data_is_compressed': False,

    'num_workers': 8,
    'batch_size': 32,
    'n_epochs': 250,
    'lr': 2e-4,
    'weight_decay': 1e-4
}

# set config (model + transforms)
config = config.Config.RESNET34

model = config.get_model()
transform = config.get_transform()

# set prediction target
target = targets.Target.SINGLE_RES_ENERGY_LEVEL

# train the model and output training results
train.train(params, transform, target, model)