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

# set model config
config = config.Config.RESNET34_SEGMENTATION

# set prediction target
target = targets.Target.SEGMENTS

# set training parameters and hyperparameters
params = {
    'seed': 22,
    'data_filename': 'single_10.json',
    'is_data_compressed': False,
    'num_workers': 8,
    'batch_size': 2,
    'n_epochs': 250,
    'lr': 2e-4,
    'weight_decay': 1e-4,
    'is_multi_resonance': config.is_multi_resonance(),
    'model': config.get_model(),
    'transform': config.get_transform(),
    'target': target
}

# train the model and output training results
train.train(params)