import torch
from torch import nn
import torchvision.models
import train
import transforms
import load_data
import targets
import model.models as models

params = {
    'seed': 22,

    'data_path': 'data/10.json',
    'data_is_compressed': False,

    'num_workers': 8,
    'batch_size': 32,
    'n_epochs': 250,
    'lr': 2e-4,
    'weight_decay': 1e-4
}

model = models._resnet34_model()
transform = transforms._resnet34_transform(sobel=False)
target = targets.Target.SINGLE_RES_ENERGY_LEVEL_NORM

train.train(params, transform, target, model)