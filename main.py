from enum import Enum
import torch
from torch import nn
import torchvision.models
import train
import transforms
import data
import targets
import model.models as models
from config import Config
from targets import Target
import header
import sys

def enum_from_key(enum, key):
    for config in enum:
        if config.key == key:
            return config
    raise ValueError(f'No {enum} member with key {key}')

# set model config
config_key = sys.argv[1]
config = enum_from_key(Config, config_key)

model = config.get_model()
transform = config.get_transform()
is_multi_resonance = config.is_multi_resonance()

# set prediction target
target_key = sys.argv[2]
target = enum_from_key(Target, target_key)

print('\n--------------------------------\n')
if is_multi_resonance:
    print(f'Multi resonance task starting...')
else:
    print(f'Single resonance task starting...')
print(f'\nConfig: {config}')
print(f'Target: {target}')
print('\n--------------------------------\n')

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
    'is_multi_resonance': is_multi_resonance,
    'model': model,
    'transform': transform,
    'target': target
}

# train the model and output training results
train.train(params)