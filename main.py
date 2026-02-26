from enum import Enum
import torch
from torch import nn
import torchvision.models
import train
import transforms
import data
import model.models as models
from config import Config
import header
import sys
import json

with open('params.json', 'r') as f:
    params = json.load(f)

config = params['config']

print('\n--------------------------------\n')
print(f'Config {config} loaded...')
print('\n--------------------------------\n')

# train the model and output training results
train.train(params)