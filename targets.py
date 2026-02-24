from enum import Enum
import torch
from torch import nn

class Target(Enum):

    ENERGY_LEVEL = ('energy_level', 0)
    GAMMA_TOTAL = ('gamma_total', 1)
    SEGMENTS = ('segments', 2)

    def __init__(self, key, index, n_classes=0):
        self.key = key
        self.index = index
        self.n_classes = n_classes

    def get(self, targets):

        if self == Target.SEGMENTS:

            # TODO: get segmentation target data
            print('# TODO: get segmentation target data')

            return None

        target = targets[:, :, self.index]

        return target

    def loss_fn(self):

        if self == Target.SEGMENTS:
            
            # TODO: segmentation loss function
            print('# TODO: segmentation loss function')

            return None

        return nn.CrossEntropyLoss()