from enum import Enum
import torch
from torch import nn

class Target(Enum):

    ENERGY_LEVEL = 0
    GAMMA_TOTAL = 1
    DETR_TARGET = 2

    def get(self, targets):

        if self == Target.DETR_TARGET:

            # TODO: detr target operation

            return None

        target = targets[:, :, self.index]

        return target

    def loss_fn(self):

        if self == Target.DETR_TARGET:
            
            # TODO: detr loss function

            return None

        return nn.CrossEntropyLoss()