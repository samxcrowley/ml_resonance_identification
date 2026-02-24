from enum import Enum
import torch

class Target(Enum):

    ENERGY_LEVEL = 0
    GAMMA_TOTAL = 1
    MASK = 2

    def __init__(self, n_classes=0):
        self.n_classes = n_classes

    def get(self, targets):
        
        if (self == Target.ENERGY_LEVEL or
                        self == Target.GAMMA_TOTAL):

            target = targets[:, :, Target.ENERGY_LEVEL.value]

            return target

        elif (self == Target.MASK):

            target = targets[:, 0, self.value]
            print(target)
            # target = target.unsqueeze(1)

            return target