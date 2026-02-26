from enum import Enum
from torch import nn

class Target(Enum):

    ALL = 0
    N_RES = 1
    
    def get_targets(self, targets):

        if self == Target.N_RES:
            return targets[2].unsqueeze(1)

    def get_loss_fn(self):

        if self == Target.N_RES:
            return nn.MSELoss()