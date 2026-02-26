from enum import Enum
from torch import nn

class Target(Enum):

    CLASS = 0
    ENERGY = 1
    GAMMA_TOTAL = 2
    N_RES = 3
    DETR = 4
    
    def get_targets(self, targets):

        if self == Target.ENERGY:
            return targets['energy'][:, 0, :]
        elif self == Target.N_RES:
            return targets['n_res'].unsqueeze(1)

    def get_loss_fn(self):

        if self == Target.ENERGY or self == Target.N_RES:
            return nn.MSELoss()