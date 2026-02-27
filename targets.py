from enum import Enum
from torch import nn
from model.detr import DETR_Loss

class Target(Enum):

    CLASS = 0
    ENERGY = 1
    GAMMA_TOTAL = 2
    N_RES = 3
    DETR = 4
    DETR_NO_GAMMA_TOTAL = 5
    
    def get_targets(self, targets):

        if self == Target.ENERGY:
            return targets['energy'][:, 0, :]
        elif self == Target.N_RES:
            return targets['n_res'].unsqueeze(1)
        elif self == Target.DETR:
            return targets

    def get_loss_fn(self):

        if self == Target.ENERGY or self == Target.N_RES:
            return nn.MSELoss()

        if self == Target.DETR:
            return DETR_Loss()

        if self == Target.DETR_NO_GAMMA_TOTAL:
            return DETR_Loss(include_gamma_total=False)