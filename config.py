from enum import Enum
from model.detr import DETR_Model
from model.resnet import RESNET34_Reg_Model
from model.transformer_encoder import Transformer_Encoder_Model
import model.models as models
import model
import transforms
from targets import Target

class Config(Enum):

    DETR = (
        'detr',
        DETR_Model(),
        transforms._resnet34_transform(),
        Target.ALL,
        True
    )

    N_RES = (
        'n_res',
        RESNET34_Reg_Model(),
        transforms._resnet34_transform(sobel=True),
        Target.N_RES,
        True
    )

    def __init__(self, key, model, transform, target, multi_resonance):
        self.key = key
        self.model = model
        self.transform = transform
        self.target = target
        self.multi_resonance = multi_resonance

    def get_key(self):
        return self.key

    def get_model(self):
        return self.model
        
    def get_transform(self):
        return self.transform

    def get_target(self):
        return self.target

    def is_multi_resonance(self):
        return self.multi_resonance
        
    @classmethod
    def from_key(cls, key):
        for config in cls:
            if config.key == key:
                return config
        raise ValueError(f'No config with key {key}')