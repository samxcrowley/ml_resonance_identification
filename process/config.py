from enum import Enum
from model.detr import DETR_Model
from model.regression import Regression_Model
from model.transformer_encoder import Transformer_Encoder_Model
import model
import process.transforms as transforms
from process.targets import Target

class Config(Enum):

    DETR = (
        'detr',
        DETR_Model,
        transforms._detr_transform(),
        Target.DETR
    )

    def __init__(self, key, model, transform, target):
        self.key = key
        self.model = model
        self.transform = transform
        self.target = target

    def get_key(self):
        return self.key

    def get_model(self, header, params):
        return self.model(header, params)
        
    def get_transform(self, inference=False):

        if inference:
            return transforms._detr_transform(noise_sigma_log10=0.0, amplitude_scale=0.0)

        return self.transform

    def get_target(self):
        return self.target
        
    @classmethod
    def from_key(cls, key):
        for config in cls:
            if config.key == key:
                return config
        raise ValueError(f'No config with key {key}')