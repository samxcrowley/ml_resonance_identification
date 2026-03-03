from enum import Enum
from model.detr import DETR_Model
from model.regression import Regression_Model
from model.transformer_encoder import Transformer_Encoder_Model
import model.models as models
import model
import transforms
from targets import Target

class Config(Enum):

    DETR = (
        'detr',
        DETR_Model,
        transforms._cnn_transform(sobel=True),
        Target.DETR
    )

    def __init__(self, key, model, transform, target):
        self.key = key
        self.model = model
        self.transform = transform
        self.target = target

    def get_key(self):
        return self.key

    def get_model(self):
        return self.model()
        
    def get_transform(self):
        return self.transform

    def get_target(self):
        return self.target
        
    @classmethod
    def from_key(cls, key):
        for config in cls:
            if config.key == key:
                return config
        raise ValueError(f'No config with key {key}')