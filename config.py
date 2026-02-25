from enum import Enum
import model.models as models
import model
import transforms

class Config(Enum):

    DETR = (
        'detr',
        model.detr.DETR,
        transforms._resnet34_transform()
        True
    )

    RESNET34_SINGLE_RES = (
        'resnet34_single_res',
        models._resnet34_single_res_model(),
        transforms._resnet34_transform(),
        False
    )

    RESNET34_SEGMENTATION = (
        'resnet34_segmentation',
        models._resnet34_segmentation_model(),
        transforms._resnet34_transform(),
        True
    )

    ENCODER = (
        'encoder',
        models._encoder_model(),
        transforms._encoder_transform(),
        True,
    )

    def __init__(self, key, model, transform, multi_resonance):
        self.key = key
        self.model = model
        self.transform = transform
        self.multi_resonance = multi_resonance

    def get_key(self):
        return self.key

    def get_model(self):
        return self.model
        
    def get_transform(self):
        return self.transform

    def is_multi_resonance(self):
        return self.multi_resonance
        
    @classmethod
    def from_key(cls, key):
        for config in cls:
            if config.key == key:
                return config
        raise ValueError(f'No config with key {key}')