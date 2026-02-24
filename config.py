from enum import Enum
import model.models as models
import transforms

class Config(Enum):

    RESNET34_SINGLE_RES = (models._resnet34_single_res_model(), transforms._resnet34_transform())
    RESNET34_SEGMENTATION = (models._resnet34_segmentation_model(), transforms._resnet34_transform(), True)
    ENCODER = (models._encoder_model(), transforms._encoder_transform())

    def __init__(self, model, transform, multi_resonance=False):
        self.model = model
        self.transform = transform
        self.multi_resonance = multi_resonance

    def get_model(self):
        return self.model
        
    def get_transform(self):
        return self.transform

    def is_multi_resonance(self):
        return self.multi_resonance