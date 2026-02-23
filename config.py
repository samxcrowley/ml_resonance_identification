from enum import Enum
import model.models as models
import transforms

class Config(Enum):

    RESNET34 = 0
    ENCODER = 1

    def get_model(self):
        if self == self.RESNET34:
            return models._resnet34_model()
        elif self == self.ENCODER:
            return models._encoder_model()
        
    def get_transform(self):
        if self == self.RESNET34:
            return transforms._resnet34_transform()
        elif self == self.ENCODER:
            return transfomrs._encoder_transform()