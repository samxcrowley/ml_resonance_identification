from enum import Enum
import model.models as models
import transforms

class Config(Enum):

    RESNET34 = (models._resnet34_model(), transforms._resnet34_transform())
    ENCODER = (models._encoder_model(), transforms._encoder_transform())

    def __init__(self, model, transform):
        self.model = model
        self.transform = transform

    def get_model(self):
        return self.model
        
    def get_transform(self):
        return self.transform