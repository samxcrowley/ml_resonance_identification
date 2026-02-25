import torch
import torch.nn as nn

class DETR(nn.Module):

    def __init__(self, d_cnn, d_transformer):

        super().__init__()

        self.d_cnn = d_cnn
        self.d_transformer = d_transformer

    def forward(self, x):

        return None