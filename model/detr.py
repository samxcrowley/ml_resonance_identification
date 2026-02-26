import torch
import torch.nn as nn
import torchvision.models
from torchvision.models._utils import IntermediateLayerGetter
import model
from model import transformer_decoder as td
import model.layer

class DETR_Model(nn.Module):

    def __init__(
        self,
        d_backbone=2048,
        d_transformer=256,
        n_hidden=2048,
        n_head=8,
        n_layers=6,
        dropout_p=0.2,
        n_queries=20,
        n_class_targets=2,
        n_reg_targets=2,
        max_len=5000
    ):

        self.backbone = model.layer.backbone.Backbone(
            d_backbone=d_backbone,
            d_transformer=d_transformer
        )

        self.encoder = model.transformer_encoder(
            d_model=d_transformer,
            n_hidden=n_hidden,
            n_head=n_head,
            n_layers=n_layers,
            dropout_p=dropout_p
        )

        self.decoder = model.transformer_decoder(
            d_model=d_transformer,
            n_hidden=n_hidden,
            n_head=n_head,
            n_layers=n_layers,
            dropout_p=dropout_p
        )

        self.pos_enc = model.layer.positional_encoding.PositionalEncoding(
            max_len=max_len,
            d_transformer=d_transformer
        )

        self.query_embedding = nn.Embedding(
            n_queries,
            d_transformer
        )

        self.class_head = nn.Linear(
            d_transformer,
            n_class_targets
        )

        self.reg_head = nn.Sequential(
            nn.Linear(d_transformer, d_transformer),
            nn.ReLU(),
            nn.Linear(d_transformer, d_transformer),
            nn.ReLU(),
            nn.Linear(d_transformer, n_reg_targets),
        )

    def forward(self, x):

        N = x.size(0)

        features = self.backbone(x)

        pos_enc = self.pos_enc(features)

        enc = self.encoder(features, pos_enc)

        query_pos = self.query_embedding.weight.unsqueeze(0).repeat(N, 1, 1)

        x = torch.zeros_like(query_pos)

        out, _ = self.decoder(x, enc, pos_enc, query_pos)

        pred_class = self.class_head(out)

        pred_reg = self.reg_head(out).sigmoid()

        pred = {
            'class': pred_class,
            'reg': pred_reg
        }