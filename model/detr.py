import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torchvision.models._utils import IntermediateLayerGetter
import model
from model import transformer_encoder
from model import transformer_decoder
from model.layer import backbone
from model.layer import positional_encoding
from scipy.optimize import linear_sum_assignment

class DETR_Model(nn.Module):

    def __init__(
        self,
        d_backbone=2048,
        d_transformer=128,
        n_hidden=512,
        n_head=4,
        n_layers=3,
        dropout_p=0.3,
        n_queries=100,
        n_class_targets=2,
        n_reg_targets=2,
        max_len=5000
    ):

        super().__init__()

        self.backbone = backbone.Backbone(
            d_backbone=d_backbone,
            d_transformer=d_transformer
        )
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.encoder = transformer_encoder.Transformer_Encoder_Model(
            d_model=d_transformer,
            n_hidden=n_hidden,
            n_head=n_head,
            n_layers=n_layers,
            dropout_p=dropout_p
        )

        self.decoder = transformer_decoder.Transformer_Decoder_Model(
            d_model=d_transformer,
            n_hidden=n_hidden,
            n_head=n_head,
            n_layers=n_layers,
            dropout_p=dropout_p
        )

        self.pos_enc = positional_encoding.PositionalEncoding(
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

        preds = {
            'class': pred_class,
            'reg': pred_reg
        }

        return preds

class DETR_Loss(nn.Module):

    def __init__(self, cost_class=1.0, cost_reg=1.0):

        super().__init__()

        self.cost_class = cost_class
        self.cost_reg = cost_reg

        self.matcher = HungarianMatcher(cost_class, cost_reg)

    def prepare_targets(self, class_targets, reg_targets):

        targets = []

        for n in range(class_targets.shape[0]):

            mask = class_targets[n, :, 1] == 1

            n_objects = mask.sum().item()
            _class = torch.zeros(n_objects, dtype=torch.long)
            reg = reg_targets[n][mask]

            targets.append((_class, reg))

        return targets

    def forward(self, preds, targets):

        targets = self.prepare_targets(targets[0], targets[1])

        indices = self.matcher(preds, targets)

        N, n_queries = preds['class'].shape[:2]

        loss_class = 0
        loss_reg = 0

        for n in range(N):

            pred_idx, target_idx = indices[n]
            device = preds['class'].device

            target_classes = torch.full(
                (n_queries,),
                1,
                dtype=torch.long,
                device=device
            )

            if len(pred_idx) > 0:
                target_classes[pred_idx] = 0

            weight = torch.tensor([1.0, 0.3], device=preds['class'].device)
            loss_class += F.cross_entropy(preds['class'][n], target_classes, weight=weight)

            if len(pred_idx) > 0:
                matched_pred_reg = preds['reg'][n][pred_idx]
                matched_target_reg = targets[n][1][target_idx].float().to(device)
                loss_reg += F.l1_loss(matched_pred_reg, matched_target_reg)
            

        loss_class /= n
        loss_reg /= n

        # return self.cost_class * loss_class + self.cost_reg * loss_reg

        return {
            'total': self.cost_class * loss_class + self.cost_reg * loss_reg,
            'class': loss_class.item(),
            'reg': loss_reg.item()
        }

class HungarianMatcher(nn.Module):

    def __init__(self, cost_class=1.0, cost_reg=1.0):

        super().__init__()

        self.cost_class = cost_class
        self.cost_reg = cost_reg

    @torch.no_grad()
    def forward(self, preds, targets):

        indices = []

        for n in range(len(targets)):

            target_class, target_reg = targets[n]
            n_objects = target_class.size(0)

            if n_objects == 0:
                indicies.append((
                    torch.tensor([], dtype=torch.long),
                    torch.tensor([], dtype=torch.long)
                ))
                continue

            pred_class = preds['class'][n]
            pred_reg = preds['reg'][n]

            target_class = target_class.to(pred_class.device)
            target_reg = target_reg.to(pred_reg.device)

            # cross-entropy between each query and target
            # [n_queries, n_objects]
            cost_class = -pred_class.softmax(-1)[:, 0]
            cost_class = cost_class.unsqueeze(1)
            cost_class = cost_class.expand(-1, n_objects)

            # L1 loss between each query and target
            # [n_queries, n_objects]
            cost_reg = torch.cdist(pred_reg, target_reg, p=1)

            cost = self.cost_class * cost_class * self.cost_reg * cost_reg
            cost = cost.cpu().numpy()

            # Hungarian algorithm
            pred_idx, target_idx = linear_sum_assignment(cost)

            indices.append((
                torch.as_tensor(pred_idx, dtype=torch.long),
                torch.as_tensor(target_idx, dtype=torch.long)
            ))

        return indices