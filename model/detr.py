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
import data

class DETR_Model(nn.Module):

    def __init__(
        self,
        d_backbone=2048,
        d_transformer=128,
        n_hidden=512,
        n_head=4,
        n_layers=3,
        dropout_p=0.3,
        n_queries=data.MAX_RESONANCES,
        n_class_targets=2,
        max_len=5000
    ):

        super().__init__()

        self.backbone = backbone.Backbone(
            d_backbone=d_backbone,
            d_transformer=d_transformer
        )
        # freeze backbone (is pre-trained)
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

        self.energy_head = nn.Sequential(
            nn.Linear(d_transformer, d_transformer),
            nn.ReLU(),
            nn.Linear(d_transformer, d_transformer),
            nn.ReLU(),
            nn.Linear(d_transformer, 1),
            nn.Sigmoid() # energy in [0, 1]
        )

        self.gamma_total_head = nn.Sequential(
            nn.Linear(d_transformer, d_transformer),
            nn.ReLU(),
            nn.Linear(d_transformer, d_transformer),
            nn.ReLU(),
            nn.Linear(d_transformer, 1)
        )

    def forward(self, x):

        N = x.size(0)

        features = self.backbone(x)

        pos_enc = self.pos_enc(features)

        enc = self.encoder(features, pos_enc)

        query_pos = self.query_embedding.weight.unsqueeze(0).repeat(N, 1, 1)

        x = torch.zeros_like(query_pos)

        out, _ = self.decoder(x, enc, pos_enc, query_pos)

        preds = {
            'class': self.class_head(out),
            'energy': self.energy_head(out),
            'gamma_total': self.gamma_total_head(out)
        }

        return preds

class DETR_Loss(nn.Module):

    def __init__(self, cost_class=1.0, cost_energy=1.0, cost_gamma_total=0.5, include_gamma_total=True):

        super().__init__()

        self.cost_class = cost_class
        self.cost_energy = cost_energy
        self.cost_gamma_total = cost_gamma_total
        self.include_gamma_total = include_gamma_total

        self.matcher = HungarianMatcher(
            cost_class,
            cost_energy,
            cost_gamma_total,
            include_gamma_total=include_gamma_total
        )

    def prepare_targets(self, targets):

        class_targets = targets['class']
        energy_targets = targets['energy']
        gamma_total_targets = targets['gamma_total']

        _targets = []

        for n in range(class_targets.shape[0]):

            mask = class_targets[n, :, 1] == 1

            _targets.append({
                'class': torch.zeros(mask.sum().item(), dtype=torch.long),
                'energy': energy_targets[n][mask],
                'gamma_total': gamma_total_targets[n][mask]
            })

        return _targets

    def forward(self, preds, targets):

        targets = self.prepare_targets(targets)

        indices = self.matcher(preds, targets)

        N, n_queries = preds['class'].shape[:2]

        loss_class = 0.0
        loss_energy = 0.0
        loss_gamma_total = 0.0

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

            weight = torch.tensor([1.0, 0.3], device=device)
            loss_class += F.cross_entropy(preds['class'][n], target_classes, weight=weight)

            if len(pred_idx) > 0:

                loss_energy += F.l1_loss(
                    preds['energy'][n][pred_idx],
                    targets[n]['energy'][target_idx].float().to(device)
                )

                if self.include_gamma_total:
                    loss_gamma_total += F.l1_loss(
                        preds['gamma_total'][n][pred_idx],
                        targets[n]['gamma_total'][target_idx].float().to(device)
                    )
            
        loss_class /= N
        loss_energy /= N

        total = self.cost_class * loss_class + self.cost_energy * loss_energy

        if self.include_gamma_total:
            loss_gamma_total /= N
            total += self.cost_gamma_total * loss_gamma_total

        return total

class HungarianMatcher(nn.Module):

    def __init__(self, cost_class, cost_energy, cost_gamma_total, include_gamma_total):

        super().__init__()

        self.cost_class = cost_class
        self.cost_energy = cost_energy
        self.cost_gamma_total = cost_gamma_total
        self.include_gamma_total = include_gamma_total

    @torch.no_grad()
    def forward(self, preds, targets):

        indices = []

        for n in range(len(targets)):

            target_class = targets[n]['class']
            target_energy = targets[n]['energy']
            target_gamma_total = targets[n]['gamma_total']

            n_objects = target_class.size(0)

            if n_objects == 0:
                indices.append((
                    torch.tensor([], dtype=torch.long),
                    torch.tensor([], dtype=torch.long)
                ))
                continue

            cost = 0.0

            # class
            pred_class = preds['class'][n]
            target_class = target_class.to(pred_class.device)
            cost_class = -pred_class.softmax(-1)[:, 0]
            cost_class = cost_class.unsqueeze(1)
            cost_class = cost_class.expand(-1, n_objects)
            cost += self.cost_class * cost_class

            # energy
            pred_energy = preds['energy'][n]
            target_energy = target_energy.to(pred_energy.device)
            cost_energy = torch.cdist(pred_energy, target_energy, p=1)
            cost += self.cost_energy * cost_energy

            # gamma_total
            if self.include_gamma_total:
                pred_gamma_total = preds['gamma_total'][n]
                target_gamma_total = target_gamma_total.to(pred_gamma_total.device)
                cost_gamma_total = torch.cdist(pred_gamma_total, target_gamma_total, p=1)
                cost += self.cost_gamma_total * cost_gamma_total

            # Hungarian algorithm
            cost = cost.cpu().numpy()
            pred_idx, target_idx = linear_sum_assignment(cost)

            indices.append((
                torch.as_tensor(pred_idx, dtype=torch.long),
                torch.as_tensor(target_idx, dtype=torch.long)
            ))

        return indices