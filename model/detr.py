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
        d_backbone=512,
        d_transformer=512,
        n_hidden=2048,
        n_head=8,
        n_layers=6,
        dropout_p=0.0,
        n_queries=data.MAX_RESONANCES,
        max_len=1000,
        freeze_backbone=False):

        super().__init__()

        self.backbone = backbone.Backbone(
            d_backbone=d_backbone,
            d_transformer=d_transformer
        )

        if freeze_backbone:
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
            2
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

    def get_optimiser(self, lr, weight_decay, slow_backbone=False):

        if slow_backbone:
            backbone_lr = 1e-6
        else:
            backbone_lr = lr

        optimiser = torch.optim.AdamW([
            {'params': self.backbone.parameters(), 'lr': backbone_lr},
            {'params': [p for n, p in self.named_parameters() if 'backbone' not in n], 'lr': lr}
        ], weight_decay=weight_decay)

        return optimiser

    def evaluate(self, loader, device, energy_tolerance=0.01):

        self.eval()

        total_true = 0
        total_detected = 0
        total_false_positive = 0

        loss_fn = DETR_Loss()
        matcher = HungarianMatcher(
            cost_class=loss_fn.cost_class,
            cost_energy=loss_fn.cost_energy,
            cost_gamma_total=loss_fn.cost_gamma_total
        )

        with torch.no_grad():

            for tensor, targets in loader:

                tensor = tensor.to(device, non_blocking=True)
                preds = self(tensor)

                targets = loss_fn.prepare_targets(targets)

                indices = matcher(preds, targets)

                for n in range(len(targets)):

                    pred_idx, target_idx = indices[n]

                    n_objects = len(targets[n]['energy'])

                    total_true += n_objects

                    if len(pred_idx) == 0:
                        continue

                    pred_energy = preds['energy'][n][pred_idx].squeeze()

                    target_energy = targets[n]['energy'][target_idx].squeeze()
                    target_energy = target_energy.to(device, non_blocking=True)

                    difference = pred_energy - target_energy

                    # true positives: matched pairs that are energetically close
                    close = (difference).abs() < energy_tolerance
                    tp = close.sum().item()
                    total_detected += tp

                    # false positives: queries predicting resonance that aren't true positives
                    n_confident = (preds['class'][n].softmax(-1)[:, 1] > 0.5).sum().item()
                    total_false_positive += max(0, n_confident - tp)

        recall = total_detected / total_true if total_true > 0 else 0
        precision = total_detected / (total_detected + total_false_positive) if (total_detected + total_false_positive) > 0 else 0

        return {
            'recall': recall,
            'precision': precision
        }

class DETR_Loss(nn.Module):

    def __init__(self, cost_class=1.0, cost_energy=2.0, cost_gamma_total=0.5, class_weights=[0.1, 1.0]):

        super().__init__()

        self.cost_class = cost_class
        self.cost_energy = cost_energy
        self.cost_gamma_total = cost_gamma_total
        self.class_weights = class_weights

        self.matcher = HungarianMatcher(
            cost_class,
            cost_energy,
            cost_gamma_total
        )

    def prepare_targets(self, targets):

        class_targets = targets['class']
        energy_targets = targets['energy']
        gamma_total_targets = targets['gamma_total']

        _targets = []

        N = class_targets.shape[0]

        for n in range(N):

            mask = class_targets[n, :, 1] == 1

            _targets.append({
                'class': torch.ones(mask.sum().item(), dtype=torch.long),
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

            pred_classes = preds['class'][n]

            target_classes = torch.full((n_queries,), 0, dtype=torch.long, device=device)
            if len(pred_idx) > 0:
                target_classes[pred_idx] = 1

            weight = torch.tensor(self.class_weights, device=device)

            loss_class += F.cross_entropy(pred_classes, target_classes, weight=weight)

            if len(pred_idx) > 0:

                loss_energy += F.l1_loss(
                    preds['energy'][n][pred_idx],
                    targets[n]['energy'][target_idx].float().to(device, non_blocking=True)
                )

                loss_gamma_total += F.l1_loss(
                    preds['gamma_total'][n][pred_idx],
                    targets[n]['gamma_total'][target_idx].float().to(device, non_blocking=True)
                )
            
        loss_class /= N
        loss_energy /= N

        total = self.cost_class * loss_class + self.cost_energy * loss_energy

        loss_gamma_total /= N
        total += self.cost_gamma_total * loss_gamma_total

        loss = {
            'total_loss': total, # can't call .item() because we run .backward() on this in train
            'class_loss': loss_class.item(),
            'energy_loss': loss_energy.item(),
            'gamma_total_loss': loss_gamma_total.item()
        }

        return loss

class HungarianMatcher(nn.Module):

    def __init__(self, cost_class, cost_energy, cost_gamma_total):

        super().__init__()

        self.cost_class = cost_class
        self.cost_energy = cost_energy
        self.cost_gamma_total = cost_gamma_total

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

            pred_prob = pred_class.softmax(-1)[:, 1].clamp(min=1e-6, max=1 - 1e-6)

            alpha, gamma = 0.25, 2.0
            focal_cost = -(alpha * (1 - pred_prob) ** gamma * pred_prob.log())
            cost_class = focal_cost.unsqueeze(1).expand(-1, n_objects)

            cost += self.cost_class * cost_class

            # energy
            pred_energy = preds['energy'][n]
            target_energy = target_energy.to(pred_energy.device)

            cost_energy = torch.cdist(pred_energy, target_energy, p=1)

            cost += self.cost_energy * cost_energy

            # gamma_total
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