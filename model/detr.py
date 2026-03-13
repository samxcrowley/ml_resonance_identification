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
import process.data as data
import json

class DETR_Model(nn.Module):

    def __init__(self, header, params):

        super().__init__()

        self.header = header
        self.params = params

        self.d_transformer = params['d_transformer']
        self.n_hidden = params['n_hidden']
        self.n_head = params['n_head']
        self.n_layers = params['n_layers']
        self.dropout_p = params['dropout_p']
        self.eval_tolerance = params['eval_tolerance']
        self.confidence_threshold = params['confidence_threshold']

        self.max_gammas = self.header.max_channels

        self.n_queries = data.MAX_RESONANCES*2
        self.n_jpi_sets = self.header.n_jpi_sets

        self.pos_enc_max_len = 1000

        self.backbone = backbone.Backbone(
            d_transformer=self.d_transformer
        )

        self.encoder = transformer_encoder.Transformer_Encoder_Model(
            d_model=self.d_transformer,
            n_hidden=self.n_hidden,
            n_head=self.n_head,
            n_layers=self.n_layers,
            dropout_p=self.dropout_p
        )

        self.decoder = transformer_decoder.Transformer_Decoder_Model(
            d_model=self.d_transformer,
            n_hidden=self.n_hidden,
            n_head=self.n_head,
            n_layers=self.n_layers,
            dropout_p=self.dropout_p
        )

        self.pos_enc = positional_encoding.PositionalEncoding(
            max_len=self.pos_enc_max_len,
            d_transformer=self.d_transformer
        )

        self.query_embedding = nn.Embedding(
            self.n_queries,
            self.d_transformer
        )

        # classification
        self.class_head = nn.Linear(
            self.d_transformer,
            2
        )

        # regression (MLP)
        self.energy_head = nn.Sequential(
            nn.Linear(self.d_transformer, self.d_transformer),
            nn.ReLU(),
            nn.Linear(self.d_transformer, self.d_transformer),
            nn.ReLU(),
            nn.Linear(self.d_transformer, 1),
            nn.Sigmoid() # energy in [0, 1]
        )

        # regression (MLP)
        self.gamma_head = nn.Sequential(
            nn.Linear(self.d_transformer, self.d_transformer),
            nn.ReLU(),
            nn.Linear(self.d_transformer, self.d_transformer),
            nn.ReLU(),
            nn.Linear(self.d_transformer, self.max_gammas)
        )

        # regression (MLP)
        self.gamma_total_head = nn.Sequential(
            nn.Linear(self.d_transformer, self.d_transformer),
            nn.ReLU(),
            nn.Linear(self.d_transformer, self.d_transformer),
            nn.ReLU(),
            nn.Linear(self.d_transformer, 1)
        )

        # classification
        self.jpi_index_head = nn.Linear(
            self.d_transformer,
            self.n_jpi_sets
        )

    def get_loss_fn(self):
        return DETR_Loss(self.header, self.params)

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
            'gamma': self.gamma_head(out),
            'gamma_total': self.gamma_total_head(out),
            'jpi_index': self.jpi_index_head(out)
        }

        return preds

    def get_optimiser(self, lr, weight_decay):

        optimiser = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        return optimiser

    def evaluate(self, loader, device):

        self.eval()

        total_true = 0
        total_detected = 0
        total_false_positive = 0

        loss_fn = DETR_Loss(self.header, self.params)

        matcher = HungarianMatcher(
            cost_class=loss_fn.cost_class,
            cost_energy=loss_fn.cost_energy,
            cost_gamma=loss_fn.cost_gamma,
            cost_gamma_total=loss_fn.cost_gamma_total,
            cost_jpi_index=loss_fn.cost_jpi_index
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

                    # true positives: matched pairs that are confident and close in energy
                    close = (difference).abs() < self.eval_tolerance
                    confident_matched = preds['class'][n].softmax(-1)[pred_idx, 1] > self.confidence_threshold
                    tp = (close & confident_matched).sum().item()
                    
                    total_detected += tp

                    # false positives: confident queries that aren't true positives
                    n_confident = (preds['class'][n].softmax(-1)[:, 1] > self.confidence_threshold).sum().item()
                    total_false_positive += n_confident - tp

        recall = total_detected / total_true if total_true > 0 else 0
        precision = total_detected / (total_detected + total_false_positive) if (total_detected + total_false_positive) > 0 else 0

        return {
            'recall': recall,
            'precision': precision
        }

class DETR_Loss(nn.Module):

    def __init__(self, header, params):

        super().__init__()

        self.header = header
        self.params = params

        self.cost_class = self.params['cost_class']
        self.cost_energy = self.params['cost_energy']
        self.cost_gamma = self.params['cost_gamma']
        self.cost_gamma_total = self.params['cost_gamma_total']
        self.cost_jpi_index = self.params['cost_jpi_index']
        self.class_weights = self.params['class_weights']

        self.matcher = HungarianMatcher(
            self.cost_class,
            self.cost_energy,
            self.cost_gamma,
            self.cost_gamma_total,
            self.cost_jpi_index
        )

    def prepare_targets(self, targets):

        class_targets = targets['class']
        energy_targets = targets['energy']
        gamma_targets = targets['gamma']
        gamma_mask_targets = targets['gamma_mask']
        gamma_total_targets = targets['gamma_total']
        jpi_index_targets = targets['jpi_index']

        _targets = []

        N = class_targets.shape[0]

        for n in range(N):

            mask = class_targets[n, :, 1] == 1

            _targets.append({
                'class': torch.ones(mask.sum().item(), dtype=torch.long),
                'energy': energy_targets[n][mask],
                'gamma': gamma_targets[n][mask],
                'gamma_mask': gamma_mask_targets[n][mask],
                'gamma_total': gamma_total_targets[n][mask],
                'jpi_index': jpi_index_targets[n][mask]
            })

        return _targets

    def forward(self, preds, targets):

        targets = self.prepare_targets(targets)

        indices = self.matcher(preds, targets)
        
        N, n_queries = preds['class'].shape[:2]

        loss_class = 0.0
        loss_energy = 0.0
        loss_gamma = 0.0
        loss_gamma_total = 0.0
        loss_jpi_index = 0.0

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

                pred_gamma = preds['gamma'][n][pred_idx]
                target_gamma = targets[n]['gamma'][target_idx].float().to(device, non_blocking=True)
                gamma_mask = targets[n]['gamma_mask'][target_idx].float().to(device, non_blocking=True)

                if gamma_mask.sum() > 0:
                    loss_gamma += (F.l1_loss(pred_gamma, target_gamma, reduction='none') * gamma_mask).sum() / gamma_mask.sum()


                loss_gamma_total += F.l1_loss(
                    preds['gamma_total'][n][pred_idx],
                    targets[n]['gamma_total'][target_idx].float().to(device, non_blocking=True)
                )

                loss_jpi_index += F.cross_entropy(
                    preds['jpi_index'][n][pred_idx],
                    targets[n]['jpi_index'][target_idx].squeeze(1).long().to(device, non_blocking=True)
                )
            
        loss_class /= N
        loss_energy /= N
        loss_gamma /= N
        loss_gamma_total /= N
        loss_jpi_index /= N

        total = self.cost_class * loss_class + self.cost_energy * loss_energy + self.cost_gamma * loss_gamma + self.cost_gamma_total * loss_gamma_total + self.cost_jpi_index * loss_jpi_index

        loss = {
            'total_loss': total,
            'class_loss': loss_class,
            'energy_loss': loss_energy,
            'gamma_loss': loss_gamma,
            'gamma_total_loss': loss_gamma_total,
            'jpi_index_loss': loss_jpi_index
        }

        return loss

class HungarianMatcher(nn.Module):

    def __init__(self, cost_class, cost_energy, cost_gamma, cost_gamma_total, cost_jpi_index):

        super().__init__()

        self.cost_class = cost_class
        self.cost_energy = cost_energy
        self.cost_gamma = cost_gamma
        self.cost_gamma_total = cost_gamma_total
        self.cost_jpi_index = cost_jpi_index

    @torch.no_grad()
    def forward(self, preds, targets):

        indices = []

        for n in range(len(targets)):

            target_class = targets[n]['class']
            target_energy = targets[n]['energy']
            target_gamma = targets[n]['gamma']
            target_gamma_mask = targets[n]['gamma_mask']
            target_gamma_total = targets[n]['gamma_total']
            target_jpi_index = targets[n]['jpi_index']

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

            # gamma (masked: only compare valid channels)
            pred_gamma = preds['gamma'][n]
            target_gamma = target_gamma.to(pred_gamma.device)
            target_gamma_mask = target_gamma_mask.to(pred_gamma.device)

            # [n_queries, n_objects, max_channels]
            diff_gamma = (pred_gamma.unsqueeze(1) - target_gamma.unsqueeze(0)).abs()
            cost_gamma = (diff_gamma * target_gamma_mask.unsqueeze(0)).sum(-1)
            # normalise by number of valid channels per object
            n_valid = target_gamma_mask.sum(-1).clamp(min=1).unsqueeze(0)
            cost_gamma = cost_gamma / n_valid

            cost += self.cost_gamma * cost_gamma

            # gamma_total
            pred_gamma_total = preds['gamma_total'][n]
            target_gamma_total = target_gamma_total.to(pred_gamma_total.device)

            cost_gamma_total = torch.cdist(pred_gamma_total, target_gamma_total, p=1)

            cost += self.cost_gamma_total * cost_gamma_total

            # jpi index
            pred_jpi_index = preds['jpi_index'][n]
            target_jpi_index = target_jpi_index.squeeze(1).long().to(pred_jpi_index.device)

            pred_jpi_probs = pred_jpi_index.softmax(-1)  # [n_queries, n_jpi]
            cost_jpi_index = -pred_jpi_probs[:, target_jpi_index]  # [n_queries, n_objects]

            cost += self.cost_jpi_index * cost_jpi_index

            # Hungarian algorithm
            cost = cost.float().cpu().numpy()
            pred_idx, target_idx = linear_sum_assignment(cost)

            indices.append((
                torch.as_tensor(pred_idx, dtype=torch.long),
                torch.as_tensor(target_idx, dtype=torch.long)
            ))

        return indices