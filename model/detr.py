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

        self.predict_gamma = params.get('predict_gamma', True)
        self.max_gammas = self.header.max_channels

        self.n_queries = 30
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

        # classification (MLP)
        self.class_head = nn.Sequential(
            nn.Linear(self.d_transformer, self.d_transformer),
            nn.ReLU(),
            nn.Linear(self.d_transformer, self.d_transformer),
            nn.ReLU(),
            nn.Linear(self.d_transformer, 2)
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

        # jpi classification
        self.jpi_index_head = nn.Linear(
            self.d_transformer,
            self.n_jpi_sets
        )

        # gamma regression (MLP)
        if self.predict_gamma:
            self.gamma_head = nn.Sequential(
                nn.Linear(self.d_transformer, self.d_transformer),
                nn.ReLU(),
                nn.Linear(self.d_transformer, self.d_transformer),
                nn.ReLU(),
                nn.Linear(self.d_transformer, self.max_gammas),
                nn.Sigmoid()
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
            'gamma': self.gamma_head(out) if self.predict_gamma else None,
            'jpi_index': self.jpi_index_head(out)
        }

        return preds

    def get_optimiser(self, lr, weight_decay):
        optimiser = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        return optimiser

    def get_loss_fn(self):
        return DETR_Loss(self.header, self.params)

class DETR_Loss(nn.Module):

    def __init__(self, header, params):

        super().__init__()

        self.header = header
        self.params = params

        self.predict_gamma = self.params.get('predict_gamma', True)
        self.cost_class = self.params['cost_class']
        self.cost_energy = self.params['cost_energy']
        self.cost_gamma = self.params['cost_gamma']
        self.cost_jpi_index = self.params['cost_jpi_index']
        self.class_weights = self.params['class_weights']

        self.matcher = HungarianMatcher()

    def prepare_targets(self, targets):

        class_targets = targets['class']
        energy_targets = targets['energy']
        jpi_index_targets = targets['jpi_index']

        _targets = []

        N = class_targets.shape[0]

        for n in range(N):

            mask = class_targets[n, :, 1] == 1

            t = {
                'class': torch.ones(mask.sum().item(), dtype=torch.long),
                'energy': energy_targets[n][mask],
                'jpi_index': jpi_index_targets[n][mask]
            }

            if self.predict_gamma:
                t['gamma'] = targets['gamma'][n][mask]
                t['gamma_mask'] = targets['gamma_mask'][n][mask]

            _targets.append(t)

        return _targets

    def forward(self, preds, targets):

        targets = self.prepare_targets(targets)

        indices = self.matcher(preds, targets)
        
        N, n_queries = preds['class'].shape[:2]

        loss_class = 0.0
        loss_energy = 0.0
        loss_gamma = 0.0
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

                loss_jpi_index += F.cross_entropy(
                    preds['jpi_index'][n][pred_idx],
                    targets[n]['jpi_index'][target_idx].squeeze(1).long().to(device, non_blocking=True)
                )

                if self.predict_gamma:

                    pred_gamma = preds['gamma'][n][pred_idx]
                    target_gamma = targets[n]['gamma'][target_idx].float().to(device, non_blocking=True)
                    gamma_mask = targets[n]['gamma_mask'][target_idx].float().to(device, non_blocking=True)

                    # zero out NaN gammas and clear their mask
                    nan_mask = target_gamma.isnan()
                    if nan_mask.any():
                        target_gamma = target_gamma.clone()
                        gamma_mask = gamma_mask.clone()
                        target_gamma[nan_mask] = 0.0
                        gamma_mask[nan_mask] = 0.0

                    if gamma_mask.sum() > 0:
                        loss_gamma += (F.mse_loss(pred_gamma, target_gamma, reduction='none') * gamma_mask).sum() / gamma_mask.sum()
            
        loss_class /= N
        loss_energy /= N
        loss_gamma /= N
        loss_jpi_index /= N

        total = self.cost_class * loss_class + self.cost_energy * loss_energy + self.cost_gamma * loss_gamma + self.cost_jpi_index * loss_jpi_index

        loss = {
            'total_loss': total,
            'class_loss': loss_class,
            'energy_loss': loss_energy,
            'jpi_index_loss': loss_jpi_index
        }

        if self.predict_gamma:
            loss['gamma_loss'] = loss_gamma

        return loss

class HungarianMatcher(nn.Module):

    def __init__(self):

        super().__init__()

    @torch.no_grad()
    def forward(self, preds, targets):

        indices = []

        for n in range(len(targets)):

            target_energy = targets[n]['energy']
            target_jpi_index = targets[n]['jpi_index']

            n_objects = target_energy.size(0)

            if n_objects == 0:
                indices.append((
                    torch.tensor([], dtype=torch.long),
                    torch.tensor([], dtype=torch.long)
                ))
                continue

            # energy cost
            pred_energy = preds['energy'][n]
            target_energy = target_energy.to(pred_energy.device)
            cost = torch.cdist(pred_energy, target_energy, p=1)

            # jpi cost
            pred_jpi_index = preds['jpi_index'][n]
            target_jpi_index = target_jpi_index.squeeze(1).long().to(pred_jpi_index.device)
            pred_jpi_probs = pred_jpi_index.softmax(-1) # [n_queries, n_jpi]
            cost_jpi_index = -pred_jpi_probs[:, target_jpi_index] # [n_queries, MAX_RESONANCES]
            cost += cost_jpi_index

            # Hungarian algorithm
            cost = cost.float().cpu().numpy()
            pred_idx, target_idx = linear_sum_assignment(cost)

            indices.append((
                torch.as_tensor(pred_idx, dtype=torch.long),
                torch.as_tensor(target_idx, dtype=torch.long)
            ))

        return indices