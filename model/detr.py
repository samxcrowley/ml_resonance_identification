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

        self.max_gammas = self.header.max_channels

        self.n_queries = params['n_queries']
        self.max_j = params['max_j']
        self.n_j_classes = int(self.max_j * 2) + 1

        self.pos_enc_max_len = 1000

        self.backbone = backbone.Backbone(
            d_transformer=self.d_transformer,
            norm=params['norm']
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

        # J classification
        self.j_head = nn.Linear(self.d_transformer, self.n_j_classes)

        # parity classification (0: -, 1: +)
        self.pi_head = nn.Linear(self.d_transformer, 2)

        # gamma regression (MLP)
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

        features, key_padding_mask = self.backbone(x)

        pos_enc = self.pos_enc(features)

        enc = self.encoder(features, pos_enc, key_padding_mask=key_padding_mask)

        query_pos = self.query_embedding.weight.unsqueeze(0).expand(N, -1, -1)
        x = features.new_zeros((N, self.n_queries, self.d_transformer))

        out, _ = self.decoder(x, enc, pos_enc, query_pos, memory_key_padding_mask=key_padding_mask)

        preds = {
            'class': self.class_head(out),
            'energy': self.energy_head(out),
            'gamma': self.gamma_head(out),
            'j': self.j_head(out),
            'pi': self.pi_head(out),
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

        self.cost_class = self.params['cost_class']
        self.cost_energy = self.params['cost_energy']
        self.cost_gamma = self.params['cost_gamma']
        self.cost_j = self.params.get('cost_j', 0.5)
        self.cost_pi = self.params.get('cost_pi', 0.5)
        self.class_weights = self.params['class_weights']

        # jpi_set_index to j_index / pi_index lookups
        self.register_buffer('jpi_to_j',  header.get_jpi_to_j_index())
        self.register_buffer('jpi_to_pi', header.get_jpi_to_pi_index())

        self.matcher = HungarianMatcher()

    def _focal_loss(self, pred_logits, targets, alpha=0.25, gamma=2.0):
        pred_logits_f = pred_logits.float()
        ce = F.cross_entropy(pred_logits_f, targets, reduction='none')
        probs = pred_logits_f.softmax(-1)
        p_t = probs[range(len(targets)), targets]
        focal_weight = (1 - p_t) ** gamma
        alpha_t = torch.where(targets == 1, alpha, 1 - alpha)
        return (alpha_t * focal_weight * ce).mean()

    def prepare_targets(self, targets):

        class_targets = targets['class']
        energy_targets = targets['energy']
        jpi_index_targets = targets['jpi_index']

        _targets = []

        N = class_targets.shape[0]

        for n in range(N):

            mask = class_targets[n, :, 1] == 1

            jpi_idx = jpi_index_targets[n][mask].squeeze(-1).long()

            t = {
                'class': torch.ones(mask.sum().item(), dtype=torch.long),
                'energy': energy_targets[n][mask],
                'j_index': self.jpi_to_j[jpi_idx],
                'pi': self.jpi_to_pi[jpi_idx],
                'gamma': targets['gamma'][n][mask],
                'gamma_mask': targets['gamma_mask'][n][mask],
            }

            if 'info_weight' in targets:
                t['info_weight'] = targets['info_weight'][n][mask]

            _targets.append(t)

        return _targets

    def forward(self, preds, targets):

        targets = self.prepare_targets(targets)

        indices = self.matcher(preds, targets)
        
        N, n_queries = preds['class'].shape[:2]

        loss_class = 0.0
        loss_energy = 0.0
        loss_gamma = 0.0
        loss_j = 0.0
        loss_pi = 0.0

        for n in range(N):

            pred_idx, target_idx = indices[n]
            device = preds['class'].device

            pred_classes = preds['class'][n]

            target_classes = torch.full((n_queries,), 0, dtype=torch.long, device=device)
            if len(pred_idx) > 0:
                target_classes[pred_idx] = 1

            loss_class += self._focal_loss(pred_classes, target_classes)

            if len(pred_idx) > 0:

                # per-resonance information weights
                has_weights = 'info_weight' in targets[n]
                if has_weights:
                    weights = targets[n]['info_weight'][target_idx].float().to(device)
                    w_sum = weights.sum()
                else:
                    w_sum = None

                if has_weights and w_sum > 0:
                    per_e = F.l1_loss(
                        preds['energy'][n][pred_idx],
                        targets[n]['energy'][target_idx].float().to(device),
                        reduction='none'
                    )
                    loss_energy += (per_e.squeeze(-1) * weights).sum() / w_sum
                elif not has_weights:
                    loss_energy += F.l1_loss(
                        preds['energy'][n][pred_idx],
                        targets[n]['energy'][target_idx].float().to(device)
                    )

                target_j = targets[n]['j_index'][target_idx].to(device)
                target_pi = targets[n]['pi'][target_idx].to(device)

                if has_weights and w_sum > 0:
                    per_j = F.cross_entropy(preds['j'][n][pred_idx], target_j,  reduction='none')
                    per_pi = F.cross_entropy(preds['pi'][n][pred_idx], target_pi, reduction='none')
                    loss_j += (per_j  * weights).sum() / w_sum
                    loss_pi += (per_pi * weights).sum() / w_sum
                elif not has_weights:
                    loss_j += F.cross_entropy(preds['j'][n][pred_idx], target_j)
                    loss_pi += F.cross_entropy(preds['pi'][n][pred_idx], target_pi)

                pred_gamma = preds['gamma'][n][pred_idx]
                target_gamma = targets[n]['gamma'][target_idx].float().to(device)
                gamma_mask = targets[n]['gamma_mask'][target_idx].float().to(device)

                # zero out NaN gammas and clear their mask
                nan_mask = target_gamma.isnan()
                if nan_mask.any():
                    target_gamma = target_gamma.clone()
                    gamma_mask = gamma_mask.clone()
                    target_gamma[nan_mask] = 0.0
                    gamma_mask[nan_mask] = 0.0

                if gamma_mask.sum() > 0:
                    if has_weights and w_sum > 0:
                        per_res_gamma = (F.mse_loss(pred_gamma, target_gamma, reduction='none') * gamma_mask).sum(dim=1)
                        gamma_count = gamma_mask.sum(dim=1).clamp(min=1)
                        per_res_gamma = per_res_gamma / gamma_count
                        loss_gamma += (per_res_gamma * weights).sum() / w_sum
                    else:
                        loss_gamma += (F.mse_loss(pred_gamma, target_gamma, reduction='none') * gamma_mask).sum() / gamma_mask.sum()
            
        loss_class /= N
        loss_energy /= N
        loss_gamma /= N
        loss_j /= N
        loss_pi /= N

        total = (self.cost_class * loss_class +
                 self.cost_energy * loss_energy +
                 self.cost_gamma * loss_gamma +
                 self.cost_j * loss_j +
                 self.cost_pi * loss_pi)

        return {
            'total_loss': total,
            'class_loss': loss_class,
            'energy_loss': loss_energy,
            'gamma_loss': loss_gamma,
            'j_loss': loss_j,
            'pi_loss': loss_pi,
        }

class HungarianMatcher(nn.Module):

    def __init__(self):

        super().__init__()

    @torch.no_grad()
    def forward(self, preds, targets):

        indices = []

        for n in range(len(targets)):

            target_energy = targets[n]['energy']

            n_objects = target_energy.size(0)

            if n_objects == 0:
                indices.append((
                    torch.tensor([], dtype=torch.long),
                    torch.tensor([], dtype=torch.long)
                ))
                continue

            # class cost
            pred_class_probs = preds['class'][n].softmax(-1) # [n_queries, 2]
            cost_class = -pred_class_probs[:, 1].unsqueeze(1).expand(-1, n_objects)

            # energy cost
            pred_energy = preds['energy'][n]
            target_energy = target_energy.to(pred_energy.device)
            cost_energy = torch.cdist(pred_energy, target_energy, p=1)

            # J cost
            pred_j_probs = preds['j'][n].softmax(-1) # [n_queries, n_j_classes]
            target_j = targets[n]['j_index'].to(pred_j_probs.device)
            cost_j = -pred_j_probs[:, target_j] # [n_queries, n_objects]

            # pi cost
            pred_pi_probs = preds['pi'][n].softmax(-1) # [n_queries, 2]
            target_pi = targets[n]['pi'].to(pred_pi_probs.device)
            cost_pi = -pred_pi_probs[:, target_pi] # [n_queries, n_objects]

            cost = cost_class + cost_energy + cost_j + cost_pi

            # Hungarian algorithm
            cost = cost.float().cpu().numpy()
            pred_idx, target_idx = linear_sum_assignment(cost)

            indices.append((
                torch.as_tensor(pred_idx, dtype=torch.long),
                torch.as_tensor(target_idx, dtype=torch.long)
            ))

        return indices
