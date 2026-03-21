import torch
import torch.nn as nn
from model import transformer_encoder
from model import transformer_decoder
from model.layer import backbone_1d
from model.detr import DETR_Loss

class SetDETR_Model(nn.Module):

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

        self.n_queries = 30
        self.n_jpi_sets = self.header.n_jpi_sets

        self.n_entrances = params['n_entrances']
        self.n_exits = params['n_exits']
        self.n_angles = params['n_angles']
        self.n_pp_combos = self.n_entrances * self.n_exits
        self.n_channels = self.n_pp_combos * self.n_angles

        # number of spatial features per channel from backbone
        self.n_features = params.get('n_features', 8)

        # channel identities
        # (pp_in, pp_out, angle_idx) per channel
        channel_ids = []
        
        for c in range(self.n_channels):
            pp_combo = c // self.n_angles
            pp_in = pp_combo // self.n_exits
            pp_out = pp_combo % self.n_exits
            angle_idx = c % self.n_angles
            channel_ids.append([pp_in, pp_out, angle_idx])

        self.register_buffer('channel_ids', torch.tensor(channel_ids, dtype=torch.long))

        # 1D backbone (shared across all channels)
        self.backbone = backbone_1d.Backbone1D(
            d_transformer=self.d_transformer,
            n_features=self.n_features
        )

        # channel identity embeddings
        self.pp_in_embed = nn.Embedding(self.n_entrances, self.d_transformer)
        self.pp_out_embed = nn.Embedding(self.n_exits, self.d_transformer)
        self.angle_embed = nn.Embedding(self.n_angles, self.d_transformer)

        # energy positional encoding (per-channel spatial positions)
        self.energy_pos = nn.Parameter(torch.zeros(self.n_features, self.d_transformer))

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

        self.query_embedding = nn.Embedding(
            self.n_queries,
            self.d_transformer
        )

        self.class_head = nn.Sequential(
            nn.Linear(self.d_transformer, self.d_transformer),
            nn.ReLU(),
            nn.Linear(self.d_transformer, self.d_transformer),
            nn.ReLU(),
            nn.Linear(self.d_transformer, 2)
        )

        self.energy_head = nn.Sequential(
            nn.Linear(self.d_transformer, self.d_transformer),
            nn.ReLU(),
            nn.Linear(self.d_transformer, self.d_transformer),
            nn.ReLU(),
            nn.Linear(self.d_transformer, 1),
            nn.Sigmoid()
        )

        self.jpi_index_head = nn.Linear(
            self.d_transformer,
            self.n_jpi_sets
        )

        self.gamma_head = nn.Sequential(
            nn.Linear(self.d_transformer, self.d_transformer),
            nn.ReLU(),
            nn.Linear(self.d_transformer, self.d_transformer),
            nn.ReLU(),
            nn.Linear(self.d_transformer, self.max_gammas),
            nn.Sigmoid()
        )

    def forward(self, x):

        # [N, 2, E, C]
        N, _, E, C = x.shape

        data = x[:, 0] # [N, E, C]
        mask = x[:, 1] # [N, E, C]

        # get channels that have any unmasked energies
        channel_active = (mask.sum(dim=1) > 0) # [N, C]

        # ensure at least one channel is active per sample to avoid all-masked attention
        all_masked = ~channel_active.any(dim=1) # [N]
        if all_masked.any():
            channel_active[all_masked, 0] = True

        # reshape for per-channel 1D processing
        # stack data and mask as 2 input channels per channel
        data_per_ch = data.permute(0, 2, 1) # [N, C, E]
        mask_per_ch = mask.permute(0, 2, 1) # [N, C, E]

        backbone_input = torch.stack([data_per_ch, mask_per_ch], dim=2) # [N, C, 2, E]
        backbone_input = backbone_input.reshape(N * C, 2, E) # [NC, 2, E]

        # shared 1D backbone
        features = self.backbone(backbone_input) # [NC, n_features, d_transformer]
        n_features = features.shape[1]
        features = features.reshape(N, C, n_features, self.d_transformer) # [N, C, n_features, d_transformer]

        # channel identity embeddings
        ch_embed = (
            self.pp_in_embed(self.channel_ids[:C, 0]) +
            self.pp_out_embed(self.channel_ids[:C, 1]) +
            self.angle_embed(self.channel_ids[:C, 2])
        ) # [C, d_transformer]

        # positional encoding = channel identity + energy position
        # channel_embed: [1, C, 1, d_transformer] broadcast over N and n_features
        # energy_pos: [1, 1, n_features, d_transformer] broadcast over N and C
        pos_enc = ch_embed.unsqueeze(0).unsqueeze(2) + self.energy_pos.unsqueeze(0).unsqueeze(0)
        pos_enc = pos_enc.expand(N, -1, -1, -1) # [N, C, n_features, d_transformer]

        # flatten to sequence
        features = features.reshape(N, C * n_features, self.d_transformer) # [N, CL, d_transformer]
        pos_enc = pos_enc.reshape(N, C * n_features, self.d_transformer) # [N, CL, d_transformer]

        # key_padding_mask: True for tokens to ignore
        # expand channel_active [N, C] -> [N, C, n_features] -> [N, CL]
        token_active = channel_active.unsqueeze(-1).expand(-1, -1, n_features).reshape(N, C * n_features)
        
        # have to invert mask as transformer key_padding_mask uses booleans in the inverse way
        # maybe we should change mask to be consistent?
        key_padding_mask = ~token_active

        # transformer encoder
        enc = self.encoder(features, pos_enc, key_padding_mask=key_padding_mask)

        # transformer decoder
        query_pos = self.query_embedding.weight.unsqueeze(0).repeat(N, 1, 1)
        tgt = torch.zeros_like(query_pos)

        out, _ = self.decoder(tgt, enc, pos_enc, query_pos, memory_key_padding_mask=key_padding_mask)

        preds = {
            'class': self.class_head(out),
            'energy': self.energy_head(out),
            'gamma': self.gamma_head(out),
            'jpi_index': self.jpi_index_head(out)
        }

        return preds

    def get_optimiser(self, lr, weight_decay):
        optimiser = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        return optimiser

    def get_loss_fn(self):
        return DETR_Loss(self.header, self.params)