import csv
import json
import math
import pandas as pd
import torch
import torchvision.transforms
from torch import nn
from tqdm import tqdm
import data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import model.detr
from model.detr import HungarianMatcher

# path = 'out/results/detr_no_gamma_total'

# detr = pd.read_csv(f'{path}.csv')
# plt.plot(detr['epoch'], detr['val_recall'], label='recall')
# plt.plot(detr['epoch'], detr['val_precision'], label='precision')
# plt.savefig(f'{path}.png')

# tensors = torch.load('data/processed/5res_training.gz_tensors.pt')
# targets = torch.load('data/processed/5res_training.gz_targets.pt')

# n = 10

# tensors = tensors[:n]
# class_targets = targets['class'][:n]
# energy_targets = targets['energy'][:n]
# gamma_total_targets = targets['gamma_total'][:n]
# n_res_targets = targets['n_res'][:n]




def test_good_match():

    # 5 queries, 2 real resonances
    # good predictions
    # should get:
    # pred [0, 1]
    # target [0, 1]

    n_queries = 5

    preds = {
        "class": torch.tensor([[
            [0.1, 5.0], # resonance
            [0.2, 4.5], # resonance
            [5.0, 0.1],
            [4.0, 0.2],
            [3.0, 0.1],
        ]]),

        "energy": torch.tensor([[
            [0.30],
            [0.75],
            [0.10],
            [0.50],
            [0.90],
        ]]),

        "gamma_total": torch.tensor([[
            [1.0],
            [2.0],
            [0.1],
            [0.2],
            [0.3],
        ]])
    }

    targets = [{
        "class": torch.tensor([1, 1]),  # 2 resonances
        "energy": torch.tensor([[0.31], [0.74]]),
        "gamma_total": torch.tensor([[1.05], [2.05]])
    }]

    return preds, targets

matcher = HungarianMatcher(
    cost_class=1.0,
    cost_energy=2.0,
    cost_gamma_total=0.5
)

preds, targets = test_good_match()

indices = matcher(preds, targets)

print("Matching result:")
for batch_idx, (pred_idx, target_idx) in enumerate(indices):
    print("Preds", pred_idx)
    print("Targets:", target_idx)