import csv
import os
import json
import math
import pandas as pd
import torch
import torchvision.transforms
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model.detr import HungarianMatcher

def print_result(name, indices):

    print(f"\n----- {name} -----")

    for b, (p, t) in enumerate(indices):
        print("Pred idx:", p)
        print("Target idx:", t)

def test_perfect_match():

    matcher = HungarianMatcher()

    preds = {
        "class": torch.tensor([[
            [0.1, 5.0],
            [0.2, 4.5],
            [5.0, 0.1],
            [4.0, 0.2],
            [3.0, 0.1],
        ]]),
        "energy": torch.tensor([[
            [0.3], [0.7], [0.1], [0.5], [0.9]
        ]]),
        "gamma_total": torch.tensor([[
            [1.0], [2.0], [0.1], [0.2], [0.3]
        ]])
    }

    targets = [{
        "class": torch.tensor([1, 1]),
        "energy": torch.tensor([[0.31], [0.69]]),
        "gamma_total": torch.tensor([[1.05], [2.05]])
    }]

    print_result("Perfect Match", matcher(preds, targets))

def test_no_targets():

    matcher = HungarianMatcher()

    preds = {
        "class": torch.randn(1, 5, 2),
        "energy": torch.rand(1, 5, 1),
        "gamma_total": torch.rand(1, 5, 1)
    }

    targets = [{
        "class": torch.empty(0, dtype=torch.long),
        "energy": torch.empty(0, 1),
        "gamma_total": torch.empty(0, 1)
    }]

    print_result("No Targets", matcher(preds, targets))

def test_energy_dominant():

    matcher = HungarianMatcher(cost_class=0.1, cost_energy=5.0, cost_gamma_total=0.1)

    preds = {
        "class": torch.tensor([[
            [5.0, 0.1],
            [5.0, 0.1],
            [5.0, 0.1],
            [5.0, 0.1],
            [5.0, 0.1],
        ]]),
        "energy": torch.tensor([[
            [0.2], [0.4], [0.6], [0.8], [0.9]
        ]]),
        "gamma_total": torch.rand(1, 5, 1)
    }

    targets = [{
        "class": torch.tensor([1, 1]),
        "energy": torch.tensor([[0.41], [0.79]]),
        "gamma_total": torch.rand(2, 1)
    }]

    print_result("Energy Dominant", matcher(preds, targets))

def test_class_dominant():

    matcher = HungarianMatcher(cost_class=5.0, cost_energy=0.1, cost_gamma_total=0.1)

    preds = {
        "class": torch.tensor([[
            [5.0, 0.1], # strong background
            [0.1, 5.0], # strong resonance
            [5.0, 0.1],
            [0.1, 5.0], # strong resonance
            [5.0, 0.1],
        ]]),
        "energy": torch.tensor([[
            [0.9], [0.1], [0.8], [0.2], [0.7]
        ]]),
        "gamma_total": torch.rand(1, 5, 1)
    }

    targets = [{
        "class": torch.tensor([1, 1]),
        "energy": torch.tensor([[0.85], [0.15]]),
        "gamma_total": torch.rand(2, 1)
    }]

    print_result("Class Dominant", matcher(preds, targets))

def test_gamma_dominant():
    
    matcher = HungarianMatcher(cost_class=0.1, cost_energy=0.1, cost_gamma_total=5.0)

    preds = {
        "class": torch.randn(1, 5, 2),
        "energy": torch.rand(1, 5, 1),
        "gamma_total": torch.tensor([[
            [1.0], [5.0], [10.0], [15.0], [20.0]
        ]])
    }

    targets = [{
        "class": torch.tensor([1, 1]),
        "energy": torch.rand(2, 1),
        "gamma_total": torch.tensor([[4.9], [19.8]])
    }]

    print_result("Gamma Dominant", matcher(preds, targets))

def test_crossing_energies():

    matcher = HungarianMatcher()

    preds = {
        "class": torch.tensor([[
            [0.1, 5.0],
            [0.1, 5.0],
            [5.0, 0.1],
            [5.0, 0.1],
            [5.0, 0.1],
        ]]),
        "energy": torch.tensor([[
            [0.2], [0.8], [0.1], [0.9], [0.5]
        ]]),
        "gamma_total": torch.zeros(1, 5, 1)
    }

    targets = [{
        "class": torch.tensor([1, 1]),
        "energy": torch.tensor([[0.75], [0.25]]),
        "gamma_total": torch.zeros(2, 1)
    }]

    print_result("Crossing Energies", matcher(preds, targets))

def test_completely_wrong():

    matcher = HungarianMatcher()

    preds = {
        "class": torch.tensor([[
            [5.0, 0.1],
            [5.0, 0.1],
            [5.0, 0.1],
            [5.0, 0.1],
            [5.0, 0.1],
        ]]),
        "energy": torch.rand(1, 5, 1),
        "gamma_total": torch.rand(1, 5, 1)
    }

    targets = [{
        "class": torch.tensor([1, 1]),
        "energy": torch.tensor([[0.3], [0.7]]),
        "gamma_total": torch.rand(2, 1)
    }]

    print_result("Completely Wrong Predictions", matcher(preds, targets))

def test_identical_costs():

    matcher = HungarianMatcher()

    preds = {
        "class": torch.zeros(1, 5, 2),
        "energy": torch.zeros(1, 5, 1),
        "gamma_total": torch.zeros(1, 5, 1)
    }

    targets = [{
        "class": torch.tensor([1, 1, 1]),
        "energy": torch.zeros(3, 1),
        "gamma_total": torch.zeros(3, 1)
    }]

    print_result("Identical Costs (Degenerate)", matcher(preds, targets))


if __name__ == "__main__":
    test_perfect_match()
    test_no_targets()
    test_energy_dominant()
    test_class_dominant()
    test_gamma_dominant()
    test_crossing_energies()
    test_completely_wrong()
    test_identical_costs()