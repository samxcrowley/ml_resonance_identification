import gzip
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import transforms

# set in params.json, default is 20
MAX_RESONANCES = 20

# x, y are the axes and z is the value at each point
x_key = 'theta_3_cm'
y_key = 'cn_ex'
z_key = 'dsdO'

# process loaded JSON list into (tensors [ls], targets [dict])
def process_json(data):

    tensors = get_tensors_from_json(data)
    targets = get_targets_from_json(data)

    return tensors, targets

# process a loaded JSON list into a list of tensors
def get_tensors_from_json(data):

    tensors = []

    for sample in data:

        points = sample['observable_sets'][0]['points']

        xs, ys = get_xs_ys(points)

        x_idx = {x: i for i, x in enumerate(xs)}
        y_idx = {y: i for i, y in enumerate(ys)}

        tensor = torch.zeros(len(ys), len(xs))

        for p in points:
            z = np.log10(max(p[z_key], 1e-30))
            x = x_idx[p[x_key]]
            y = y_idx[p[y_key]]
            tensor[y, x] = z

        tensors.append(tensor)

    return tensors

# process a loaded JSON list into stacked target tensors
def get_targets_from_json(data):

    class_targets = []
    energy_targets = []
    gamma_total_targets = []
    n_res_targets = []

    for sample in data:

        points = sample['observable_sets'][0]['points']

        xs, ys = get_xs_ys(points)

        n_resonances = len(sample['levels'])

        class_target = torch.zeros([MAX_RESONANCES, 2], dtype=torch.float32)
        energy_target = torch.zeros([MAX_RESONANCES, 1], dtype=torch.float32)
        gamma_total_target = torch.zeros([MAX_RESONANCES, 1], dtype=torch.float32)

        for n in range(n_resonances):

            level = sample['levels'][n]

            # normalise energy
            energy = level['energy']
            energy = transforms._normalise(energy, min(ys), max(ys))
            energy_target[n] = energy

            # sum gammas and take the log
            gamma_total = 0.0
            gammas = level['Gamma']
            for gamma in gammas:
                gamma_total += gamma
            gamma_total = np.log10(max(gamma_total, 1e-30))
            gamma_total_target[n] = gamma_total

        for n in range(MAX_RESONANCES):
            if n < n_resonances:
                class_target[n, 1] = 1.0
            else:
                class_target[n, 0] = 1.0

        class_targets.append(class_target)
        energy_targets.append(energy_target)
        gamma_total_targets.append(gamma_total_target)

        n_res_norm = transforms._normalise(n_resonances, 0, MAX_RESONANCES)
        n_res_targets.append(n_res_norm)

    return {
        'class': torch.stack(class_targets),
        'energy': torch.stack(energy_targets),
        'gamma_total': torch.stack(gamma_total_targets),
        'n_res': torch.tensor(n_res_targets, dtype=torch.float32),
    }

def open_data_file(path):

    if path.endswith('gz'):
        with gzip.open(path, 'rb') as f:
            json_bytes = f.read()
            json_str = json_bytes.decode()
            data = json.loads(json_str)
    elif path.endswith('json'):
        with open(path, 'r') as f:
            data = json.load(f)
    else:
        print('Invalid data file type.')
        return None

    return data

def get_xs_ys(points):

    xs = sorted(set(p[x_key] for p in points))
    ys = sorted(set(p[y_key] for p in points))

    return xs, ys

class ResonanceDataset(Dataset):

    # accepts preprocessed .pt files only
    def __init__(self, path, transform=None):
            
        saved = torch.load(path, weights_only=False)

        self.tensors = saved['tensors']
        self.targets = saved['targets']

        self.transform = transform

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):

        tensor = self.tensors[idx]
        if self.transform:
            tensor = self.transform(tensor)

        target = {key: self.targets[key][idx] for key in self.targets}

        return tensor, target