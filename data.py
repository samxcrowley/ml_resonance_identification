import gzip
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import transforms

# x, y are the axes and z is the value at each point
x_key = 'theta_cm_out'
y_key = 'cn_ex'
z_key = 'dsdO'

# output sequence shape:
# [n, n_A, n_E]
# output target shape:
# [n]
def get_tensors(path, log_cx=True, compressed=True):

    data = open_file(path, compressed)

    n = len(data)

    tensors = []
    
    for i in range(n):
        
        points = data[i]['observable_sets'][0]['points']

        xs = sorted(set(p[x_key] for p in points))
        ys = sorted(set(p[y_key] for p in points))

        x_idx = {x: i for i, x in enumerate(xs)}
        y_idx = {y: i for i, y in enumerate(ys)}
        
        tensor = torch.zeros(len(xs), len(ys))

        for p in points:

            if log_cx:
                z = np.log10(p[z_key])
            else:
                z = p[z_key]

            x = x_idx[p[x_key]]
            y = y_idx[p[y_key]]
            tensor[x, y] = z

        tensors.append(tensor)

    return tensors

def get_targets(path, compressed=True):

    data = open_file(path, compressed)

    n = len(data)

    targets = []
    
    for i in range(n):

        points = data[i]['observable_sets'][0]['points']

        xs = sorted(set(p[x_key] for p in points))
        ys = sorted(set(p[y_key] for p in points))

        energy = data[i]['levels'][0]['energy']
        energy_norm = transforms._normalise(energy, min=min(ys), max=max(ys))

        gamma_total = data[i]['levels'][0]['Gamma_total']
        log_gamma_total = float(np.log10(gamma_total))

        target = torch.tensor([energy, energy_norm, gamma_total, log_gamma_total])
        targets.append(target)

    return targets

def open_file(path, compressed=True):

    if compressed:
        with gzip.open(path, 'rb') as f:
            json_bytes = f.read()
            json_str = json_bytes.decode()
            data = json.loads(json_str)
    else:
        with open(path, 'r') as f:
            data = json.load(f)

    return data

# display a tensor of shape [H, W]
def display_tensor(tensor, name):

    tensor = tensor.permute(1, 0)

    plt.figure(figsize=(10, 6))
    plt.imshow(tensor.numpy(), cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.savefig(f'out/tensor/{name}')

# display an RGB image
def display_image(img, name):
    
    plt.figure(figsize=(10, 6))
    plt.imshow(img.permute(1, 2, 0).numpy(), aspect='auto')
    plt.axis('off')
    plt.savefig(f'out/image/{name}')

class EnergyLevelDataset(Dataset):

    def __init__(self, path, transform=None, log_cx=True, compressed=True):
        self.tensors = get_tensors(path, log_cx, compressed)
        self.targets = get_targets(path, compressed)
        self.transform = transform

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):

        tensor = self.tensors[idx]
        target = self.targets[idx]

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, target