import gzip
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import transforms
from targets import Target

MAX_RESONANCES = 10

# x, y are the axes and z is the value at each point
x_key = 'theta_cm_out'
y_key = 'cn_ex'
z_key = 'dsdO'

# output sequence shape:
# [n, E, A]
def get_tensors(data_filename, log_cx=True, compressed=True):

    data = open_file(f'data/{data_filename}', compressed)

    n = len(data)

    tensors = []
    
    for i in range(n):
        
        points = data[i]['observable_sets'][0]['points']

        xs = sorted(set(p[x_key] for p in points))
        ys = sorted(set(p[y_key] for p in points))

        x_idx = {x: i for i, x in enumerate(xs)}
        y_idx = {y: i for i, y in enumerate(ys)}
        
        tensor = torch.zeros(len(ys), len(xs))

        for p in points:

            if log_cx:
                z = np.log10(p[z_key])
            else:
                z = p[z_key]

            x = x_idx[p[x_key]]
            y = y_idx[p[y_key]]
            tensor[y, x] = z

        tensors.append(tensor)

    return tensors

def get_single_res_targets(data_filename, compressed=True):

    data = open_file(f'data/{data_filename}', compressed)

    n = len(data)

    targets = []
    
    for i in range(n):

        points = data[i]['observable_sets'][0]['points']
        xs, ys = get_xs_ys(points)

        n_resonances = len(data[i]['levels'])

        target = torch.zeros(10, len(Target), dtype=torch.float32)

        for n in range(n_resonances):

            level = data[i]['levels'][n]

            energy = level['energy']
            energy = transforms._normalise(energy, min=min(ys), max=max(ys))
            target[n, Target.ENERGY_LEVEL.value] = float(energy)

            gamma_total = level['Gamma_total']
            gamma_total = float(np.log10(gamma_total))
            target[n, Target.GAMMA_TOTAL.value] = float(gamma_total)

        targets.append(target)

    return torch.stack(targets, dim=0)

# returns a mask targets tensor of shape [n, MAX_RESONANCES, 1]
# and a regression targets tensor of shape [n, MAX_RESONANCES, 2]
def get_multi_res_targets(data_filename, compressed=True):

    data = open_file(f'data/{data_filename}', compressed)

    n = len(data)

    mask_targets = []
    reg_targets = []
    
    for i in range(n):

        points = data[i]['observable_sets'][0]['points']
        xs, ys = get_xs_ys(points)

        n_resonances = len(data[i]['levels'])

        mask_target = torch.zeros([MAX_RESONANCES, 1], dtype=torch.float32)
        reg_target = torch.zeros([MAX_RESONANCES, 2], dtype=torch.float32)

        for n in range(n_resonances):

            level = data[i]['levels'][n]

            mask_target[n, 0] = 1.0

            energy = level['energy']
            energy = transforms._normalise(energy, min(ys), max(ys))
            reg_target[n, 0] = energy

            gamma_total = level['Gamma_total']
            gamma_total = np.log10(gamma_total)
            reg_target[n, 1] = gamma_total

        mask_targets.append(mask_target)
        reg_targets.append(reg_target)

    mask_targets = torch.stack(mask_targets, dim=0)
    reg_targets = torch.stack(reg_targets, dim=0)

    return mask_targets, reg_targets

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

def get_xs_ys(points):

    xs = sorted(set(p[x_key] for p in points))
    ys = sorted(set(p[y_key] for p in points))

    return xs, ys

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

class ResonanceDataset(Dataset):

    def __init__(self, path, multi_resonance=False, transform=None, log_cx=True, compressed=True):

        self.tensors = get_tensors(path, log_cx, compressed)
        self.multi_resonance = multi_resonance

        if self.multi_resonance:
            self.targets = get_multi_res_targets(path, compressed)
        else:
            self.targets = get_single_res_targets(path, compressed)

        self.transform = transform

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):

        tensor = self.tensors[idx]
        
        if self.multi_resonance:
            target = (self.targets[0][idx], self.targets[1][idx])
        else:
            target = self.targets[idx]

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, target