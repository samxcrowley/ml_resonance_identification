import gzip
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import transforms

MAX_RESONANCES = 5

# x, y are the axes and z is the value at each point
x_key = 'theta_cm_out'
y_key = 'cn_ex'
z_key = 'dsdO'

# output sequence shape:
# [n, E, A]
def get_tensors(data_filename):

    data = open_data_file(f'data/{data_filename}')

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

            z = np.log10(p[z_key])

            x = x_idx[p[x_key]]
            y = y_idx[p[y_key]]
            tensor[y, x] = z

        tensors.append(tensor)

    return tensors

def get_targets(data_filename, n_class_targets=2, n_reg_targets=2):
    
    data = open_data_file(f'data/{data_filename}')

    n = len(data)

    class_targets = []
    energy_targets = []
    gamma_total_targets = []
    n_res_targets = []
    
    for i in range(n):

        points = data[i]['observable_sets'][0]['points']
        xs, ys = get_xs_ys(points)

        n_resonances = len(data[i]['levels'])

        class_target = torch.zeros([MAX_RESONANCES, n_class_targets], dtype=torch.float32)

        energy_target = torch.zeros([MAX_RESONANCES, 1], dtype=torch.float32)
        gamma_total_target = torch.zeros([MAX_RESONANCES, 1], dtype=torch.float32)

        for n in range(n_resonances):

            level = data[i]['levels'][n]

            energy = level['energy']
            energy = transforms._normalise(energy, min(ys), max(ys))
            energy_target[n] = energy

            gamma_total = level['Gamma_total']
            gamma_total = np.log10(gamma_total)
            gamma_total_target[n] = gamma_total

        # fill class targets
        # class 0: no resonance
        # class 1: resonance
        for n in range(MAX_RESONANCES):

            if n < n_resonances:
                class_target[n, 1] = 1.0
            else:
                class_target[n, 0] = 1.0

        class_targets.append(class_target)
        energy_targets.append(energy_target)
        gamma_total_targets.append(gamma_total_target)

        n_resonances = transforms._normalise(n_resonances, 0, 10)
        n_res_targets.append(n_resonances)

    class_targets = torch.stack(class_targets, dim=0)
    energy_targets = torch.stack(energy_targets, dim=0)
    gamma_total_targets = torch.stack(gamma_total_targets, dim=0)
    n_res_targets = torch.tensor(n_res_targets, dtype=torch.float32)

    return {
        'class': class_targets,
        'energy': energy_targets,
        'gamma_total': gamma_total_targets,
        'n_res': n_res_targets
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

    def __init__(self, path, transform=None):

        # self.tensors = get_tensors(path)
        # self.targets = get_targets(path)

        self.tensors = torch.load('data/processed/5res_training.gz_tensors.pt')
        self.targets = torch.load('data/processed/5res_training.gz_targets.pt')

        self.transform = transform

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):

        tensor = self.tensors[idx]
        if self.transform:
            tensor = self.transform(tensor)
        
        target = {}
        for key in self.targets.keys():
            target[key] = self.targets[key][idx]

        return tensor, target