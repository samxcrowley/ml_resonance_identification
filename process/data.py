import gzip
import json
import multiprocessing
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import process.transforms as transforms

# set in params.json, default is 20
MAX_RESONANCES = 20

# x, y are the axes and z is the value at each point
x_key = 'theta_3_cm'
y_key = 'ke_cm_in'
z_key = 'dsdO'

# process loaded JSON list into (tensors [list], targets [dict])
def process_json(data, n_x=6, n_y=512, clamp=1e-8):

    tensors = []

    class_targets = []
    energy_targets = []
    gamma_total_targets = []
    n_res_targets = []
    jpi_index_targets = []

    energies = []

    for sample in data:

        points = sample['observable_sets'][0]['points']

        # load tensor

        energies.append([p[y_key] for p in points])

        xs_raw = np.array([p[x_key] for p in points])
        ys_raw = np.array([p[y_key] for p in points])
        zs_raw = np.array([np.log10(max(p[z_key], clamp)) for p in points])

        y_min = ys_raw.min()
        y_max = ys_raw.max()

        xs_uniform = np.linspace(xs_raw.min(), xs_raw.max(), n_x)
        ys_uniform = np.linspace(y_min, y_max, n_y)

        grid_x, grid_y = np.meshgrid(xs_uniform, ys_uniform)

        grid_z = griddata((xs_raw, ys_raw), zs_raw, (grid_x, grid_y), method='linear')
        grid_z = np.nan_to_num(grid_z, nan=np.log10(clamp))

        tensors.append(torch.tensor(grid_z, dtype=torch.float32))
        
        # load targets

        class_target = torch.zeros([MAX_RESONANCES, 2], dtype=torch.float32)
        energy_target = torch.zeros([MAX_RESONANCES, 1], dtype=torch.float32)
        gamma_total_target = torch.zeros([MAX_RESONANCES, 1], dtype=torch.float32)
        jpi_index_target = torch.zeros([MAX_RESONANCES, 1], dtype=torch.float32)

        # filter to only resonances within the data energy range
        levels = [l for l in sample['levels'] if y_min <= l['energy'] <= y_max]
        n_resonances = len(levels)

        for n in range(n_resonances):

            level = levels[n]

            # normalise energy to [0, 1] relative to the uniform grid range
            energy = level['energy']
            energy = transforms._normalise(energy, y_min, y_max)
            energy_target[n] = energy

            # sum gammas and take the log
            gamma_total = sum(level['Gamma'])
            gamma_total = np.log10(max(gamma_total, clamp))
            gamma_total_target[n] = gamma_total

            # jpi set index
            jpi_index_target[n] = level['jpi_index']

        # fill classes: index 0 = no resonance, index 1 = resonance
        for n in range(MAX_RESONANCES):
            if n < n_resonances:
                class_target[n, 1] = 1.0
            else:
                class_target[n, 0] = 1.0

        class_targets.append(class_target)
        energy_targets.append(energy_target)
        gamma_total_targets.append(gamma_total_target)
        jpi_index_targets.append(jpi_index_target)
        n_res_norm = transforms._normalise(n_resonances, 0, MAX_RESONANCES)
        n_res_targets.append(n_res_norm)

    targets = {
        'class': torch.stack(class_targets),
        'energy': torch.stack(energy_targets),
        'gamma_total': torch.stack(gamma_total_targets),
        'jpi_index': torch.stack(jpi_index_targets),
        'n_res': torch.tensor(n_res_targets, dtype=torch.float32),
    }

    return tensors, targets

class ResonanceDataset(Dataset):

    # accepts preprocessed .pt files only
    def __init__(self, path, crop_strength, transform=None):

        saved = torch.load(path, weights_only=False)

        self.tensors = saved['tensors']
        self.targets = saved['targets']

        # shared memory so persistent DataLoader workers see updates
        self._crop_strength = multiprocessing.Value('d', crop_strength)

        self.transform = transform

    @property
    def crop_strength(self):
        return self._crop_strength.value

    @crop_strength.setter
    def crop_strength(self, value):
        self._crop_strength.value = value

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):

        tensor = self.tensors[idx]
        target = {key: self.targets[key][idx] for key in self.targets}

        # crop
        tensor, target = transforms._crop(tensor, target, self.crop_strength)

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, target

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