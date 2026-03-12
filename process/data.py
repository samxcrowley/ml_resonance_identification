import gzip
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import process.transforms as transforms
from process.header import Header

# set in params.json, default is 20
MAX_RESONANCES = 20

# process loaded JSON list into (tensors [list], targets [dict])
# tensor shape: [n_y, n_channels]
# where n_channels = n_pp_combos * n_angles (for now, 9 * 6 = 54)
def process_json(data, n_y=512, clamp=1e-8):

    tensors = []

    class_targets = []
    energy_targets = []
    gamma_targets = []
    gamma_total_targets = []
    n_res_targets = []
    jpi_index_targets = []

    header = Header()

    for sample in data:

        points = sample['observable_sets'][0]['points']

        # get pp combinations and angles
        pp_combos = sorted(set((p['pp_in_index'], p['pp_out_index']) for p in points))
        angles = sorted(set(p['theta_3_cm'] for p in points))

        n_channels = len(pp_combos) * len(angles) # 54

        cn_ex_all = np.array([p['cn_ex'] for p in points])
        e_min = cn_ex_all.min()
        e_max = cn_ex_all.max()
        e_uniform = np.linspace(e_min, e_max, n_y)

        # build grid: 1D interpolation per (pp_in, pp_out, angle)
        grid = np.full((n_y, n_channels), np.log10(clamp), dtype=np.float32)

        # index points by (pp_in, pp_out, angle) for fast lookup
        # O(n_channels + n_points)
        # instead of naive approach of looping over all channels
        # and all points every time, and adding each -- O(n_channels * n_points)
        # this approach scans through all points, which are universal to the
        # channels and buckets them into a dict, then over all channels and
        # looks them up in dict in O(1) time.
        point_groups = {}
        for p in points:

            key = (p['pp_in_index'], p['pp_out_index'], p['theta_3_cm'])

            if key not in point_groups:
                point_groups[key] = ([], [])

            point_groups[key][0].append(p['cn_ex'])
            point_groups[key][1].append(np.log10(max(p['dsdO'], clamp)))

        ch_idx = 0
        for pp_in, pp_out in pp_combos:
            for angle in angles:

                key = (pp_in, pp_out, angle)

                if key in point_groups:

                    e = np.array(point_groups[key][0])
                    z = np.array(point_groups[key][1])

                    order = np.argsort(e)

                    grid[:, ch_idx] = np.interp(
                        e_uniform,
                        e[order],
                        z[order],
                        left=np.log10(clamp),
                        right=np.log10(clamp)
                    )

                ch_idx += 1

        grid = np.nan_to_num(grid, nan=np.log10(clamp))
        tensors.append(torch.tensor(grid, dtype=torch.float32))

        # load targets
        class_target = torch.zeros([MAX_RESONANCES, 2], dtype=torch.float32)
        energy_target = torch.zeros([MAX_RESONANCES, 1], dtype=torch.float32)
        gamma_target = torch.zeros([MAX_RESONANCES, header.max_channels], dtype=torch.float32)
        gamma_total_target = torch.zeros([MAX_RESONANCES, 1], dtype=torch.float32)
        jpi_index_target = torch.zeros([MAX_RESONANCES, 1], dtype=torch.float32)

        # filter to only resonances within the energy range
        levels = [l for l in sample['levels'] if e_min <= l['energy'] <= e_max]
        n_resonances = len(levels)

        for n in range(n_resonances):

            level = levels[n]

            energy = level['energy']
            energy = transforms._normalise(energy, e_min, e_max)
            energy_target[n] = energy

            for i in range(len(level['Gamma'])):
                gamma_target[n, i] = np.log10(max(level['Gamma'][i], clamp))

            gamma_total = sum(level['Gamma'])
            gamma_total = np.log10(max(gamma_total, clamp))
            gamma_total_target[n] = gamma_total

            jpi_index_target[n] = level['jpi_index']

        # fill classes
        # index 0 = no resonance, index 1 = resonance
        for n in range(MAX_RESONANCES):
            if n < n_resonances:
                class_target[n, 1] = 1.0
            else:
                class_target[n, 0] = 1.0

        class_targets.append(class_target)
        energy_targets.append(energy_target)
        gamma_targets.append(gamma_target)
        gamma_total_targets.append(gamma_total_target)
        jpi_index_targets.append(jpi_index_target)
        n_res_norm = transforms._normalise(n_resonances, 0, MAX_RESONANCES)
        n_res_targets.append(n_res_norm)

    targets = {
        'class': torch.stack(class_targets),
        'energy': torch.stack(energy_targets),
        'gamma': torch.stack(gamma_targets),
        'gamma_total': torch.stack(gamma_total_targets),
        'jpi_index': torch.stack(jpi_index_targets),
        'n_res': torch.tensor(n_res_targets, dtype=torch.float32),
    }

    return tensors, targets

class ResonanceDataset(Dataset):

    # accepts preprocessed .pt files only
    def __init__(self, path, max_crop=0.0, transform=None):

        saved = torch.load(path, weights_only=False)

        self.tensors = saved['tensors']
        self.targets = saved['targets']
        self.max_crop = max_crop
        self.transform = transform

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):

        tensor = self.tensors[idx]
        target = {key: self.targets[key][idx] for key in self.targets}

        # crop
        tensor, target = transforms._crop(tensor, target, np.random.rand() * self.max_crop)

        # initial transform
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