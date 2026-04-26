import gzip
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import process.transforms as transforms
from process.header import Header

# set in params.json, default is 20
MAX_RESONANCES = 20

GAMMA_LOG_MIN = -8.0 # log10(1e-8)
GAMMA_LOG_MAX = 2.0 # log10(100 MeV)

# process experimental JSON data (no targets)
# returns tensors and metadata
# angles are continuous and binned into n_angle_bins
def process_exp_json(data, n_y=512, n_angle_bins=15, clamp=1e-8):

    tensors = []

    samples = data['data']

    # get pp combos
    pp_combos = sorted(set((s['pp_in_index'], s['pp_out_index']) for s in samples))
    n_channels = len(pp_combos) * n_angle_bins

    for sample in samples:

        points = sample['points']
        pp_in = sample['pp_in_index']
        pp_out = sample['pp_out_index']
        pp_idx = pp_combos.index((pp_in, pp_out))

        cn_ex_all = np.array([p['cn_ex'] for p in points])
        e_min = cn_ex_all.min()
        e_max = cn_ex_all.max()
        e_uniform = np.linspace(e_min, e_max, n_y)

        # bin continuous angles
        theta_all = np.array([p['theta_cm_out'] for p in points])
        angle_bins = np.linspace(theta_all.min(), theta_all.max(), n_angle_bins + 1)

        # group points by angle bin
        point_groups = [[] for _ in range(n_angle_bins)]
        for p in points:
            bin_idx = np.searchsorted(angle_bins[1:], p['theta_cm_out'], side='right')
            bin_idx = min(bin_idx, n_angle_bins - 1)
            point_groups[bin_idx].append(p)

        # build grid: 1D interpolation per angle bin
        grid = np.full((n_y, n_channels), np.log10(clamp), dtype=np.float32)

        for a_idx in range(n_angle_bins):

            if not point_groups[a_idx]:
                continue

            ch_idx = pp_idx * n_angle_bins + a_idx

            e = np.array([p['cn_ex'] for p in point_groups[a_idx]])
            z = np.array([np.log10(max(p['dsdO'], clamp)) for p in point_groups[a_idx]])

            order = np.argsort(e)

            grid[:, ch_idx] = np.interp(
                e_uniform, e[order], z[order],
                left=np.log10(clamp), right=np.log10(clamp)
            )

        grid = np.nan_to_num(grid, nan=np.log10(clamp))
        tensors.append(torch.tensor(grid, dtype=torch.float32))

    entrances = sorted(set(pp[0] for pp in pp_combos))
    exits = sorted(set(pp[1] for pp in pp_combos))

    metadata = {
        'n_entrances': len(entrances),
        'n_exits': len(exits),
        'n_angles': n_angle_bins,
        'n_pp_combos': len(pp_combos),
        'pp_combos': pp_combos,
        'angles': angle_bins[:-1].tolist(),
    }

    return tensors, metadata

# process loaded JSON list into (tensors [list], targets [dict], metadata [dict])
# tensor shape: [n_y, n_channels]
# where n_channels = n_pp_combos * n_angles
def process_json(data, n_y=512, clamp=1e-8):

    tensors = []

    class_targets = []
    energy_targets = []
    gamma_targets = []
    gamma_mask_targets = []
    n_res_targets = []
    jpi_index_targets = []
    e_min_targets = []
    e_max_targets = []

    header = Header()

    # get pp combos and angles from first sample
    first_points = data[0]['observable_sets'][0]['points']
    pp_combos = sorted(set((p['pp_in_index'], p['pp_out_index']) for p in first_points))
    angles = sorted(set(p['theta_3_cm'] for p in first_points))
    n_channels = len(pp_combos) * len(angles)

    for sample in data:

        points = sample['observable_sets'][0]['points']

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
        gamma_mask_target = torch.zeros([MAX_RESONANCES, header.max_channels], dtype=torch.float32)
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
                val = level['Gamma'][i]
                if val is None or np.isnan(val):
                    continue
                g = np.clip(np.log10(max(val, clamp)), GAMMA_LOG_MIN, GAMMA_LOG_MAX)
                gamma_target[n, i] = (g - GAMMA_LOG_MIN) / (GAMMA_LOG_MAX - GAMMA_LOG_MIN)
                gamma_mask_target[n, i] = 1.0

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
        gamma_mask_targets.append(gamma_mask_target)
        jpi_index_targets.append(jpi_index_target)
        n_res_norm = transforms._normalise(n_resonances, 0, MAX_RESONANCES)
        n_res_targets.append(n_res_norm)
        e_min_targets.append(e_min)
        e_max_targets.append(e_max)

    targets = {
        'class': torch.stack(class_targets),
        'energy': torch.stack(energy_targets),
        'gamma': torch.stack(gamma_targets),
        'gamma_mask': torch.stack(gamma_mask_targets),
        'jpi_index': torch.stack(jpi_index_targets),
        'n_res': torch.tensor(n_res_targets, dtype=torch.float32),
        'e_min': torch.tensor(e_min_targets, dtype=torch.float32),
        'e_max': torch.tensor(e_max_targets, dtype=torch.float32),
    }

    entrances = sorted(set(pp[0] for pp in pp_combos))
    exits = sorted(set(pp[1] for pp in pp_combos))

    # channel to particle pair mapping for each jpi set
    channel_pp_map = torch.full([header.n_jpi_sets, header.max_channels], -1, dtype=torch.long)
    for j, jpi_set in enumerate(header.jpi_sets):
        for i, ch in enumerate(jpi_set['channels']):
            channel_pp_map[j, i] = ch['pp_index']

    metadata = {
        'n_entrances': len(entrances),
        'n_exits': len(exits),
        'n_angles': len(angles),
        'n_pp_combos': len(pp_combos),
        'pp_combos': pp_combos,
        'angles': angles,
        'channel_pp_map': channel_pp_map,
    }

    return tensors, targets, metadata

class ResonanceDataset(Dataset):

    # accepts preprocessed .pt files only
    # pass tensors, targets, metadata to share data from another dataset
    # without reloading from disk
    # channel_filter: 'elastic' (pp_in==pp_out), 'inelastic' (pp_in!=pp_out), or None
    def __init__(self, path, crop_params=None, transform=None,
                 tensors=None, targets=None, metadata=None,
                 crop_fn=None, channel_filter=None):

        if tensors is not None and targets is not None:
            self.tensors = tensors
            self.targets = targets
            default_metadata = {
                'n_entrances': 3, 'n_exits': 3,
                'n_angles': self.tensors[0].shape[1] // 9,
            }
            self.metadata = metadata or default_metadata
        else:
            saved = torch.load(path, weights_only=False)
            self.tensors = saved['tensors']
            self.targets = saved['targets']
            default_metadata = {
                'n_entrances': 3, 'n_exits': 3,
                'n_angles': self.tensors[0].shape[1] // 9,
            }
            self.metadata = saved.get('metadata', default_metadata)

        if channel_filter is not None:
            self.apply_channel_filter(channel_filter)

        self.crop_params = crop_params or {}
        self.transform = transform
        self.crop_fn = crop_fn if crop_fn is not None else transforms._crop

    def apply_channel_filter(self, channel_filter):

        pp_combos = self.metadata.get('pp_combos', [])
        n_angles = self.metadata['n_angles']

        if channel_filter == 'elastic':
            keep = [i for i, (pp_in, pp_out) in enumerate(pp_combos) if pp_in == pp_out]
        elif channel_filter == 'inelastic':
            keep = [i for i, (pp_in, pp_out) in enumerate(pp_combos) if pp_in != pp_out]
        else:
            raise ValueError(f"Unknown channel_filter: {channel_filter!r} (expected 'elastic' or 'inelastic')")

        keep_cols = [i * n_angles + a for i in keep for a in range(n_angles)]
        self.tensors = [t[:, keep_cols] for t in self.tensors]

        filtered_pp = [pp_combos[i] for i in keep]
        self.metadata = dict(self.metadata)
        self.metadata['pp_combos'] = filtered_pp
        self.metadata['n_pp_combos'] = len(filtered_pp)

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):

        tensor = self.tensors[idx]
        target = {key: self.targets[key][idx] for key in self.targets}

        # crop
        tensor, target = self.crop_fn(tensor, target, self.metadata, **self.crop_params)

        # initial transform
        if self.transform:
            tensor = self.transform(tensor)

        return tensor, target

def open_data_file(path):

    if path.endswith('jsonl.gz'):

        with gzip.open(path, 'rb') as f:
            data = [json.loads(line) for line in f]

    elif path.endswith('gz'):

        with gzip.open(path, 'rb') as f:
            json_bytes = f.read()
            json_str = json_bytes.decode()
            data = json.loads(json_str)

    else:
        
        print('Invalid data file type.')
        return None

    return data