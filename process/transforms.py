import torch
import torch.nn.functional as F
import torchvision.transforms
import process.data as data
import numpy as np

def get_augment_transform(noise_sigma_log10=0.1, amplitude_scale=0.2):

    ls = []

    if noise_sigma_log10 > 0.0:
        ls.append(_lambda(lambda x: _add_noise(x, noise_sigma_log10)))

    if amplitude_scale > 0.0:
        ls.append(_lambda(lambda x: _amplitude_scale(x, amplitude_scale)))

    transform = torchvision.transforms.Compose(ls)

    return transform

FLOOR = -7.9
def _crop(tensor, target, metadata, crop_energy=0.0, crop_angle=False, crop_channel=False):

    # mask out padding
    mask = (tensor > FLOOR).float()

    if crop_energy == 0.0 and not crop_angle and not crop_channel:
        return torch.stack([tensor, mask], dim=0), target

    E, C = tensor.shape

    n_entrances = metadata['n_entrances']
    n_exits = metadata['n_exits']
    n_pp = n_entrances * n_exits
    n_angles = metadata['n_angles']

    max_resonances = target['energy'].shape[0]
    energies = target['energy'].squeeze(1)
    n_true = int(target['class'][:, 1].sum().item())

    crop_mask = mask.clone()
    e_start = 0.0
    e_end = 1.0

    # pick a fraction of energies to cut from the top and bottom
    if crop_energy > 0.0:
        E_crop_ratio = np.random.rand() * crop_energy
        E_keep_ratio = 1.0 - E_crop_ratio
        e_start = np.random.rand() * (1.0 - E_keep_ratio)
        e_end = e_start + E_keep_ratio
        e_idx_start = int(e_start * E)
        e_idx_end = int(e_end * E)
        crop_mask[:e_idx_start, :] = 0.0
        crop_mask[e_idx_end:, :] = 0.0

    # drop entrance channels
    if crop_channel:
        n_keep = np.random.randint(1, n_entrances + 1)
        kept_entrances = np.random.choice(n_entrances, size=n_keep, replace=False)
        for ent in range(n_entrances):
            if ent not in kept_entrances:
                for ext in range(n_exits):
                    pp = ent * n_exits + ext
                    start = pp * n_angles
                    end = start + n_angles
                    crop_mask[:, start:end] = 0.0

    # drop angles across all channels
    if crop_angle:
        max_drop = n_angles - 3
        if max_drop > 0:
            n_drop = np.random.randint(0, max_drop + 1)
            drop_indices = np.random.choice(n_angles, size=n_drop, replace=False)
            for a in drop_indices:
                for pp in range(n_pp):
                    crop_mask[:, pp * n_angles + a] = 0.0

    cropped_tensor = torch.stack([torch.where(crop_mask > 0, tensor, torch.tensor(-8.0)), crop_mask], dim=0)

    # keep resonances within cropped energy range
    res_mask = torch.zeros(max_resonances, dtype=torch.bool)
    for i in range(n_true):
        e = energies[i].item()
        if e >= e_start and e <= e_end:
            res_mask[i] = True

    n_kept = res_mask.sum().item()

    cropped_target = {}
    target_keys = ['class', 'energy', 'gamma', 'gamma_mask', 'jpi_index']
    for k in target_keys:
        filtered = target[k][res_mask]
        pad_shape = (max_resonances - n_kept, *filtered.shape[1:])
        cropped_target[k] = torch.cat([filtered, torch.zeros(pad_shape, dtype=filtered.dtype)], dim=0)

    cropped_target['class'][n_kept:, 0] = 1.0
    cropped_target['n_res'] = _normalise(torch.tensor(n_kept, dtype=target['n_res'].dtype), 0, data.MAX_RESONANCES)

    return cropped_tensor, cropped_target

# Gaussian noise in log10 space, approx. lognormal multiplicative
# noise in linear
# reasonable range is 0.05-0.2
def _add_noise(tensor, noise_sigma_log10):
    noisy = tensor.clone()
    noisy[0] = tensor[0] + torch.randn_like(tensor[0]) * noise_sigma_log10
    return noisy

# random additive offset in log10 space (multiplicative scale in linear)
# simulates variation in overall cross-section magnitude
def _amplitude_scale(tensor, max_scale):
    offset = (torch.rand(1).item() * 2 - 1) * max_scale
    scaled = tensor.clone()
    scaled[0] = tensor[0] + offset
    return scaled

def _lambda(foo, flag=True):
    return torchvision.transforms.Lambda(foo if flag else (lambda x: x))

def _normalise(x, _min=None, _max=None):

    if _min is None:
        _min = x.min()
    if _max is None:
        _max = x.max()

    return (x - _min) / (_max - _min)

def _unsqueeze(x, dim):
    x = x.unsqueeze(dim=dim)
    return x

def _print_shape(x):
    print(x.shape)
    return x