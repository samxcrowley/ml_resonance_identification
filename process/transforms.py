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

def _crop(tensor, target, strength=0.0):

    E, A = tensor.shape

    if strength == 0.0:
        mask = torch.ones(E, A)
        return torch.stack([tensor, mask], dim=0), target

    # pick a fraction of energies to cut out
    E_crop_ratio = np.random.rand() * strength
    E_keep_ratio = 1.0 - E_crop_ratio
    e_start = np.random.rand() * (1.0 - E_keep_ratio)
    e_end = e_start + E_keep_ratio
    e_idx_start = int(e_start * E)
    e_idx_end = int(e_end * E)

    # build spatial mask
    # 0: cropped
    # 1: valid
    mask = torch.ones(E, A)
    mask[:e_idx_start, :] = 0.0
    mask[e_idx_end:, :] = 0.0

    # randomly reduce energy resolution
    strides = [1, 1, 2, 2, 3]
    if np.random.rand() < strength:
        stride = strides[np.random.randint(len(strides))]
        if stride > 1:
            for e in range(e_idx_start, e_idx_end):
                if (e - e_idx_start) % stride != 0:
                    mask[e, :] = 0.0

    # crop out angles (same angles dropped across all 9 pp combos)
    n_pp = 9
    n_angles = A // n_pp
    for a in range(n_angles):
        if np.random.rand() <= (strength / 2):
            for pp in range(n_pp):
                mask[:, pp * n_angles + a] = 0.0

    # * operation zeros out cropped values
    cropped_tensor = torch.stack([tensor * mask, mask], dim=0)

    # filter resonances that fall within the crop window
    max_resonances = target['energy'].shape[0]
    energies = target['energy'].squeeze(1)
    res_mask = (energies >= e_start) & (energies <= e_end)

    n_kept = res_mask.sum().item()

    # pad filtered targets back to max_resonances
    cropped_target = {}
    target_keys = ['class', 'energy', 'gamma', 'gamma_mask', 'jpi_index']
    for k in target_keys:
        filtered = target[k][res_mask]
        pad_shape = (max_resonances - n_kept, *filtered.shape[1:])
        cropped_target[k] = torch.cat([filtered, torch.zeros(pad_shape, dtype=filtered.dtype)], dim=0)

    # mark padded slots as no-resonance in class target
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