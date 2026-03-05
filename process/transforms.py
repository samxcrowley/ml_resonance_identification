import torch
import torch.nn.functional as F
import torchvision.transforms
import process.data as data
import numpy as np

def _detr_transform(sobel=True, noise_sigma_log10=0.1):

    ls = []

    # if noise_sigma_log10 > 0.0:
    #     ls.append(_lambda(lambda x: _add_noise(x, noise_sigma_log10)))

    if sobel:
        ls.append(_lambda(lambda x: _sobel(x)))

    # ls.append(_lambda(lambda x: _unsqueeze(x, dim=0)))

    transform = torchvision.transforms.Compose(ls)

    return transform

def _resnet_transform(sobel=False):

    ls = []

    ls.append(_lambda(lambda x: _normalise(x)))
    if sobel:
        ls.append(_lambda(lambda x: _sobel(x)))
    ls.append(_lambda(lambda x: x.repeat(3, 1, 1)))
    ls.append(torchvision.models.ResNet34_Weights.DEFAULT.transforms())

    transform = torchvision.transforms.Compose(ls)

    return transform

def _encoder_transform(sobel=False):

    ls = []

    ls.append(_lambda(lambda x: _normalise(x)))
    if sobel:
        ls.append(_lambda(lambda x: _sobel(x)))

    transform = torchvision.transforms.Compose(ls)

    return transform

def _crop(tensor, target, strength=0.0):

    E, A = tensor.shape

    if strength == 0.0:
        mask = torch.ones(E, A)
        return torch.stack([tensor, mask], dim=0), target

    # pick a fraction of energies to cut out (up to strength)
    E_crop_ratio = np.random.rand() * strength
    E_keep_ratio = 1.0 - E_crop_ratio

    e_start = np.random.rand() * (1.0 - E_keep_ratio)
    e_end = e_start + E_keep_ratio

    e_idx_start = int(e_start * E)
    e_idx_end = int(e_end * E)

    # pick a block of angles to crop out
    A_crop_n = np.random.randint(0, A)
    A_crop_start = np.random.randint(0, A - A_crop_n + 1)
    A_crop_end = A_crop_start + A_crop_n

    # build spatial mask
    # 0: cropped
    # 1: valid
    mask = torch.ones(E, A)
    mask[e_idx_start:e_idx_end, A_crop_start:A_crop_end] = 0.0
    mask[:e_idx_start, :] = 0.0
    mask[e_idx_end:, :] = 0.0

    # * operation zeros out cropped values
    cropped_tensor = torch.stack([tensor * mask, mask], dim=0)

    # filter resonances that fall within the crop window
    max_resonances = target['energy'].shape[0]
    energies = target['energy'].squeeze(1)
    res_mask = (energies >= e_start) & (energies <= e_end)

    n_kept = res_mask.sum().item()

    # pad filtered targets back to max_resonances
    cropped_target = {}
    target_keys = ['class', 'energy', 'gamma_total', 'jpi_index']
    for k in target_keys:
        filtered = target[k][res_mask]
        pad_shape = (max_resonances - n_kept, *filtered.shape[1:])
        cropped_target[k] = torch.cat([filtered, torch.zeros(pad_shape, dtype=filtered.dtype)], dim=0)

    # re-normalise energy
    cropped_target['energy'][:n_kept] = (cropped_target['energy'][:n_kept] - e_start) / (e_end - e_start)

    # mark padded slots as no-resonance in class target
    cropped_target['class'][n_kept:, 0] = 1.0

    cropped_target['n_res'] = _normalise(torch.tensor(n_kept, dtype=target['n_res'].dtype), 0, data.MAX_RESONANCES)

    return cropped_tensor, cropped_target

# Gaussian noise in log10 space, approx. lognormal multiplicative
# noise in linear
# reasonable range is 0.05-0.2
def _add_noise(tensor, noise_sigma_log10):
    return tensor + torch.randn_like(tensor) * noise_sigma_log10

def _lambda(foo, flag=True):
    return torchvision.transforms.Lambda(foo if flag else (lambda x: x))

def _normalise(x, min=None, max=None):

    if min == None:
        min = x.min()
    if max == None:
        max = x.max()

    return (x - min) / (max - min)

def _unsqueeze(x, dim):
    x = x.unsqueeze(dim=dim)
    return x

def _print_shape(x):
    print(x.shape)
    return x

def _sobel(x):

    Kx = torch.tensor([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=torch.float32)
    Ky = torch.tensor([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=torch.float32)

    has_mask = x.dim() == 3
    if has_mask:
        data_ch = x[0]
        mask = x[1]
    else:
        data_ch = x

    # [1, 1, E, A]
    inp = data_ch.float().unsqueeze(0).unsqueeze(0)

    # [1, 1, 3, 3]
    Kx = Kx.unsqueeze(0).unsqueeze(0)
    Ky = Ky.unsqueeze(0).unsqueeze(0)

    Gx = F.conv2d(inp, Kx, padding=1)
    Gy = F.conv2d(inp, Ky, padding=1)

    magnitude = torch.sqrt(Gx**2 + Gy**2).squeeze()

    if has_mask:
        # zero out false edges at crop boundaries
        magnitude = magnitude * mask
        return torch.stack([magnitude, mask], dim=0)

    return magnitude