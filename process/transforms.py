import torch
import torch.nn.functional as F
import torchvision.transforms
import process.data as data

def get_augment_transform(noise_sigma_log10=0.1, amplitude_scale=0.2, gaussian_blur_sigma=0.0):

    ls = []

    if gaussian_blur_sigma > 0.0:
        ls.append(_lambda(lambda x: _gaussian_blur_1d(x, gaussian_blur_sigma)))

    if noise_sigma_log10 > 0.0:
        ls.append(_lambda(lambda x: _add_noise(x, noise_sigma_log10)))

    if amplitude_scale > 0.0:
        ls.append(_lambda(lambda x: _amplitude_scale(x, amplitude_scale)))

    transform = torchvision.transforms.Compose(ls)

    return transform

def _crop(
    tensor,
    target,
    metadata,
    crop_energy=0.0,
    min_angles=1,
    min_pp_combos=1,
    use_info_weight=False):

    FLOOR = -7.9
    VISIBILITY_WINDOW = 5

    E, C = tensor.shape
    n_pp = metadata['n_entrances'] * metadata['n_exits']
    n_angles = metadata['n_angles']

    max_resonances = target['energy'].shape[0]
    energies = target['energy'].squeeze(1)
    n_true = int(target['class'][:, 1].sum().item())

    crop_mask = (tensor > FLOOR).float()

    # drop pp_combos
    if min_pp_combos < n_pp:
        n_keep = torch.randint(min_pp_combos, n_pp + 1, (1,)).item()
        kept = torch.randperm(n_pp)[:n_keep]
        drop = torch.ones(n_pp, dtype=torch.bool)
        drop[kept] = False
        for pp in drop.nonzero(as_tuple=True)[0].tolist():
            crop_mask[:, pp * n_angles:(pp + 1) * n_angles] = 0.0

    # drop angles
    if min_angles < n_angles:
        for pp in range(n_pp):
            col_s = pp * n_angles
            col_e = col_s + n_angles
            if crop_mask[:, col_s:col_e].sum() == 0:
                continue  # pp_combo already fully masked
            n_keep = torch.randint(min_angles, n_angles + 1, (1,)).item()
            kept = torch.randperm(n_angles)[:n_keep]
            drop = torch.ones(n_angles, dtype=torch.bool)
            drop[kept] = False
            for a in drop.nonzero(as_tuple=True)[0].tolist():
                crop_mask[:, col_s + a] = 0.0

    # crop energy per-pp-combo
    if crop_energy > 0.0:
        for pp in range(n_pp):
            col_s = pp * n_angles
            col_e = col_s + n_angles
            active = crop_mask[:, col_s:col_e].sum(dim=1).nonzero(as_tuple=True)[0]
            if len(active) == 0:
                continue
            row_lo = active[0].item()
            row_hi = active[-1].item() + 1
            active_len = row_hi - row_lo
            ratio = torch.rand(1).item() * crop_energy
            keep = max(1, int((1.0 - ratio) * active_len))
            offset = torch.randint(0, max(1, active_len - keep + 1), (1,)).item()
            win_lo = row_lo + offset
            win_hi = win_lo + keep
            crop_mask[:win_lo, col_s:col_e] = 0.0
            crop_mask[win_hi:,  col_s:col_e] = 0.0

    # safety check (a very small portion of samples have zero cross section for some reason)
    if crop_mask.sum() == 0:
        crop_mask = torch.ones_like(crop_mask)

    # build output tensor [2, E, C]
    cropped_data = torch.where(crop_mask > 0, tensor, torch.tensor(-8.0))
    cropped_tensor = torch.stack([cropped_data, crop_mask], dim=0)

    # keep resonances with data within VISIBILITY_WINDOW bins of their energy
    res_mask = torch.zeros(max_resonances, dtype=torch.bool)
    for i in range(n_true):
        e_bin = int(energies[i].item() * E)
        bin_lo = max(0, e_bin - VISIBILITY_WINDOW)
        bin_hi = min(E, e_bin + VISIBILITY_WINDOW)
        if crop_mask[bin_lo:bin_hi, :].sum() > 0:
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
    cropped_target['e_min'] = target['e_min']
    cropped_target['e_max'] = target['e_max']

    if use_info_weight:
        info_weight = _get_info_weights(cropped_target, n_kept, cropped_tensor[1], metadata)
        cropped_target['info_weight'] = info_weight

    return cropped_tensor, cropped_target

# information weight per resonance
# W = sum_c[N_c * G_c] / NG
# N = total non-masked data points within G of energy level
# N_c = non-masked data points in c within G of energy level
# G = total width
# G_c = partial width in c
def _get_info_weights(target, n_kept, crop_mask, metadata):

    max_resonances = target['energy'].shape[0]
    info_weight = torch.zeros(max_resonances, dtype=torch.float32)

    if n_kept == 0:
        return info_weight

    E, C = crop_mask.shape
    e_min = target['e_min'].item()
    e_max = target['e_max'].item()
    e_range = e_max - e_min

    n_entrances = metadata['n_entrances']
    n_exits = metadata['n_exits']
    n_angles = metadata['n_angles']
    channel_pp_map = metadata['channel_pp_map']

    # precompute per-pp column sums for each energy bin row
    # pp_col_sums[pp, e] = sum of crop_mask[e, cols_involving_pp]
    n_particle_pairs = max(n_entrances, n_exits)
    pp_col_sums = torch.zeros(n_particle_pairs, E, dtype=torch.float32)
    for ent in range(n_entrances):
        for ext in range(n_exits):
            pp_combo_idx = ent * n_exits + ext
            col_start = pp_combo_idx * n_angles
            col_end = col_start + n_angles
            col_sum = crop_mask[:, col_start:col_end].sum(dim=1)
            pp_col_sums[ent] += col_sum
            pp_col_sums[ext] += col_sum

    # total non-masked points per energy bin row
    row_sums = crop_mask.sum(dim=1)

    for i in range(n_kept):

        e_norm = target['energy'][i].item()
        jpi_idx = int(target['jpi_index'][i].item())
        gammas_norm = target['gamma'][i]
        g_mask = target['gamma_mask'][i]

        # un-normalise gammas
        gamma_log = gammas_norm * (data.GAMMA_LOG_MAX - data.GAMMA_LOG_MIN) + data.GAMMA_LOG_MIN
        gamma_linear = (10.0 ** gamma_log) * g_mask

        G = gamma_linear.sum().item()
        if G <= 0:
            continue

        # energy window: E +- G in bin space
        G_norm = G / e_range
        e_bin = e_norm * E
        G_bins = G_norm * E
        bin_lo = max(0, int(e_bin - G_bins))
        bin_hi = min(E, int(e_bin + G_bins) + 1)

        if bin_lo >= bin_hi:
            continue

        N = row_sums[bin_lo:bin_hi].sum().item()

        if N == 0:
            continue

        # group partial widths by particle pair
        pp_indices = channel_pp_map[jpi_idx]
        pp_gammas = {}
        for ch_idx in range(len(pp_indices)):
            pp = pp_indices[ch_idx].item()
            if pp < 0:
                break
            if g_mask[ch_idx].item() == 0:
                continue
            pp_gammas[pp] = pp_gammas.get(pp, 0.0) + gamma_linear[ch_idx].item()

        # weight = (sum_c N_c * G_c) / (N * G)
        numerator = 0.0
        for pp, Gc in pp_gammas.items():
            Nc = pp_col_sums[pp, bin_lo:bin_hi].sum().item()
            numerator += Nc * Gc

        info_weight[i] = numerator / (N * G)

    return info_weight

# Gaussian blur along the energy axis per channel (log space)
# sigma_bins: std dev in energy bins, reasonable range is 0.5-3.0
def _gaussian_blur_1d(tensor, sigma_bins):
    
    blurred = tensor.clone()
    data = tensor[0] # [E, C]

    sigma_bins = torch.rand(1).item() * sigma_bins
    if sigma_bins < 0.1:
        return blurred

    radius = max(1, int(3 * sigma_bins))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=data.device)
    kernel = torch.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel = kernel / kernel.sum()

    E, C = data.shape
    data_t = data.T.unsqueeze(1) # [C, 1, E]
    kernel_t = kernel.view(1, 1, -1)
    blurred[0] = F.conv1d(data_t, kernel_t, padding=radius).squeeze(1).T # [E, C]

    return blurred

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