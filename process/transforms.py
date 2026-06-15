import torch
import torch.nn.functional as F
import torchvision.transforms
import process.data as data

FLOOR = -7.9
FLOOR_FILL = -8.0

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
    max_pp_combos=9,
    elastic_max_pp_combos=None,
    contiguous_angle_crop_p=0.0,
    shared_energy_crop_p=0.0,
    inelastic_dropout_p=0.0,
    min_kept_channel_weight=0.0):

    E, C = tensor.shape
    n_pp = metadata.get('n_pp_combos', metadata['n_entrances'] * metadata['n_exits'])
    n_angles = metadata['n_angles']

    natural_mask = (tensor > FLOOR).float()
    crop_mask = natural_mask.clone()

    pp_combos = metadata.get('pp_combos')
    if not pp_combos:
        n_exits = metadata['n_exits']
        pp_combos = [(pp // n_exits, pp % n_exits) for pp in range(n_pp)]

    elastic_pps = [pp for pp, (pp_in, pp_out) in enumerate(pp_combos) if pp_in == pp_out]
    force_elastic_only = (
        len(elastic_pps) > 0 and
        inelastic_dropout_p > 0.0 and
        torch.rand(1).item() < inelastic_dropout_p
    )

    allowed_pps = elastic_pps if force_elastic_only else list(range(n_pp))
    pp_upper_limit = max_pp_combos
    if force_elastic_only and elastic_max_pp_combos is not None:
        pp_upper_limit = elastic_max_pp_combos
    pp_upper = min(pp_upper_limit, len(allowed_pps))
    min_keep = min(min_pp_combos, pp_upper)

    if pp_upper > 0:
        if min_keep < pp_upper:
            n_keep = torch.randint(min_keep, pp_upper + 1, (1,)).item()
        else:
            n_keep = pp_upper
        kept_allowed = torch.randperm(len(allowed_pps))[:n_keep].tolist()
        kept_pps = [allowed_pps[i] for i in kept_allowed]
    else:
        kept_pps = []

    pp_keep_mask = torch.zeros(n_pp, dtype=torch.bool)
    if kept_pps:
        pp_keep_mask[torch.tensor(kept_pps, dtype=torch.long)] = True

    for pp in (~pp_keep_mask).nonzero(as_tuple=True)[0].tolist():
        crop_mask[:, pp * n_angles:(pp + 1) * n_angles] = 0.0

    # drop angles
    if min_angles < n_angles:
        for pp in range(n_pp):
            col_s = pp * n_angles
            col_e = col_s + n_angles
            if crop_mask[:, col_s:col_e].sum() == 0:
                continue
            n_keep = torch.randint(min_angles, n_angles + 1, (1,)).item()
            use_contiguous = (
                contiguous_angle_crop_p > 0.0 and
                torch.rand(1).item() < contiguous_angle_crop_p
            )
            if use_contiguous:
                start = torch.randint(0, n_angles - n_keep + 1, (1,)).item()
                kept = torch.arange(start, start + n_keep)
            else:
                kept = torch.randperm(n_angles)[:n_keep]
            drop = torch.ones(n_angles, dtype=torch.bool)
            drop[kept] = False
            for a in drop.nonzero(as_tuple=True)[0].tolist():
                crop_mask[:, col_s + a] = 0.0

    # crop energy per column, vectorised across columns. The shared option
    # gives all kept channels one contiguous experimental-style energy window.
    if crop_energy > 0.0:
        active_cols = crop_mask.sum(dim=0) > 0
        if active_cols.any():
            col_mask = crop_mask > 0
            row_lo = col_mask.float().argmax(dim=0)
            row_hi = E - col_mask.flip(0).float().argmax(dim=0)
            active_len = (row_hi - row_lo).clamp(min=1)

            use_shared_energy = (
                shared_energy_crop_p > 0.0 and
                torch.rand(1).item() < shared_energy_crop_p
            )
            if use_shared_energy:
                active_lo = row_lo[active_cols].min()
                active_hi = row_hi[active_cols].max()
                shared_len = (active_hi - active_lo).clamp(min=1)
                ratio = torch.rand((), device=tensor.device) * crop_energy
                keep = ((1.0 - ratio) * shared_len.float()).long().clamp(min=1)
                max_offset = (shared_len - keep + 1).clamp(min=1)
                offset = torch.randint(0, int(max_offset.item()), (), device=tensor.device)
                win_lo = torch.full((C,), int((active_lo + offset).item()), dtype=torch.long, device=tensor.device)
                win_hi = torch.full((C,), int((active_lo + offset + keep).item()), dtype=torch.long, device=tensor.device)
            else:
                ratios = torch.rand(C, device=tensor.device) * crop_energy
                keep = ((1.0 - ratios) * active_len.float()).long().clamp(min=1)
                max_offsets = (active_len - keep + 1).clamp(min=1)
                offsets = (torch.rand(C, device=tensor.device) * max_offsets.float()).long()
                win_lo = row_lo + offsets
                win_hi = win_lo + keep

            rows = torch.arange(E, device=tensor.device).unsqueeze(1)
            keep_mask = (rows >= win_lo.unsqueeze(0)) & (rows < win_hi.unsqueeze(0))
            crop_mask = crop_mask * torch.where(active_cols.unsqueeze(0), keep_mask, torch.zeros_like(keep_mask)).float()

    # safety: restore the selected pp-combos if angle/energy crops removed
    # everything, instead of falling back to all channels
    if crop_mask.sum() == 0:
        crop_mask = torch.zeros_like(natural_mask)
        for pp in pp_keep_mask.nonzero(as_tuple=True)[0].tolist():
            col_s = pp * n_angles
            crop_mask[:, col_s:col_s + n_angles] = natural_mask[:, col_s:col_s + n_angles]
        if crop_mask.sum() == 0:
            crop_mask = natural_mask

    return _finalise_crop(
        tensor, crop_mask, target, metadata,
        min_kept_channel_weight=min_kept_channel_weight,
    )

# apply a single pre-defined visibility mask uniformly to every sample with no curriculum
def _apply_fixed_mask(tensor, target, metadata, mask=None, min_kept_channel_weight=0.0):

    natural_mask = (tensor > FLOOR).float()

    if mask is None:
        crop_mask = natural_mask
    else:
        crop_mask = natural_mask * mask.to(dtype=tensor.dtype, device=tensor.device)

    return _finalise_crop(
        tensor, crop_mask, target, metadata,
        min_kept_channel_weight=min_kept_channel_weight,
    )

VISIBILITY_WINDOW = 5

# build model-ready [2, E, C] tensor from a mask and drop resonances that are masked out
def _finalise_crop(tensor, crop_mask, target, metadata, min_kept_channel_weight=0.0):

    E, C = tensor.shape
    n_pp = metadata.get('n_pp_combos', metadata['n_entrances'] * metadata['n_exits'])
    n_angles = metadata['n_angles']

    max_resonances = target['energy'].shape[0]
    energies = target['energy'].squeeze(1)
    n_true = int(target['class'][:, 1].sum().item())

    # build output tensor [2, E, C]
    cropped_data = torch.where(crop_mask > 0, tensor, tensor.new_full((), FLOOR_FILL))
    cropped_tensor = torch.stack([cropped_data, crop_mask], dim=0)

    # keep resonances with data within a fixed physical energy window
    # (5 bins at 512-bin resolution) of their energy
    # and kept_weight >= min_kept_channel_weight
    weight_per_channel = target.get('weight_per_channel', target.get('prominence_per_channel')) # [max_res, n_pp * n_angles]
    weight_per_combo = target.get('weight_per_combo', target.get('prominence_per_combo')) # [max_res, n_pp]
    use_channel_weight_filter = weight_per_channel is not None and min_kept_channel_weight > 0.0
    use_combo_weight_filter = (
        not use_channel_weight_filter and
        weight_per_combo is not None and
        min_kept_channel_weight > 0.0
    )

    res_mask = torch.zeros(max_resonances, dtype=torch.bool)
    for i in range(n_true):
        e_bin = int(energies[i].item() * E)
        bin_lo = max(0, e_bin - VISIBILITY_WINDOW)
        bin_hi = min(E, e_bin + VISIBILITY_WINDOW)
        win = crop_mask[bin_lo:bin_hi, :]
        if win.sum() == 0:
            continue
        if use_channel_weight_filter:
            channel_present = win.sum(dim=0) > 0
            if not channel_present.any():
                continue
            kept_weight = float(weight_per_channel[i, channel_present].max().item())
            if kept_weight < min_kept_channel_weight:
                continue
        elif use_combo_weight_filter:
            combo_present = win.view(win.shape[0], n_pp, n_angles).sum(dim=(0, 2)) > 0
            if not combo_present.any():
                continue
            kept_weight = float(weight_per_combo[i, combo_present].max().item())
            if kept_weight < min_kept_channel_weight:
                continue
        res_mask[i] = True

    n_kept = res_mask.sum().item()

    cropped_target = {}
    target_keys = ['class', 'energy', 'gamma', 'gamma_mask', 'jpi_index']
    for k in target_keys:
        filtered = target[k][res_mask]
        pad_shape = (max_resonances - n_kept, *filtered.shape[1:])
        cropped_target[k] = torch.cat([filtered, torch.zeros(pad_shape, dtype=filtered.dtype)], dim=0)

    cropped_target['class'][n_kept:, 0] = 1.0
    cropped_target['e_min'] = target['e_min']
    cropped_target['e_max'] = target['e_max']

    return cropped_tensor, cropped_target

# Gaussian blur along the energy axis per channel (log space)
# sigma_bins: std dev in energy bins, reasonable range is 0.5-3.0
def _gaussian_blur_1d(tensor, sigma_bins):

    blurred = tensor.clone()
    data = tensor[0] # [E, C]
    mask = tensor[1]

    sigma_bins = torch.rand(1).item() * sigma_bins
    if sigma_bins < 0.1:
        return blurred

    radius = max(1, int(3 * sigma_bins))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=data.device)
    kernel = torch.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel = kernel / kernel.sum()

    weighted_data = data * mask
    data_t = weighted_data.T.unsqueeze(1) # [C, 1, E]
    mask_t = mask.T.unsqueeze(1) # [C, 1, E]
    kernel_t = kernel.view(1, 1, -1)
    data_sum = F.conv1d(data_t, kernel_t, padding=radius).squeeze(1).T # [E, C]
    mask_sum = F.conv1d(mask_t, kernel_t, padding=radius).squeeze(1).T # [E, C]
    blurred_data = data_sum / mask_sum.clamp(min=1e-6)
    blurred[0] = torch.where(mask > 0, blurred_data, data.new_full((), FLOOR_FILL))

    return blurred

# Gaussian noise in log10 space, approx. lognormal multiplicative
# noise in linear
# reasonable range is 0.05-0.2
def _add_noise(tensor, noise_sigma_log10):
    noisy = tensor.clone()
    mask = tensor[1]
    noisy_data = tensor[0] + torch.randn_like(tensor[0]) * noise_sigma_log10
    noisy[0] = torch.where(mask > 0, noisy_data, tensor[0].new_full((), FLOOR_FILL))
    return noisy

# random additive offset in log10 space (multiplicative scale in linear)
# simulates variation in overall cross-section magnitude
def _amplitude_scale(tensor, max_scale):
    offset = (torch.rand(1).item() * 2 - 1) * max_scale
    scaled = tensor.clone()
    mask = tensor[1]
    scaled_data = tensor[0] + offset
    scaled[0] = torch.where(mask > 0, scaled_data, tensor[0].new_full((), FLOOR_FILL))
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
