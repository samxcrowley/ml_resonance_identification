import gc
import glob
import os
import sys
import tempfile
import time
from functools import partial
from multiprocessing import Pool
import torch
from tqdm import tqdm
import process.data as data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

N_WORKERS = 1
TRAIN_SPLIT = 0.9

FLOOR = -7.9
WINDOW_BINS = 3
OUTER_K = 8
MIN_OUTER_HALF = 30

# channel weights for resonance
# max. deviation from local baseline within a small window
def compute_channel_weights(tensors, targets, metadata):

    if isinstance(tensors, (list, tuple)):
        tensors = torch.stack(list(tensors))

    n_pp = metadata.get('n_pp_combos', len(metadata.get('pp_combos', [])))
    if not n_pp:
        n_pp = metadata['n_entrances'] * metadata['n_exits']
    n_angles = metadata['n_angles']

    N = tensors.shape[0]
    max_res = targets['energy'].shape[1]
    out_channel = torch.zeros(N, max_res, n_pp * n_angles, dtype=torch.float32)
    out_combo = torch.zeros(N, max_res, n_pp, dtype=torch.float32)

    energies = targets['energy'].squeeze(-1)
    n_true = targets['class'][:, :, 1].sum(dim=1).long()
    e_mins = targets['e_min']
    e_maxs = targets['e_max']

    log_min = data.GAMMA_LOG_MIN
    log_max = data.GAMMA_LOG_MAX

    for s in range(N):

        tensor = tensors[s]
        E = tensor.shape[0]

        masked = tensor.clone()
        masked[tensor <= FLOOR] = float('nan')

        e_min = float(e_mins[s].item())
        e_max = float(e_maxs[s].item())
        e_range = e_max - e_min
        if e_range <= 0:
            continue

        for r in range(int(n_true[s].item())):

            e_bin = int(energies[s, r].item() * E)
            e_bin = min(E - 1, max(0, e_bin))

            gammas_norm = targets['gamma'][s, r]
            gamma_mask = targets['gamma_mask'][s, r]
            gamma_log = gammas_norm * (log_max - log_min) + log_min
            gamma_linear = (10.0 ** gamma_log) * gamma_mask
            total_width = float(gamma_linear.sum().item())
            if total_width <= 0:
                continue

            width_bins = max(1, int((total_width / e_range) * E))
            inner_half = max(WINDOW_BINS, width_bins)
            inner_lo = max(0, e_bin - inner_half)
            inner_hi = min(E, e_bin + inner_half + 1)

            outer_half = max(inner_half + MIN_OUTER_HALF, OUTER_K * width_bins)
            outer_half = min(outer_half, E // 2)
            outer_lo = max(0, e_bin - outer_half)
            outer_hi = min(E, e_bin + outer_half + 1)

            outer_data = masked[outer_lo:outer_hi, :].clone()
            blank_lo = max(0, inner_lo - outer_lo)
            blank_hi = min(outer_data.shape[0], inner_hi - outer_lo)
            outer_data[blank_lo:blank_hi, :] = float('nan')

            baseline = torch.nanmedian(outer_data, dim=0).values
            baseline = torch.where(
                torch.isnan(baseline),
                torch.full_like(baseline, FLOOR),
                baseline
            )

            window = tensor[inner_lo:inner_hi, :]
            dev = torch.where(window > FLOOR, (window - baseline).abs(), torch.zeros_like(window))
            weight_per_channel = dev.max(dim=0).values.clamp(min=0.0)

            out_channel[s, r, :] = weight_per_channel
            out_combo[s, r, :] = weight_per_channel.view(n_pp, n_angles).max(dim=1).values

    return out_channel, out_combo

def add_channel_weights(tensors, targets, metadata):

    channel_weight, channel_weight_combo = compute_channel_weights(tensors, targets, metadata)

    targets['weight_per_channel'] = channel_weight
    targets['weight_per_combo'] = channel_weight_combo
    
    return targets

def _process_file(filepath):

    t0 = time.time()

    raw_data = data.open_data_file(filepath)

    tensors_list, file_targets, metadata = data.process_json(raw_data)

    add_channel_weights(tensors_list, file_targets, metadata)

    total_time = time.time() - t0

    n_samples = len(tensors_list)

    # save to temp file to avoid large IPC transfers
    tmp = tempfile.NamedTemporaryFile(suffix='.pt', delete=False)
    torch.save({'tensors': tensors_list, 'targets': file_targets, 'metadata': metadata}, tmp.name)
    tmp.close()

    return os.path.basename(filepath), tmp.name, n_samples, total_time

# combine and preprocess multiple .gz files into train and test .pt files
def preprocess(pattern):

    files = sorted(glob.glob(os.path.join(f'data/raw/', f'*{pattern}*')))

    if not files:
        print(f'No files with pattern {pattern}.')
        return

    n_files = len(files)

    print(f'\nFound {n_files} files. Processing with {N_WORKERS} worker(s).\n')

    tmp_results = []

    with Pool(processes=N_WORKERS, maxtasksperchild=1) as pool:

        for name, tmp_path, n_samples, total_time in tqdm(
            pool.imap(_process_file, files),
            total=n_files,
            desc='Processing',
            unit='file'
        ):
            msg = f'{name}: time={total_time:.1f}s'
            msg += f', samples={n_samples}'
            tqdm.write(msg)
            tmp_results.append((tmp_path, n_samples))

    n_total = sum(n for _, n in tmp_results)
    print(f'\nTotal samples: {n_total}')

    first_result = torch.load(tmp_results[0][0], weights_only=False)
    sample_shape = first_result['tensors'][0].shape
    target_shapes = {k: v.shape[1:] for k, v in first_result['targets'].items()}
    target_keys = list(first_result['targets'].keys())
    metadata = first_result['metadata']
    del first_result
    gc.collect()

    all_tensors = torch.zeros(n_total, *sample_shape, dtype=torch.float32)
    all_targets = {k: torch.zeros(n_total, *shape, dtype=torch.float32) for k, shape in target_shapes.items()}

    offset = 0
    for tmp_path, n_samples in tqdm(tmp_results, desc='Merging', unit='file'):

        result = torch.load(tmp_path, weights_only=False)
        os.unlink(tmp_path)

        all_tensors[offset:offset + n_samples] = torch.stack(result['tensors'])

        for k in target_keys:
            all_targets[k][offset:offset + n_samples] = result['targets'][k]

        del result
        offset += n_samples

    gc.collect()

    # shuffle and split into train/test sets
    generator = torch.Generator().manual_seed(22)
    perm = torch.randperm(n_total, generator=generator)
    n_train = int(n_total * TRAIN_SPLIT)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    del perm

    out_dir = 'data/preprocessed'
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, f'{pattern}_train.pt')
    test_path = os.path.join(out_dir, f'{pattern}_test.pt')

    torch.save({
        'tensors': all_tensors[train_idx],
        'targets': {k: all_targets[k][train_idx] for k in target_keys},
        'metadata': metadata,
    }, train_path)

    torch.save({
        'tensors': all_tensors[test_idx],
        'targets': {k: all_targets[k][test_idx] for k in target_keys},
        'metadata': metadata,
    }, test_path)

if __name__ == '__main__':

    # pattern to look for in raw filenames
    pattern = sys.argv[1]
    preprocess(pattern)