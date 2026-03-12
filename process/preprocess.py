import argparse
import glob
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
import torch
from tqdm import tqdm
import process.data as data

target_keys = ['class', 'energy', 'gamma_total', 'jpi_index', 'n_res']

def _process_file(filepath):

    t0 = time.time()
    raw_data = data.open_data_file(filepath)
    t_load = time.time() - t0

    t0 = time.time()
    tensors_list, file_targets = data.process_json(raw_data)
    t_proc = time.time() - t0

    # save to temp file to avoid large IPC transfers
    tmp = tempfile.NamedTemporaryFile(suffix='.pt', delete=False)
    torch.save({'tensors': tensors_list, 'targets': file_targets}, tmp.name)
    tmp.close()

    return os.path.basename(filepath), tmp.name, t_load, t_proc

# combine and preprocess multiple .gz files into train and test .pt files
def preprocess(max_resonances, workers=4, train_split=0.8, seed=22):

    pattern = f'*nlevel_{max_resonances}*'
    files = sorted(glob.glob(os.path.join(f'data/raw/', pattern)))

    if not files:
        print(f'No files with pattern {pattern}')
        return

    n_files = len(files)

    if workers is None:
        workers = min(n_files, os.cpu_count() or 1)

    print(f'\nFound {n_files} files. Processing with {workers} workers.\n')

    all_tensors = []
    all_targets = {k: [] for k in target_keys}

    with ProcessPoolExecutor(max_workers=workers) as pool:

        for name, tmp_path, t_load, t_proc in tqdm(
            pool.map(_process_file, files), total=n_files, desc='Files', unit='file'
        ):

            tqdm.write(f'{name}: load={t_load:.1f}s, process={t_proc:.1f}s')

            result = torch.load(tmp_path, weights_only=False)
            os.unlink(tmp_path)

            all_tensors.extend(result['tensors'])

            for k in target_keys:
                all_targets[k].append(result['targets'][k])

    all_tensors = torch.stack(all_tensors)

    combined_targets = {k: torch.cat(all_targets[k], dim=0) for k in target_keys}

    # shuffle and split into train/test sets
    n_total = len(all_tensors)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=generator)

    n_train = int(n_total * train_split)

    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    train_data = {
        'tensors': all_tensors[train_idx],
        'targets': {k: combined_targets[k][train_idx] for k in target_keys},
    }
    test_data = {
        'tensors': all_tensors[test_idx],
        'targets': {k: combined_targets[k][test_idx] for k in target_keys},
    }

    out_dir = 'data/preprocessed'
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, f'nlevels_{max_resonances}_train.pt')
    test_path = os.path.join(out_dir, f'nlevels_{max_resonances}_test.pt')

    torch.save(train_data, train_path)
    torch.save(test_data, test_path)

if __name__ == '__main__':

    max_resonances = sys.argv[1]

    preprocess(max_resonances, workers=4)