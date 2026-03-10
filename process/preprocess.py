import argparse
import glob
import os
import sys
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

    return os.path.basename(filepath), tensors_list, file_targets, t_load, t_proc

# combine and preprocess multiple .gz files into one .pt file
def preprocess(max_resonances, n_sets, training=True, workers=None):

    if training:
        pattern = f'*training_nlevel_{max_resonances}_n_{n_sets}*'
        files = sorted(glob.glob(os.path.join('data/raw/training', pattern)))
    else:
        pattern = f'*testing_nlevel_{max_resonances}_n_{n_sets}*'
        files = sorted(glob.glob(os.path.join('data/raw/testing', pattern)))

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

        for name, tensors_list, file_targets, t_load, t_proc in tqdm(
            pool.map(_process_file, files), total=n_files, desc='Files', unit='file'
        ):

            tqdm.write(f'{name}: load={t_load:.1f}s, process={t_proc:.1f}s')

            all_tensors.extend(tensors_list)

            for k in target_keys:
                all_targets[k].append(file_targets[k])

    all_tensors = torch.stack(all_tensors)

    combined_targets = {k: torch.cat(all_targets[k], dim=0) for k in target_keys}

    n_samples = n_sets * n_files

    if training:
        output_path = f'data/preprocessed/training/nlevels_{max_resonances}_n_{n_samples}_training.pt'
    else:
        output_path = f'data/preprocessed/testing/nlevels_{max_resonances}_n_{n_samples}_testing.pt'

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({'tensors': all_tensors, 'targets': combined_targets}, output_path)

    print(f'\nDone. {len(all_tensors)} total samples saved to {output_path}')
    print(f'Tensor shape: {all_tensors.shape}')

if __name__ == '__main__':

    max_resonances = sys.argv[1]
    n_samples = sys.argv[2]
    training = (sys.argv[3] == 'train')

    preprocess(max_resonances, n_samples, training=training, workers=8)