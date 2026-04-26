import gc
import glob
import os
import sys
import tempfile
import time
from multiprocessing import Pool
import torch
from tqdm import tqdm
import process.data as data

target_keys = ['class', 'energy', 'gamma', 'gamma_mask', 'jpi_index', 'e_min', 'e_max']

def _process_file(filepath):

    t0 = time.time()
    raw_data = data.open_data_file(filepath)
    t_load = time.time() - t0

    t0 = time.time()
    tensors_list, file_targets, metadata = data.process_json(raw_data)
    t_proc = time.time() - t0

    n_samples = len(tensors_list)

    # save to temp file to avoid large IPC transfers
    tmp = tempfile.NamedTemporaryFile(suffix='.pt', delete=False)
    torch.save({'tensors': tensors_list, 'targets': file_targets, 'metadata': metadata}, tmp.name)
    tmp.close()

    return os.path.basename(filepath), tmp.name, n_samples, t_load, t_proc

# combine and preprocess multiple .gz files into train and test .pt files
def preprocess(pattern, workers=1, train_split=0.9):

    files = sorted(glob.glob(os.path.join(f'data/raw/', f'*{pattern}*')))

    if not files:
        print(f'No files with pattern {pattern}.')
        return

    n_files = len(files)

    if workers is None:
        workers = min(n_files, os.cpu_count() or 1)

    print(f'\nFound {n_files} files. Processing with {workers} worker(s).\n')

    tmp_results = []

    with Pool(processes=workers, maxtasksperchild=1) as pool:

        for name, tmp_path, n_samples, t_load, t_proc in tqdm(
            pool.imap(_process_file, files),
            total=n_files,
            desc='Processing',
            unit='file'
        ):

            tqdm.write(f'{name}: load={t_load:.1f}, process={t_proc:.1f}s, samples={n_samples}')
            tmp_results.append((tmp_path, n_samples))

    n_total = sum(n for _, n in tmp_results)
    print(f'\nTotal samples: {n_total}')

    first_result = torch.load(tmp_results[0][0], weights_only=False)
    sample_shape = first_result['tensors'][0].shape
    target_shapes = {k: v.shape[1:] for k, v in first_result['targets'].items()}
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
    n_train = int(n_total * train_split)
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

    pattern = sys.argv[1]

    preprocess(pattern)