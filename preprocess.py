import argparse
import glob
import os
import sys
import torch
import data

target_keys = ['class', 'energy', 'gamma_total', 'jpi_index', 'n_res']

# combine and preprocess multiple .gz files into one .pt file
def preprocess(max_resonances):

    pattern = f'*nlevel_{max_resonances}*'
    output_path = f'data/preprocessed/nlevels_{max_resonances}.pt'

    files = sorted(glob.glob(os.path.join('data/raw', pattern)))

    if not files:
        print(f'No files with pattern {pattern}')
        return

    print(f'Found {len(files)} files.\n')

    all_tensors = []
    all_targets = {k: [] for k in target_keys}

    for i, filepath in enumerate(files):

        print(f'[{i+1}/{len(files)}] {os.path.basename(filepath)}', flush=True)

        raw_data = data.open_data_file(filepath)
        tensors_list, file_targets = data.process_json(raw_data)

        all_tensors.extend(tensors_list)

        for k in target_keys:
            all_targets[k].append(file_targets[k])

    all_tensors = torch.stack(all_tensors)

    combined_targets = {k: torch.cat(all_targets[k], dim=0) for k in target_keys}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({'tensors': all_tensors, 'targets': combined_targets}, output_path)

    print(f'\nDone. {len(all_tensors)} total samples saved to {output_path}')
    print(f'Tensor shape: {all_tensors.shape}')

if __name__ == '__main__':

    max_resonances = sys.argv[1]
    preprocess(max_resonances)
