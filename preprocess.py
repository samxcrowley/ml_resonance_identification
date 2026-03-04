import argparse
import glob
import os
import sys
import torch
import data

# combine and preprocess multiple .gz files into one .pt file
def preprocess(max_resonances):

    pattern = f'*nlevel_{max_resonances}*'
    output_path = f'data/preprocessed/nlevels_{max_resonances}.pt'

    files = sorted(glob.glob(os.path.join('data', pattern)))

    if not files:
        print(f'No files with pattern {pattern}')
        return

    print(f'Found {len(files)} files.\n')

    all_tensors = []
    all_targets = {k: [] for k in ['class', 'energy', 'gamma_total', 'n_res']}

    for i, filepath in enumerate(files):

        print(f'[{i+1}/{len(files)}] {os.path.basename(filepath)}', flush=True)

        raw_data = data.open_data_file(filepath)
        tensors_list, targets = data.process_json(raw_data)

        all_tensors.extend(tensors_list)

        all_targets['class'].append(targets['class'])
        all_targets['energy'].append(targets['energy'])
        all_targets['gamma_total'].append(targets['gamma_total'])
        all_targets['n_res'].append(targets['n_res'])

    all_tensors = torch.stack(all_tensors)

    combined_targets = {
        'class': torch.cat(all_targets['class'], dim=0),
        'energy': torch.cat(all_targets['energy'], dim=0),
        'gamma_total': torch.cat(all_targets['gamma_total'], dim=0),
        'n_res': torch.cat(all_targets['n_res'], dim=0),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({'tensors': all_tensors, 'targets': combined_targets}, output_path)

    print(f'\nDone. {len(all_tensors)} total samples saved to {output_path}')
    print(f'Tensor shape: {all_tensors.shape}')

if __name__ == '__main__':

    max_resonances = sys.argv[1]
    preprocess(max_resonances)