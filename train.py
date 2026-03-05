import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
import data
from config import Config
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from model.detr import DETR_Model
from header import Header

train_stats = ['total_loss', 'class_loss', 'energy_loss', 'gamma_total_loss']

def run_epoch(n_epoch, model, loader, _target, is_eval, optimiser, device):

    if is_eval:
        model.eval()
    else:
        model.train()

    n = 0.0
    stats = {}
    for stat in train_stats:
        stats[stat] = 0.0

    for tensor, targets in loader:

        tensor = tensor.to(device, non_blocking=True)
        
        preds = model(tensor)

        targets = _target.get_targets(targets)

        loss_fn = _target.get_loss_fn()
        loss = loss_fn(preds, targets)

        if not is_eval:
            optimiser.zero_grad()
            loss['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

        for stat in loss.keys():

            # TODO: this is a quick workaround as total_loss is returned as a tensor
            # so that we can call .backward() on it
            if stat == 'total_loss':
                stats[stat] += loss[stat].item() * tensor.size(0)
            else:
                stats[stat] += loss[stat] * tensor.size(0)
        
        n += tensor.size(0)

    for stat in stats.keys():
        stats[stat] = stats[stat] / n

    return stats

def train(params):

    seed = params['seed']

    header_name = params['header']
    header = Header(filename=header_name)
    
    max_resonances = params['max_resonances']
    data.MAX_RESONANCES = max_resonances

    n_subset = params['n_subset']
    num_workers = params['num_workers']
    batch_size = params['batch_size']
    n_epochs = params['n_epochs']
    lr = params['lr']
    weight_decay = params['weight_decay']
    epoch_n_print = params['epoch_n_print']
    
    config = Config.from_key(params['config'])
    model = config.get_model()
    transform = config.get_transform()
    target = config.get_target()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')

    path = f'data/preprocessed/nlevels_{max_resonances}.pt'

    dataset = data.ResonanceDataset(path, transform=transform)

    # -1 in params['n_subset'] indicates to use the entire dataset
    if n_subset != -1:
        dataset = Subset(dataset, np.arange(n_subset))

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, \
                                     [train_size, val_size], \
                                        generator=torch.Generator().manual_seed(seed))

    print(f'Data loaded, maximum resonances is {data.MAX_RESONANCES}.')
    print(f'Training size: {len(train_dataset)}')
    print(f'Validation size: {len(val_dataset)}\n')
    
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              persistent_workers=True,
                              prefetch_factor=4)
    
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=4)

    model.to(device)

    optimiser = model.get_optimiser(lr=lr, weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=5)

    results = {
        'epoch': []
    }

    for stat in train_stats:
        results[f'train_{stat}'] = []
        results[f'val_{stat}'] = []

    results['val_precision'] = []
    results['val_recall'] = []

    print('Starting training...\n')

    for epoch in range(1, n_epochs + 1):

        train_m = run_epoch(epoch, model, train_loader, target, False, optimiser, device)
        val_m = run_epoch(epoch, model, val_loader, target, True, optimiser, device)

        scheduler.step(val_m['total_loss'])

        results['epoch'].append(epoch)

        for stat in train_stats:
            results[f'train_{stat}'].append(train_m[stat])
            results[f'val_{stat}'].append(val_m[stat])

        # evaluate model statistics

        evaluate_m = model.evaluate(loader=val_loader, device=device)
        precision = evaluate_m["precision"]
        recall = evaluate_m["recall"]
        results['val_precision'].append(precision)
        results['val_recall'].append(recall)

        if epoch % epoch_n_print == 0:
            print(
                f'Epoch {epoch} '
                f'| Train loss {train_m["total_loss"]:.4f} '
                f'| Val loss {val_m["total_loss"]:.4f} '
                f'| Precision {precision:.4f} '
                f'| Recall {recall:.4f}\n'
            )

    # save results
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{params['config']}"
    run_dir = os.path.join('out', 'runs', run_id)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=4)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(run_dir, 'results.csv'), index=False)

    print(f'\nResults saved to {run_dir}')

    return run_id