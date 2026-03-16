import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from process import data
import process.transforms as transforms
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from model.detr import DETR_Model, DETR_Loss
from process.header import Header

MODELS = {
    'detr': DETR_Model,
}

train_stats = [
    'total_loss',
    'class_loss',
    'energy_loss',
    'gamma_loss',
    'jpi_index_loss'
]

def run_epoch(n_epoch, model, loader, loss_fn, is_eval, optimiser, device):

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

        loss = loss_fn(preds, targets)

        if not is_eval:
            optimiser.zero_grad()
            loss['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

        for stat in loss.keys():
            stats[stat] += loss[stat].item() * tensor.size(0)
        
        n += tensor.size(0)

    for stat in stats.keys():
        stats[stat] = stats[stat] / n

    return stats

def train(params):

    seed = params['seed']

    data_path = params['data_path']

    max_resonances = params['max_resonances']
    data.MAX_RESONANCES = max_resonances

    n_subset = params['n_subset']
    num_workers = params['num_workers']
    batch_size = params['batch_size']
    n_epochs = params['n_epochs']
    lr = params['lr']
    weight_decay = params['weight_decay']
    epoch_n_print = params['epoch_n_print']
    eval_every_n = params.get('eval_every_n', 1)

    header_name = params['header']
    header = Header(filename=header_name)

    max_crop = params.get('max_crop', False)

    model_cls = MODELS[params['model']]
    model = model_cls(header, params)

    loss_fn = model.get_loss_fn()

    transform = transforms.get_augment_transform(
        noise_sigma_log10=params.get('noise_sigma_log10', 0.1),
        amplitude_scale=params.get('amplitude_scale', 0.2)
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')

    base_dataset = data.ResonanceDataset(data_path, max_crop, transform)
    dataset = base_dataset

    # -1 in params['n_subset'] indicates to use the entire dataset
    if n_subset != -1:
        dataset = Subset(base_dataset, np.arange(n_subset))

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, \
                                     [train_size, val_size], \
                                        generator=torch.Generator().manual_seed(seed))

    uncropped_val_dataset = data.ResonanceDataset(
        data_path,
        0.0,
        transforms.get_augment_transform(noise_sigma_log10=0.0, amplitude_scale=0.0)
    )
    if n_subset != -1:
        uncropped_val_dataset = Subset(uncropped_val_dataset, np.arange(n_subset))
    uncropped_val_dataset = Subset(uncropped_val_dataset, val_dataset.indices)
    
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

    resume_from = params.get('resume_from', None)
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model'])
        optimiser.load_state_dict(checkpoint['optimiser'])
        for pg in optimiser.param_groups:
            pg['lr'] = lr
        print(f'Resumed from {resume_from}\n')

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=5)

    results = {
        'epoch': []
    }

    for stat in train_stats:
        results[f'train_{stat}'] = []
        results[f'val_{stat}'] = []

    t_start = datetime.now()
    print(f'Started training at {t_start.strftime("%Y-%m-%d %H:%M:%S")}\n')

    best_val_loss = float('inf')
    best_model_state = None
    best_optimiser_state = None

    for epoch in range(1, n_epochs + 1):

        train_m = run_epoch(epoch, model, train_loader, loss_fn, False, optimiser, device)
        val_m = run_epoch(epoch, model, val_loader, loss_fn, True, optimiser, device)

        scheduler.step(val_m['total_loss'])

        results['epoch'].append(epoch)

        for stat in train_stats:
            results[f'train_{stat}'].append(train_m[stat])
            results[f'val_{stat}'].append(val_m[stat])

        # track best model by val loss
        if val_m['total_loss'] < best_val_loss:
            best_val_loss = val_m['total_loss']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_optimiser_state = optimiser.state_dict()

        if epoch % epoch_n_print == 0:
            print(
                f'Epoch {epoch}/{n_epochs} '
                f'| Train loss {train_m["total_loss"]:.4f} '
                f'| Val loss {val_m["total_loss"]:.4f} '
                f'| Train E loss {train_m["energy_loss"]:.4f} '
                f'| Val E loss {val_m["energy_loss"]:.4f} '
                f'| Train J loss {train_m["jpi_index_loss"]:.4f} '
                f'| Val J loss {val_m["jpi_index_loss"]:.4f} '
                f'| Train G loss {train_m["gamma_loss"]:.4f} '
                f'| Val G loss {val_m["gamma_loss"]:.4f}'
            )

    t_end = datetime.now()
    duration = t_end - t_start
    print(f'Training finished at {t_end.strftime("%Y-%m-%d %H:%M:%S")} (total: {str(duration).split(".")[0]}).')

    # save results
    run_name = params.get('run_name', '')
    run_id = f"{datetime.now().strftime('%m%d_%H%M')}_{params['model']}"
    if run_name:
        run_id += f"_{run_name}"
    run_dir = os.path.join('out', 'runs', run_id)
    os.makedirs(run_dir, exist_ok=True)

    params['t_start'] = t_start.strftime('%Y-%m-%d %H:%M:%S')
    params['t_end'] = t_end.strftime('%Y-%m-%d %H:%M:%S')
    params['duration_s'] = int(duration.total_seconds())
    params['best_val_loss'] = best_val_loss

    with open(os.path.join(run_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=4)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(run_dir, 'train_results.csv'), index=False)

    torch.save({
        'model': best_model_state,
        'optimiser': best_optimiser_state,
    }, os.path.join(run_dir, 'checkpoint.pt'))

    print(f'Results saved to {run_dir}.')

    return run_id