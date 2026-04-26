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
    'detr': DETR_Model
}

train_stats = [
    'total_loss',
    'class_loss',
    'energy_loss',
    'gamma_loss',
    'j_loss',
    'pi_loss',
]

def run_epoch(n_epoch, model, loader, loss_fn, is_eval, optimiser, device, scaler=None):

    if is_eval:
        model.eval()
    else:
        model.train()

    use_amp = scaler is not None

    n = 0.0
    stats = {}
    for stat in train_stats:
        stats[stat] = 0.0

    for tensor, targets in loader:

        tensor = tensor.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            preds = model(tensor)
            loss = loss_fn(preds, targets)

        if not is_eval:
            optimiser.zero_grad()
            if use_amp:
                scaler.scale(loss['total_loss']).backward()
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimiser)
                scaler.update()
            else:
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

    crop_params = {
        'crop_energy': params['crop_energy'],
        'min_angles': params['min_angles'],
        'min_pp_combos': params['min_pp_combos'],
        'use_info_weight': params['use_info_weight'],
    }

    curriculum_epochs = params.get('curriculum_epochs', 0)

    if curriculum_epochs > 0:
        crop_energy_max = crop_params['crop_energy']
        min_angles_final = crop_params['min_angles']
        min_pp_combos_final = crop_params['min_pp_combos']
        n_pp = params['n_entrances'] * params['n_exits']
        n_angles = params['n_angles']

    model_cls = MODELS[params['model']]
    model = model_cls(header, params)

    loss_fn = model.get_loss_fn()

    transform = transforms.get_augment_transform(
        noise_sigma_log10=params.get('noise_sigma_log10', 0.1),
        amplitude_scale=params.get('amplitude_scale', 0.2),
        gaussian_blur_sigma=params.get('gaussian_blur_sigma', 0.0)
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')

    channel_filter = params.get('channel_filter', None)
    base_dataset = data.ResonanceDataset(data_path, crop_params, transform, crop_fn=transforms._crop,
                                         channel_filter=channel_filter)
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
        tensors=base_dataset.tensors,
        targets=base_dataset.targets,
        metadata=base_dataset.metadata)
    if n_subset != -1:
        uncropped_val_dataset = Subset(uncropped_val_dataset, np.arange(n_subset))
    uncropped_val_dataset = Subset(uncropped_val_dataset, val_dataset.indices)

    print(f'Training size: {len(train_dataset)}')
    print(f'Validation size: {len(uncropped_val_dataset)}\n')

    # persistent workers cache dataset state, disable when curriculum
    # updates crop_params between epochs
    use_persistent_train = (curriculum_epochs == 0) and (num_workers > 0)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              persistent_workers=use_persistent_train,
                              prefetch_factor=4)

    val_loader = DataLoader(uncropped_val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            persistent_workers=num_workers > 0,
                            prefetch_factor=4)

    model.to(device)

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    optimiser = model.get_optimiser(lr=lr, weight_decay=weight_decay)

    resume_from = params.get('resume_from', None)
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimiser.load_state_dict(checkpoint['optimiser'])
        for pg in optimiser.param_groups:
            pg['lr'] = lr
        print(f'Resumed from {resume_from}\n')

    warmup_epochs = params.get('warmup_epochs', 5)
    scheduler_type = params.get('scheduler', 'cosine')

    if scheduler_type == 'cosine':
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, n_epochs - warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)
    elif scheduler_type == 'step':
        lr_drop_epoch = params.get('lr_drop_epoch', 400)
        lr_drop_factor = params.get('lr_drop_factor', 0.1)
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            if epoch >= lr_drop_epoch:
                return lr_drop_factor
            return 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)
    else:
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

    try:
        for epoch in range(1, n_epochs + 1):

            if curriculum_epochs > 0:
                progress = min(1.0, epoch / curriculum_epochs)
                base_dataset.crop_params['crop_energy'] = crop_energy_max * progress
                base_dataset.crop_params['min_angles'] = round(n_angles - (n_angles - min_angles_final) * progress)
                base_dataset.crop_params['min_pp_combos'] = round(n_pp - (n_pp - min_pp_combos_final) * progress)

            train_m = run_epoch(epoch, model, train_loader, loss_fn, False, optimiser, device, scaler)
            val_m = run_epoch(epoch, model, val_loader, loss_fn, True, optimiser, device)

            if scheduler_type == 'plateau':
                scheduler.step(val_m['total_loss'])
            else:
                scheduler.step()

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
                    f'Epoch {epoch:03d}/{n_epochs} '
                    f'|| T {train_m["total_loss"]:.4f} '
                    f'| V {val_m["total_loss"]:.4f} '
                    f'|| T_C {train_m["class_loss"]:.4f} '
                    f'| V_C {val_m["class_loss"]:.4f} '
                    f'|| T_E {train_m["energy_loss"]:.4f} '
                    f'| V_E {val_m["energy_loss"]:.4f} '
                    f'|| T_G {train_m["gamma_loss"]:.4f} '
                    f'| V_G {val_m["gamma_loss"]:.4f} '
                    f'|| T_J {train_m["j_loss"]:.4f} '
                    f'| V_J {val_m["j_loss"]:.4f} '
                    f'|| T_P {train_m["pi_loss"]:.4f} '
                    f'| V_P {val_m["pi_loss"]:.4f}'
                )

    except KeyboardInterrupt:
        print(f'\nTraining interrupted at epoch {epoch}. Saving results...')

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