import gzip
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
    'j_loss',
    'pi_loss',
]

def _loader_kwargs(num_workers, persistent_workers):
    kwargs = {
        'num_workers': num_workers,
        'pin_memory': True,
    }
    if num_workers > 0:
        kwargs['persistent_workers'] = persistent_workers
        kwargs['prefetch_factor'] = 4
    return kwargs

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

    grad_clip_norm = getattr(loss_fn, 'grad_clip_norm', 1.0)

    for tensor, targets in loader:

        tensor = tensor.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            preds = model(tensor)
            loss = loss_fn(preds, targets)

        if not is_eval:
            optimiser.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss['total_loss']).backward()
                if grad_clip_norm and grad_clip_norm > 0:
                    scaler.unscale_(optimiser)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimiser)
                scaler.update()
            else:
                loss['total_loss'].backward()
                if grad_clip_norm and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
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
    val_subset = params.get('val_subset', -1)
    num_workers = params['num_workers']
    batch_size = params['batch_size']
    n_epochs = params['n_epochs']
    lr = params['lr']
    weight_decay = params['weight_decay']
    epoch_n_print = params['epoch_n_print']
    eval_every_n = params.get('eval_every_n', epoch_n_print)

    header_name = params['header']
    header = Header(filename=header_name)

    crop_params = {
        'crop_energy': params['crop_energy'],
        'min_angles': params['min_angles'],
        'min_pp_combos': params['min_pp_combos'],
        'max_pp_combos': params.get('max_pp_combos', 9),
        'elastic_max_pp_combos': params.get('elastic_max_pp_combos', None),
        'contiguous_angle_crop_p': params.get('contiguous_angle_crop_p', 0.0),
        'shared_energy_crop_p': params.get('shared_energy_crop_p', 0.0),
        'inelastic_dropout_p': params.get('inelastic_dropout_p', 0.0),
        'min_kept_prom': params.get('min_kept_prom', 0.0),
    }

    curriculum_epochs = params.get('curriculum_epochs', 0)
    curriculum_update_every = max(1, params.get('curriculum_update_every', 1))

    if curriculum_epochs > 0:
        crop_energy_max = crop_params['crop_energy']
        min_angles_final = crop_params['min_angles']
        min_pp_combos_final = crop_params['min_pp_combos']
        inelastic_dropout_p_max = crop_params['inelastic_dropout_p']
        n_pp = params['n_entrances'] * params['n_exits']
        n_angles = params['n_angles']

    model_cls = MODELS[params['model']]
    model = model_cls(header, params)

    loss_fn = model.get_loss_fn()
    loss_fn.grad_clip_norm = params.get('grad_clip_norm', 1.0)

    transform = transforms.get_augment_transform(
        noise_sigma_log10=params.get('noise_sigma_log10', 0.1),
        amplitude_scale=params.get('amplitude_scale', 0.2),
        gaussian_blur_sigma=params.get('gaussian_blur_sigma', 0.0)
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')

    channel_filter = params.get('channel_filter', None)
    base_dataset = data.ResonanceDataset(data_path, crop_params, transform, crop_fn=transforms._crop, channel_filter=channel_filter)
    dataset = base_dataset
    if n_subset != -1:
        dataset = Subset(base_dataset, np.arange(n_subset))

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, \
                                     [train_size, val_size], \
                                        generator=torch.Generator().manual_seed(seed))

    if val_subset != -1 and val_subset < len(val_dataset):
        val_dataset = Subset(val_dataset, np.arange(val_subset))

    print(f'Training size: {len(train_dataset)}')
    print(f'Validation size: {len(val_dataset)}\n')

    # Persistent workers cache dataset state. With curriculum_update_every > 1,
    # use a staircase curriculum and recreate workers only at step boundaries.
    use_persistent_train = (
        (curriculum_epochs == 0 or curriculum_update_every > 1) and
        num_workers > 0
    )
    use_persistent_val = (curriculum_epochs == 0) and (num_workers > 0)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              **_loader_kwargs(num_workers, use_persistent_train))

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            **_loader_kwargs(num_workers, use_persistent_val))

    torch.backends.cudnn.benchmark = True

    model.to(device)

    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        model = torch.compile(model, mode=params.get('compile_mode', 'default'))

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
    best_after_epoch = params.get('best_after_epoch', max(1, curriculum_epochs))
    early_stop_patience = params.get('early_stop_patience', 0)
    early_stop_min_delta = params.get('early_stop_min_delta', 0.0)
    epochs_without_improvement = 0
    snapshot_every = params.get('snapshot_every', 0)
    snapshot_after_epoch = params.get('snapshot_after_epoch', best_after_epoch)
    max_snapshots = params.get('max_snapshots', 0)
    snapshot_states = []

    try:
        for epoch in range(1, n_epochs + 1):

            if curriculum_epochs > 0:

                should_update_curriculum = (
                    epoch == 1 or
                    epoch == curriculum_epochs or
                    curriculum_update_every == 1 or
                    ((epoch - 1) % curriculum_update_every == 0 and epoch < curriculum_epochs)
                )

                if should_update_curriculum:
                    progress = min(1.0, epoch / curriculum_epochs)
                    base_dataset.crop_params['crop_energy'] = crop_energy_max * progress
                    base_dataset.crop_params['min_angles'] = round(n_angles - (n_angles - min_angles_final) * progress)
                    base_dataset.crop_params['min_pp_combos'] = round(n_pp - (n_pp - min_pp_combos_final) * progress)
                    base_dataset.crop_params['inelastic_dropout_p'] = inelastic_dropout_p_max * progress

                should_rebuild_train_loader = (
                    should_update_curriculum and
                    num_workers > 0 and
                    (
                        (curriculum_update_every > 1 and epoch > 1) or
                        (curriculum_update_every == 1 and epoch == curriculum_epochs)
                    )
                )
                if should_rebuild_train_loader:
                    train_loader = DataLoader(train_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              **_loader_kwargs(num_workers, True))

            train_m = run_epoch(epoch, model, train_loader, loss_fn, False, optimiser, device, scaler)

            do_eval = epoch % eval_every_n == 0
            val_m = run_epoch(epoch, model, val_loader, loss_fn, True, optimiser, device) if do_eval else None

            if scheduler_type == 'plateau':
                if val_m is not None:
                    scheduler.step(val_m['total_loss'])
            else:
                scheduler.step()

            results['epoch'].append(epoch)

            for stat in train_stats:
                results[f'train_{stat}'].append(train_m[stat])
                results[f'val_{stat}'].append(val_m[stat] if val_m is not None else None)

            if epoch >= best_after_epoch and val_m is not None:
                improved = val_m['total_loss'] < (best_val_loss - early_stop_min_delta)
                if improved:
                    best_val_loss = val_m['total_loss']
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_optimiser_state = optimiser.state_dict()
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += eval_every_n

                if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
                    print(f'Early stopping at epoch {epoch}: '
                          f'no val improvement for {epochs_without_improvement} epoch(s).')
                    break

            if (
                snapshot_every > 0 and
                epoch >= snapshot_after_epoch and
                epoch % snapshot_every == 0
            ):
                snapshot_states.append((
                    epoch,
                    {k: v.cpu().clone() for k, v in model.state_dict().items()}
                ))
                if max_snapshots > 0 and len(snapshot_states) > max_snapshots:
                    snapshot_states = snapshot_states[-max_snapshots:]

            if epoch % epoch_n_print == 0:
                def _fmt(m):
                    return (f'tot={m["total_loss"]:.4f} '
                            f'cls={m["class_loss"]:.4f} '
                            f'eng={m["energy_loss"]:.4f} '
                            f'gam={m["gamma_loss"]:.4f} '
                            f'j={m["j_loss"]:.4f} '
                            f'pi={m["pi_loss"]:.4f}')
                print(f'Epoch {epoch:03d}/{n_epochs}')
                print(f'  TRAIN  {_fmt(train_m)}')
                if val_m is not None:
                    print(f'  VAL    {_fmt(val_m)}')

    except KeyboardInterrupt:
        print(f'\nTraining interrupted at epoch {epoch}. Saving results...')

    t_end = datetime.now()
    duration = t_end - t_start
    print(f'Training finished at {t_end.strftime("%Y-%m-%d %H:%M:%S")} (total: {str(duration).split(".")[0]}).')

    if best_model_state is None:
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_optimiser_state = optimiser.state_dict()

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

    for snapshot_epoch, snapshot_state in snapshot_states:
        torch.save({
            'model': snapshot_state,
            'epoch': snapshot_epoch,
        }, os.path.join(run_dir, f'checkpoint_epoch{snapshot_epoch:04d}.pt'))

    print(f'Results saved to {run_dir}.')

    return run_id
