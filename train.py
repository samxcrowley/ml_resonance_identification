import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import data
from config import Config
import pandas as pd
from model.detr import DETR_Model

def run_batch(tensor, targets, _target, model, device):

    tensor = tensor.to(device)

    preds = model(tensor)

    targets = _target.get_targets(targets)
    
    if type(targets) is not dict:
        targets = targets.to(device)

    loss_fn = _target.get_loss_fn()
    loss = loss_fn(preds, targets)

    return loss

def train_epoch(model, loader, _target, optimiser, device):

    model.train()

    running_stats = {
        'total_loss': 0.0,
        'class_loss': 0.0,
        'energy_loss': 0.0,
        'count': 0.0
    }

    for tensor, targets in loader:

        optimiser.zero_grad()

        loss = run_batch(tensor, targets, _target, model, device)

        if type(loss) is dict:
            loss['total'].backward()
        else:
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimiser.step()
        
        with torch.no_grad():

            if type(loss) is dict:
                running_stats['total_loss'] += loss['total'] * tensor.size(0)
                running_stats['class_loss'] += loss['class'] * tensor.size(0)
                running_stats['energy_loss'] += loss['energy'] * tensor.size(0)

            else:
                running_stats['total_loss'] += loss * tensor.size(0)

            running_stats['count'] += tensor.size(0)

    n = running_stats['count']

    metrics = {
        'total_loss': running_stats['total_loss'] / n,
        'class_loss': running_stats['class_loss'] / n,
        'energy_loss': running_stats['energy_loss'] / n
    }

    return metrics

def eval_epoch(model, loader, _target, device):

    model.eval()

    running_stats = {
        'total_loss': 0.0,
        'class_loss': 0.0,
        'energy_loss': 0.0,
        'count': 0.0
    }

    with torch.no_grad():

        for tensor, targets in loader:

            loss = run_batch(tensor, targets, _target, model, device)

            if type(loss) is dict:
                running_stats['total_loss'] += loss['total'] * tensor.size(0)
                running_stats['class_loss'] += loss['class'] * tensor.size(0)
                running_stats['energy_loss'] += loss['energy'] * tensor.size(0)

            else:
                running_stats['total_loss'] += loss * tensor.size(0)

            running_stats['count'] += tensor.size(0)

    n = running_stats['count']

    metrics = {
        'total_loss': running_stats['total_loss'] / n,
        'class_loss': running_stats['class_loss'] / n,
        'energy_loss': running_stats['energy_loss'] / n
    }

    return metrics

def train(params):

    seed = params['seed']
    data_filename = params['data_filename']
    num_workers = params['num_workers']
    batch_size = params['batch_size']
    n_epochs = params['n_epochs']
    lr = params['lr']
    weight_decay = params['weight_decay']
    epoch_n_print = params['epoch_n_print']
    do_evaluate = params['evaluate']
    
    config = Config.from_key(params['config'])
    model = config.get_model()
    transform = config.get_transform()
    target = config.get_target()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dataset = data.ResonanceDataset(data_filename, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, \
                                     [train_size, val_size], \
                                        generator=torch.Generator().manual_seed(seed))

    print(f'Training size: {len(train_dataset)}')
    print(f'Validation size: {len(val_dataset)}')
    
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)

    model.to(device)

    optimiser = model.get_optimiser(lr=lr, weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=5)

    results = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'train_class_loss': [],
        'train_energy_loss': [],
        'val_class_loss': [],
        'val_energy_loss': []
    }

    if do_evaluate:
        results['val_precision'] = []
        results['val_recall'] = []

    for epoch in range(1, n_epochs + 1):

        train_m = train_epoch(model, train_loader, target, optimiser, device)
        val_m = eval_epoch(model, val_loader, target, device)

        scheduler.step(val_m['total_loss'])

        results['epoch'].append(epoch)
        results['train_loss'].append(train_m['total_loss'])
        results['val_loss'].append(val_m['total_loss'])

        results['train_class_loss'].append(train_m['class_loss'])
        results['val_class_loss'].append(val_m['class_loss'])

        results['train_energy_loss'].append(train_m['energy_loss'])
        results['val_energy_loss'].append(val_m['energy_loss'])

        if epoch % epoch_n_print == 0:

            print(
                f'Epoch {epoch} '
                f'| Train loss {train_m["total_loss"]:.4f} '
                f'| Val loss {val_m["total_loss"]:.4f}'
            )

        # evaluate model statistics

        if do_evaluate:

            evaluate_m = DETR_Model.evaluate(model=model, loader=val_loader, device=device)
            precision = evaluate_m["precision"]
            recall = evaluate_m["recall"]

            results['val_precision'].append(precision)
            results['val_recall'].append(recall)

            if epoch % epoch_n_print == 0:

                print(
                    f'\nPrecision {precision:.4f} '
                    f'| Recall {recall:.4f}\n'
                )

    df = pd.DataFrame(results)
    df.to_csv(f'out/results/{params["config"]}.csv', index=False)