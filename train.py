import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from model import encoder
from model import torch_encoder
import load_data

def train_epoch(model, loader, loss_fn, optimiser, device):

    model.train()

    running_stats = {
        'loss': 0.0, 'count': 0.0
    }

    for tensor, targets in loader:

        optimiser.zero_grad()

        tensor = tensor.to(device)
        targets = targets.to(device)

        pred = model(tensor)[:, 0]
        target = targets[:, 1] # normalised energy

        loss = loss_fn(pred, target)

        loss.backward()
        optimiser.step()

        with torch.no_grad():
            running_stats['loss'] += loss.item() * tensor.size(0)
            running_stats['count'] += tensor.size(0)

    n = running_stats['count']

    metrics = {
        'loss': running_stats['loss'] / n
    }

    return metrics

def eval_epoch(model, loader, loss_fn, device):

    model.eval()

    running_stats = {
        'loss': 0.0, 'count': 0.0
    }

    with torch.no_grad():

        for tensor, targets in loader:

            tensor = tensor.to(device)
            targets = targets.to(device)

            pred = model(tensor)[:, 0]
            target = targets[:, 1] # normalised energy

            loss = loss_fn(pred, target)

            running_stats['loss'] += loss.item() * tensor.size(0)
            running_stats['count'] += tensor.size(0)

    n = running_stats['count']

    metrics = {
        'loss': running_stats['loss'] / n
    }

    return metrics

def train(params, transform, model):

    seed = params['seed']
    data_path = params['data_path']
    data_is_compressed = params['data_is_compressed']
    num_workers = params['num_workers']
    batch_size = params['batch_size']
    n_epochs = params['n_epochs']
    lr = params['lr']
    weight_decay = params['weight_decay']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dataset = load_data.EnergyLevelDataset(data_path, transform=transform, compressed=data_is_compressed)

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
                            shuffle=True,
                            num_workers=num_workers)

    model.to(device)

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    loss_fn = nn.MSELoss()

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=5)

    results = {
        'epoch': [],
        'train_loss': [],
        'val_loss': []
    }

    for epoch in range(1, n_epochs + 1):

        train_m = train_epoch(model, train_loader, loss_fn, optimiser, device)
        val_m = eval_epoch(model, val_loader, loss_fn, device)

        scheduler.step(val_m['loss'])

        results['epoch'].append(epoch)
        results['train_loss'].append(train_m['loss'])
        results['val_loss'].append(val_m['loss'])

        if epoch % 1 == 0:
            print(
                f'Epoch {epoch} '
                f'| train loss {train_m["loss"]:.4f} '
                f'| val loss {val_m["loss"]:.4f}'
            )