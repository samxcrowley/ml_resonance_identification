import json
import sys
import torch
from torch.utils.data import DataLoader
import process.data as data
from model.detr import DETR_Model, DETR_Loss, HungarianMatcher
from process.header import Header
from process.config import Config

def evaluate(run_dir, test_data_path):

    # load params and model

    with open(f'{run_dir}/params.json', 'r') as f:
        params = json.load(f)

    data.MAX_RESONANCES = params['max_resonances']

    config = Config.from_key(params['config'])
    transform = config.get_transform(inference=True)

    header = Header(params['header'])
    model = DETR_Model(header, params)
    state_dict = torch.load(f'{run_dir}/model.pt', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    dataset = data.ResonanceDataset(test_data_path, 0.0, transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    loss_fn = DETR_Loss()
    matcher = HungarianMatcher(
        cost_class=loss_fn.cost_class,
        cost_energy=loss_fn.cost_energy,
        cost_gamma_total=loss_fn.cost_gamma_total,
        cost_jpi_index=loss_fn.cost_jpi_index
    )

    eval_tolerance = params.get('eval_tolerance', 0.05)

    total_true = 0
    total_tp = 0
    total_fp = 0

    energy_errors = []
    gamma_total_errors = []
    jpi_correct = 0
    jpi_total = 0

    with torch.no_grad():

        for tensor, targets in loader:

            tensor = tensor.to(device, non_blocking=True)
            preds = model(tensor)
            targets = loss_fn.prepare_targets(targets)
            indices = matcher(preds, targets)

            for n in range(len(targets)):

                pred_idx, target_idx = indices[n]
                n_objects = len(targets[n]['energy'])
                total_true += n_objects

                # count all confident predictions for false positive calculation
                confident_all = preds['class'][n].softmax(-1)[:, 1] > 0.5
                n_confident = confident_all.sum().item()

                if len(pred_idx) == 0:
                    total_fp += n_confident
                    continue

                pred_energy = preds['energy'][n][pred_idx].squeeze(-1).to(device)
                target_energy = targets[n]['energy'][target_idx].squeeze(-1).to(device)

                close = (pred_energy - target_energy).abs() < eval_tolerance
                confident_matched = preds['class'][n].softmax(-1)[pred_idx, 1] > 0.5
                tp_mask = close & confident_matched
                tp_mask_cpu = tp_mask.cpu()

                tp = tp_mask.sum().item()
                total_tp += tp
                total_fp += n_confident - tp

                # errors on true positives only
                if tp > 0:
                    energy_errors.append((pred_energy[tp_mask] - target_energy[tp_mask]).abs())

                    pred_gamma = preds['gamma_total'][n][pred_idx][tp_mask].squeeze(-1)
                    target_gamma = targets[n]['gamma_total'][target_idx][tp_mask_cpu].squeeze(-1).to(device)
                    gamma_total_errors.append((pred_gamma - target_gamma).abs())

                    pred_jpi = preds['jpi_index'][n][pred_idx][tp_mask].argmax(dim=-1)
                    target_jpi = targets[n]['jpi_index'][target_idx][tp_mask_cpu].squeeze(-1).long().to(device)
                    jpi_correct += (pred_jpi == target_jpi).sum().item()
                    jpi_total += tp

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / total_true if total_true > 0 else 0.0

    energy_mae = torch.cat(energy_errors).mean().item() if energy_errors else float('nan')
    gamma_total_mae = torch.cat(gamma_total_errors).mean().item() if gamma_total_errors else float('nan')
    jpi_accuracy = jpi_correct / jpi_total if jpi_total > 0 else float('nan')

    results = {
        'precision': precision,
        'recall': recall,
        'energy_mae': energy_mae,
        'gamma_total_mae': gamma_total_mae,
        'jpi_accuracy': jpi_accuracy,
        'total_true': total_true,
        'total_tp': total_tp,
        'total_fp': total_fp,
    }

    return results

def print_results(results):

    print(f'Precision: {results["precision"]:.4f}')
    print(f'Recall: {results["recall"]:.4f}')
    print(f'\nEnergy MAE: {results["energy_mae"]:.6f}')
    print(f'Gamma Total MAE: {results["gamma_total_mae"]:.6f}')
    print(f'J^pi Accuracy: {results["jpi_accuracy"]:.4f}')
    print(f'\nTotal true: {results["total_true"]}')
    print(f'True positives: {results["total_tp"]}')
    print(f'False positives: {results["total_fp"]}')

if __name__ == '__main__':

    run_dir = sys.argv[1]
    test_data_path = sys.argv[2]

    results = evaluate(run_dir, test_data_path)
    print_results(results)
