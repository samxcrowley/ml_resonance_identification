import json
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
import process.data as data
from model.detr import DETR_Model, DETR_Loss, HungarianMatcher
from model.set_detr import SetDETR_Model
from process.header import Header
import process.transforms as transforms

MODELS = {
    'detr': DETR_Model,
    'set_detr': SetDETR_Model,
}

def evaluate(run_dir, crop_strength=0.0, confidence_threshold=0.5, test_data_path='data/preprocessed/nlevels_20_test.pt'):

    with open(f'{run_dir}/params.json', 'r') as f:
        params = json.load(f)

    data.MAX_RESONANCES = params['max_resonances']

    transform = transforms.get_augment_transform(noise_sigma_log10=0.0, amplitude_scale=0.0)

    header = Header(params['header'])
    model_cls = MODELS[params['model']]
    model = model_cls(header, params)
    checkpoint = torch.load(f'{run_dir}/checkpoint.pt', weights_only=False)
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    crop_params = {}
    if crop_strength > 0.0:
        crop_params = {
            'crop_energy': crop_strength,
            'crop_angle': True,
            'crop_channel': True
        }

    dataset = data.ResonanceDataset(test_data_path, crop_params, transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    loss_fn = model.get_loss_fn()
    matcher = HungarianMatcher()

    total_true = 0
    total_tp = 0
    total_fp = 0
    total_slots = 0

    energy_errors = []
    gamma_errors = []
    jpi_correct = 0
    jpi_total = 0

    true_counts = []
    pred_counts = []

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

                confidences = preds['class'][n].softmax(-1)[:, 1]
                n_confident = (confidences > confidence_threshold).sum().item()
                n_queries = len(confidences)
                total_slots += n_queries

                true_counts.append(n_objects)
                pred_counts.append(n_confident)

                if len(pred_idx) == 0:
                    total_fp += n_confident
                    continue

                pred_gamma = preds['gamma'][n][pred_idx]
                target_gamma = targets[n]['gamma'][target_idx].to(device)
                gamma_mask = targets[n]['gamma_mask'][target_idx].to(device)

                # tolerance is T * max partial width (width is log and normalised)
                T = 0.05
                max_partial_gamma = target_gamma.max(dim=-1).values
                eval_tolerance = T * max_partial_gamma

                pred_energy = preds['energy'][n][pred_idx].squeeze(-1)
                target_energy = targets[n]['energy'][target_idx].squeeze(-1).to(device)
                close = (pred_energy - target_energy).abs() < eval_tolerance

                pred_jpi = preds['jpi_index'][n][pred_idx].argmax(dim=-1)
                target_jpi = targets[n]['jpi_index'][target_idx].squeeze(-1).long().to(device)
                jpi_match = (pred_jpi == target_jpi)

                # correct prediction is defined as:
                # jpi matched exactly
                # energy within tolerance (see above)
                # confidence greater than threshold
                tp_mask = close & jpi_match & (confidences[pred_idx] > confidence_threshold)
                tp = tp_mask.sum().item()
                total_tp += tp
                total_fp += n_confident - tp

                matched_energy_err = (pred_energy - target_energy).abs()[tp_mask]
                if matched_energy_err.numel() > 0:
                    energy_errors.append(matched_energy_err.cpu())

                nan_mask = target_gamma.isnan()
                if nan_mask.any():
                    target_gamma = target_gamma.clone()
                    target_gamma[nan_mask] = 0.0
                    gamma_mask = gamma_mask.clone()
                    gamma_mask[nan_mask] = 0.0
                gamma_diff = (pred_gamma - target_gamma).abs()[tp_mask]
                gamma_m = gamma_mask[tp_mask].bool()
                gamma_diff = gamma_diff[gamma_m]
                if gamma_diff.numel() > 0:
                    gamma_errors.append(gamma_diff.cpu())

                # jpi accuracy will always be 1.0 for now, as both of these are equal
                jpi_correct += jpi_match[tp_mask].sum().item()
                jpi_total += tp

    total_fn = total_true - total_tp
    total_tn = total_slots - total_tp - total_fp - total_fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / total_true if total_true > 0 else 0.0
    energy_mae = torch.cat(energy_errors).mean().item() if energy_errors else float('nan')
    gamma_mae = torch.cat(gamma_errors).mean().item() if gamma_errors else float('nan')
    jpi_accuracy = jpi_correct / jpi_total if jpi_total > 0 else float('nan')

    results = {
        'confidence_threshold': confidence_threshold,
        'precision': precision,
        'recall': recall,
        'energy_mae': energy_mae,
        'gamma_mae': gamma_mae,
        'jpi_accuracy': jpi_accuracy,
        'total_true': total_true,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'total_tn': total_tn,
    }

    raw = {
        'energy_errors': torch.cat(energy_errors).numpy() if energy_errors else np.array([]),
        'gamma_errors': torch.cat(gamma_errors).numpy() if gamma_errors else np.array([]),
        'true_counts': np.array(true_counts),
        'pred_counts': np.array(pred_counts),
    }
    
    # save results
    with open(os.path.join(run_dir, f'test_results_threshold{confidence_threshold}.csv'), 'w') as f:
        json.dump(results, f, indent=4)

    return results, raw

def print_results(results):

    print(f'Confidence threshold: {results["confidence_threshold"]:.2f}')
    print(f'Precision: {results["precision"]:.4f}')
    print(f'Recall: {results["recall"]:.4f}')
    print(f'\nEnergy MAE: {results["energy_mae"]:.6f}')
    print(f'Gamma MAE: {results["gamma_mae"]:.6f}')
    print(f'J^pi Accuracy: {results["jpi_accuracy"]:.4f}')
    print(f'\nTotal true: {results["total_true"]}')
    print(f'True positives: {results["total_tp"]}')
    print(f'False positives: {results["total_fp"]}')
    print(f'False negatives: {results["total_fn"]}')
    print(f'True negatives: {results["total_tn"]}')

def plot_results(results, raw, run_dir):

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # confusion matrix
    ax = axes[0]
    ax.axis('off')
    confusion_data = [
        [f'TP = {results["total_tp"]}', f'FN = {results["total_fn"]}'],
        [f'FP = {results["total_fp"]}', f'TN = {results["total_tn"]}'],
    ]
    table = ax.table(cellText=confusion_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.0)
    ax.set_title('Confusion Table')

    # n_resonances confusion matrix
    ax = axes[1]
    max_count = max(raw['true_counts'].max(), raw['pred_counts'].max()) + 1
    cm = np.zeros((max_count, max_count), dtype=int)
    for t, p in zip(raw['true_counts'], raw['pred_counts']):
        cm[t, p] += 1
    im = ax.imshow(cm, origin='lower', cmap='Blues')
    fig.colorbar(im, ax=ax)

    # display numbers (doesn't look great)
    # for i in range(max_count):
    #     for j in range(max_count):
    #         if cm[i, j] > 0:
    #             ax.text(j, i, str(cm[i, j]), ha='center', va='center',
    #                     color='white' if cm[i, j] > cm.max() / 2 else 'black')

    ax.set_xlabel('Predicted Count')
    ax.set_ylabel('True Count')
    ax.set_title('Resonance Count Confusion Matrix')
    ax.set_xticks(range(max_count))
    ax.set_yticks(range(max_count))

    # metrics table
    ax = axes[2]
    ax.axis('off')
    table_data = [
        ['Precision', f'{results["precision"]:.4f}'],
        ['Recall', f'{results["recall"]:.4f}'],
        ['Energy MAE', f'{results["energy_mae"]:.6f}'],
        ['Gamma MAE', f'{results["gamma_mae"]:.6f}'],
        ['J^pi Accuracy', f'{results["jpi_accuracy"]:.4f}']
    ]
    table = ax.table(cellText=table_data, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.4)
    ax.set_title('Metrics')

    fig.suptitle(f'Test Results (confidence threshold {results["confidence_threshold"]:.2f})', fontsize=14)
    plt.tight_layout()
    path = f'{run_dir}/test_results_threshold{confidence_threshold}.png'
    plt.savefig(path, dpi=150)
    plt.close()

if __name__ == '__main__':

    run_dir = sys.argv[1]
    crop_strength = float(sys.argv[2])
    confidence_threshold = float(sys.argv[3])

    results, raw = evaluate(run_dir, crop_strength, confidence_threshold)

    print_results(results)

    plot_results(results, raw, run_dir)