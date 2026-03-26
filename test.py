import json
import os
import sys
import argparse
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

def evaluate(
    run_dir,
    do_crop,
    confidence_threshold,
    width_tolerance,
    test_data_path='data/preprocessed/nlevel_20_test.pt'):

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
    if do_crop:
        crop_params = {
            'crop_energy': 0.5,
            'crop_angle': True,
            'crop_channel': True,
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
    all_confidences = []
    all_is_matched = []

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

                all_confidences.append(confidences.cpu())
                matched_flags = torch.zeros(n_queries, dtype=torch.bool)

                if len(pred_idx) == 0:
                    total_fp += n_confident
                    all_is_matched.append(matched_flags)
                    continue

                pred_gamma = preds['gamma'][n][pred_idx]
                target_gamma = targets[n]['gamma'][target_idx].to(device)
                gamma_mask = targets[n]['gamma_mask'][target_idx].to(device)

                max_partial_gamma = target_gamma.max(dim=-1).values
                eval_tolerance = width_tolerance * max_partial_gamma

                pred_energy = preds['energy'][n][pred_idx].squeeze(-1)
                target_energy = targets[n]['energy'][target_idx].squeeze(-1).to(device)
                close = (pred_energy - target_energy).abs() < eval_tolerance

                pred_jpi = preds['jpi_index'][n][pred_idx].argmax(dim=-1)
                target_jpi = targets[n]['jpi_index'][target_idx].squeeze(-1).long().to(device)
                jpi_match = (pred_jpi == target_jpi)

                # detection TP: energy within tolerance + confident
                # jpi accuracy is measured separately among TPs
                tp_mask = close & (confidences[pred_idx] > confidence_threshold)
                tp = tp_mask.sum().item()
                total_tp += tp
                total_fp += n_confident - tp

                matched_flags[pred_idx[tp_mask.cpu()]] = True
                all_is_matched.append(matched_flags)

                # jpi accuracy among detection TPs
                jpi_correct += jpi_match[tp_mask].sum().item()
                jpi_total += tp

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

    total_fn = total_true - total_tp
    total_tn = total_slots - total_tp - total_fp - total_fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / total_true if total_true > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    energy_mae = torch.cat(energy_errors).mean().item() if energy_errors else float('nan')
    gamma_mae = torch.cat(gamma_errors).mean().item() if gamma_errors else float('nan')
    jpi_accuracy = jpi_correct / jpi_total if jpi_total > 0 else float('nan')

    results = {
        'confidence_threshold': confidence_threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
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
        'all_confidences': torch.cat(all_confidences).numpy(),
        'all_is_matched': torch.cat(all_is_matched).numpy(),
    }

    # save results
    with open(os.path.join(run_dir, f'test_results_threshold{confidence_threshold}.csv'), 'w') as f:
        json.dump(results, f, indent=4)

    return results, raw

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=str)
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--confidence-threshold', type=float, default=0.5)
    parser.add_argument('--width-tolerance', type=float, default=0.1)
    args = parser.parse_args()

    results, raw = evaluate(args.run_dir, args.crop, args.confidence_threshold, args.width_tolerance)

    print(f'Precision: {results["precision"]:.4f}')
    print(f'Recall: {results["recall"]:.4f}')
    print(f'F1: {results["f1"]:.4f}')
    print(f'\nEnergy MAE: {results["energy_mae"]:.6f}')
    print(f'Gamma MAE: {results["gamma_mae"]:.6f}')
    print(f'J^pi Accuracy: {results["jpi_accuracy"]:.4f}')
    print(f'\nTotal true: {results["total_true"]}')
    print(f'True positives: {results["total_tp"]}')
    print(f'False positives: {results["total_fp"]}')
    print(f'False negatives: {results["total_fn"]}')
    print(f'True negatives: {results["total_tn"]}')