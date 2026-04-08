import json
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
import process.data as data
from model.detr import DETR_Model, DETR_Loss, HungarianMatcher
from process.header import Header
import process.transforms as transforms

test_data_path = 'data/preprocessed/nlevel_20_test.pt'

N_BINS = 512
E_RANGE_MEV = 8.0
N_RAW_POINTS = 400
RAW_SPACING_KEV = E_RANGE_MEV / N_RAW_POINTS * 1000

def _floor_to_kev(floor_bins):
    return floor_bins / N_BINS * E_RANGE_MEV * 1000

# returns (matched, unmatched_confs)
# matched is a list of dicts, one for each prediction and target (gt: ground truth)
# dict contains pred_conf, err_norm, err_mev, gt_width_mev, gt_width_bins, pred_jpi_correct, e_range
# unmatched_confs is list of confidences for unmatched queries
def _collect_records(run_dir, do_crop, width_tolerance):

    with open(f'{run_dir}/params.json', 'r') as f:
        params = json.load(f)
    data.MAX_RESONANCES = params['max_resonances']

    header = Header(params['header'])
    model = DETR_Model(header, params)
    checkpoint = torch.load(f'{run_dir}/checkpoint.pt', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    transform = transforms.get_augment_transform(noise_sigma_log10=0.0, amplitude_scale=0.0)
    crop_params = {}
    if do_crop:
        crop_params = {'crop_energy': 0.5, 'crop_angle': True, 'crop_channel': True}

    dataset = data.ResonanceDataset(test_data_path, crop_params, transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    loss_fn = DETR_Loss(header, params)
    matcher = HungarianMatcher()

    matched = []
    unmatched_confs = []

    with torch.no_grad():

        for tensor, targets_batch in loader:

            tensor = tensor.to(device, non_blocking=True)
            preds = model(tensor)

            targets = loss_fn.prepare_targets(targets_batch)
            indices = matcher(preds, targets)

            for n in range(len(targets)):

                pred_idx, target_idx = indices[n]
                n_queries = preds['class'][n].shape[0]
                confidences = preds['class'][n].softmax(-1)[:, 1].cpu()

                e_min = targets_batch['e_min'][n].item()
                e_max = targets_batch['e_max'][n].item()
                e_range = e_max - e_min

                matched_pred_set = set(pred_idx.tolist()) if len(pred_idx) > 0 else set()
                for q in range(n_queries):
                    if q not in matched_pred_set:
                        unmatched_confs.append(confidences[q].item())

                if len(pred_idx) == 0:
                    continue

                target_gamma = targets[n]['gamma'][target_idx].to(device)
                gamma_mask = targets[n]['gamma_mask'][target_idx].to(device)
                nan_mask = target_gamma.isnan()
                if nan_mask.any():
                    target_gamma = target_gamma.clone()
                    target_gamma[nan_mask] = 0.0
                    gamma_mask = gamma_mask.clone()
                    gamma_mask[nan_mask] = 0.0
                gamma_log = target_gamma * (data.GAMMA_LOG_MAX - data.GAMMA_LOG_MIN) + data.GAMMA_LOG_MIN
                gamma_linear = (10.0 ** gamma_log) * gamma_mask
                total_width_mev = gamma_linear.sum(dim=-1).cpu()

                pred_energy = preds['energy'][n][pred_idx].squeeze(-1).cpu()
                target_energy = targets[n]['energy'][target_idx].squeeze(-1).cpu()
                err_norm = (pred_energy - target_energy).abs()

                pred_j = preds['j'][n][pred_idx].argmax(dim=-1).cpu()
                pred_pi = preds['pi'][n][pred_idx].argmax(dim=-1).cpu()
                gt_j = targets[n]['j_index'][target_idx].cpu()
                gt_pi = targets[n]['pi'][target_idx].cpu()

                for i in range(len(pred_idx)):
                    qi = pred_idx[i].item()
                    w_mev = total_width_mev[i].item()
                    matched.append({
                        'pred_conf': confidences[qi].item(),
                        'err_norm': err_norm[i].item(),
                        'err_mev': err_norm[i].item() * e_range,
                        'gt_width_mev': w_mev,
                        'gt_width_bins': (w_mev / e_range) * N_BINS if e_range > 0 else 0.0,
                        'pred_j_correct': (pred_j[i]  == gt_j[i]).item(),
                        'pred_pi_correct': (pred_pi[i] == gt_pi[i]).item(),
                        'e_range': e_range,
                    })

    return matched, unmatched_confs

# precision, recall, F1 overall
# recall and energy MAE by width tolerance
def _compute_metrics(matched, unmatched_confs, confidence_threshold,
                     width_tolerance, tol_floor_bins):

    n_true = len(matched)
    n_true_res = sum(1 for r in matched if r['gt_width_bins'] >= 1.0)
    n_true_sub = n_true - n_true_res

    fp = sum(1 for c in unmatched_confs if c > confidence_threshold)

    tp = tp_res = tp_sub = 0
    j_correct = pi_correct = jpi_both_correct = jpi_total = 0
    err_all, err_res, err_sub = [], [], []

    for r in matched:
        floor = tol_floor_bins / N_BINS
        tol = max(width_tolerance * r['gt_width_mev'] / r['e_range'], floor) if r['e_range'] > 0 else floor
        confident = r['pred_conf'] > confidence_threshold
        close = r['err_norm'] < tol
        is_tp = confident and close

        if is_tp:
            tp += 1
            jpi_total += 1
            if r['pred_j_correct']:
                j_correct += 1
            if r['pred_pi_correct']:
                pi_correct += 1
            if r['pred_j_correct'] and r['pred_pi_correct']:
                jpi_both_correct += 1
            err_all.append(r['err_mev'])
            if r['gt_width_bins'] >= 1.0:
                tp_res += 1
                err_res.append(r['err_mev'])
            else:
                tp_sub += 1
                err_sub.append(r['err_mev'])
        elif confident:
            fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / n_true if n_true > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    def _stats(errs):
        a = np.array(errs)
        return (float(a.mean()), float(np.median(a)), float(a.std())) if len(a) > 0 else (float('nan'),) * 3

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'recall_resolvable': tp_res / n_true_res if n_true_res > 0 else 0.0,
        'recall_subbin': tp_sub / n_true_sub if n_true_sub > 0 else 0.0,
        'j_accuracy': j_correct / jpi_total if jpi_total > 0 else float('nan'),
        'pi_accuracy': pi_correct / jpi_total if jpi_total > 0 else float('nan'),
        'jpi_accuracy': jpi_both_correct / jpi_total if jpi_total > 0 else float('nan'),
        'tp': tp, 'tp_resolvable': tp_res, 'tp_subbin': tp_sub,
        'fp': fp, 'fn': n_true - tp,
        'n_true': n_true, 'n_true_resolvable': n_true_res, 'n_true_subbin': n_true_sub,
        'energy_mae_mev': _stats(err_all)[0],
        'energy_mae_mev_resolvable': _stats(err_res)[0],
        'energy_mae_mev_subbin': _stats(err_sub)[0],
    }

def _plot_sweep(run_dir, sweep, floors, matched_records):

    out_dir = os.path.join(run_dir, 'analysis')
    os.makedirs(out_dir, exist_ok=True)

    # metrics plot
    x = np.arange(len(floors))
    w = 0.25
    floor_labels = [f'{f}/512\n({_floor_to_kev(f):.0f} keV)' for f in floors]

    precisions  = [sweep[f]['precision'] for f in floors]
    recalls = [sweep[f]['recall'] for f in floors]
    f1s = [sweep[f]['f1'] for f in floors]
    recalls_res = [sweep[f]['recall_resolvable'] for f in floors]
    recalls_sub = [sweep[f]['recall_subbin'] for f in floors]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, bar_groups, title, ylabel in [
        (axes[0],
         [('Precision', precisions, 'steelblue'),
          ('Recall', recalls, 'seagreen'),
          ('F1', f1s, 'firebrick')],
         'Overall Metrics vs Tolerance Floor', 'Score'),
        (axes[1],
         [('Resolvable (≥1 bin)', recalls_res, 'steelblue'),
          ('Sub-bin (<1 bin)', recalls_sub, 'coral')],
         'Recall by Width Tier vs Tolerance Floor', 'Recall'),
    ]:

        n_groups = len(bar_groups)
        offsets = np.linspace(-w * (n_groups - 1) / 2, w * (n_groups - 1) / 2, n_groups)

        for (label, vals, color), offset in zip(bar_groups, offsets):
            bars = ax.bar(x + offset, vals, w, label=label, color=color, alpha=0.85)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(floor_labels, fontsize=9)
        ax.set_xlabel('Tolerance Floor')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'metrics_sweep.png'), dpi=150)
    plt.close(fig)

    # energy error plot
    err_res_kev = np.array([r['err_mev'] * 1000 for r in matched_records if r['gt_width_bins'] >= 1.0])
    err_sub_kev = np.array([r['err_mev'] * 1000 for r in matched_records if r['gt_width_bins'] < 1.0])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, errs, label, color in [
        (axes[0], err_res_kev, f'Resolvable (≥1 bin)  n={len(err_res_kev)}', 'steelblue'),
        (axes[1], err_sub_kev, f'Sub-bin (<1 bin)  n={len(err_sub_kev)}',    'coral'),
    ]:
        if len(errs) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        lo = max(errs.min(), 0.1)
        hi = errs.max()
        bins = np.logspace(np.log10(lo), np.log10(hi), 60) if hi > lo else 30
        ax.hist(errs, bins=bins, color=color, alpha=0.75)
        ax.axvline(errs.mean(), color='red', linestyle='--', label=f'Mean = {errs.mean():.1f} keV')
        ax.axvline(np.median(errs), color='orange', linestyle='--', label=f'Median = {np.median(errs):.1f} keV')
        ax.set_xscale('log')
        ax.set_xlabel('Energy Error (keV)')
        ax.set_ylabel('Count (matched pairs)')
        ax.set_title(f'Energy Error Distribution\n{label}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    ymax = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    for ax in axes:
        ax.set_ylim(0, ymax)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'energy_errors.png'), dpi=150)
    plt.close(fig)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=str)
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--confidence-threshold', type=float, default=0.5)
    parser.add_argument('--width-tolerance', type=float, default=0.5)
    args = parser.parse_args()

    matched, unmatched = _collect_records(args.run_dir, args.crop, args.width_tolerance)

    floors = [1, 2, 3, 4, 5]

    sweep = {}
    for f in floors:
        sweep[f] = _compute_metrics(matched, unmatched, args.confidence_threshold, args.width_tolerance, f)

    matplotlib.use('agg')

    _plot_sweep(args.run_dir, sweep, floors, matched)

    floor = 5

    m = sweep[floor]
    print(f'Results for floor = {floor}/512 = {_floor_to_kev(floor):.0f} keV:')
    print(f'  Precision:  {m["precision"]:.4f}')
    print(f'  Recall:     {m["recall"]:.4f}  (overall)')
    print(f'              {m["recall_resolvable"]:.4f}  '
          f'(resolvable ≥1 bin, n={m["n_true_resolvable"]})')
    print(f'              {m["recall_subbin"]:.4f}  '
          f'(sub-bin <1 bin, n={m["n_true_subbin"]})')
    print(f'  F1:         {m["f1"]:.4f}')
    print(f'  J acc:      {m["j_accuracy"]:.4f}')
    print(f'  pi acc:     {m["pi_accuracy"]:.4f}')
    print(f'  J^pi acc:   {m["jpi_accuracy"]:.4f}')
    print(f'  Energy MAE: {m["energy_mae_mev"]*1000:.1f} keV  (overall TPs)')
    print(f'              {m["energy_mae_mev_resolvable"]*1000:.1f} keV  (resolvable)')
    print(f'              {m["energy_mae_mev_subbin"]*1000:.1f} keV  (sub-bin)')
    print(f'  Counts:\n{m["n_true"]} true  '
          f'{m["tp"]} TP  {m["fp"]} FP  {m["fn"]} FN')
