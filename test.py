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

test_data_path = 'data/preprocessed/nlevel_20_angle20-170_test.pt'

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
    state_dict = {k.removeprefix('_orig_mod.'): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(state_dict)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    transform = transforms.get_augment_transform(noise_sigma_log10=0.0, amplitude_scale=0.0)
    crop_params = {}
    if do_crop:
        crop_params = {'crop_energy': 0.5, 'min_angles': 3, 'min_pp_combos': 1}

    channel_filter = params.get('channel_filter', None)
    dataset = data.ResonanceDataset(test_data_path, crop_params, transform,
                                    channel_filter=channel_filter)
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

def _sweep_line_plot(ax, sweep_u, sweep_c, floors, xs, floor_labels, groups, title, ylabel):

    for label, key, color in groups:
        vals_u = [sweep_u[f][key] for f in floors]
        vals_c = [sweep_c[f][key] for f in floors]
        ax.plot(xs, vals_u, color=color, linestyle='-', marker='o', label=f'{label} (uncropped)')
        ax.plot(xs, vals_c, color=color, linestyle='--', marker='s', label=f'{label} (cropped)')
        for x, v in zip(xs, vals_u):
            ax.annotate(f'{v:.2f}', (x, v), textcoords='offset points', xytext=(0, 6),
                        ha='center', fontsize=7, color=color)
        for x, v in zip(xs, vals_c):
            ax.annotate(f'{v:.2f}', (x, v), textcoords='offset points', xytext=(0, -12),
                        ha='center', fontsize=7, color=color)

    ax.set_xticks(xs)
    ax.set_xticklabels(floor_labels, fontsize=9)
    ax.set_xlabel('Tolerance Floor')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

def _plot_sweep(run_dir, sweep_u, sweep_c, floors, matched_u, matched_c):

    out_dir = os.path.join(run_dir, 'analysis')
    os.makedirs(out_dir, exist_ok=True)

    floor_labels = [f'{f}/512\n({_floor_to_kev(f):.0f} keV)' for f in floors]
    xs = list(range(len(floors)))

    # recall
    fig, ax = plt.subplots(figsize=(8, 5))
    _sweep_line_plot(ax, sweep_u, sweep_c, floors, xs, floor_labels, [
        ('Overall', 'recall', 'steelblue'),
        ('Resolvable (≥1 bin)', 'recall_resolvable', 'seagreen'),
        ('Sub-bin (<1 bin)', 'recall_subbin', 'coral'),
    ], 'Recall vs Tolerance Floor', 'Recall')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'recall.png'), dpi=150)
    plt.close(fig)

    # precision
    fig, ax = plt.subplots(figsize=(8, 5))
    _sweep_line_plot(ax, sweep_u, sweep_c, floors, xs, floor_labels, [
        ('Precision', 'precision', 'steelblue'),
    ], 'Precision vs Tolerance Floor', 'Precision')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'precision.png'), dpi=150)
    plt.close(fig)

    # F1
    fig, ax = plt.subplots(figsize=(8, 5))
    _sweep_line_plot(ax, sweep_u, sweep_c, floors, xs, floor_labels, [
        ('F1', 'f1', 'firebrick'),
    ], 'F1 vs Tolerance Floor', 'F1')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'f1.png'), dpi=150)
    plt.close(fig)

    _plot_energy_errors(out_dir, matched_u, matched_c,
                        'localisation_errors.png',
                        'Localisation Error (all matched pairs)')

def _filter_tp(matched, confidence_threshold, width_tolerance, tol_floor_bins):
    tp = []
    for r in matched:
        floor = tol_floor_bins / N_BINS
        tol = max(width_tolerance * r['gt_width_mev'] / r['e_range'], floor) if r['e_range'] > 0 else floor
        if r['pred_conf'] > confidence_threshold and r['err_norm'] < tol:
            tp.append(r)
    return tp

def _plot_energy_errors(out_dir, matched_u, matched_c, filename, suptitle):

    row_data = [
        ('Uncropped', matched_u),
        ('Cropped', matched_c),
    ]

    col_colours = [
        ('Resolvable (≥1 bin)', lambda r: r['gt_width_bins'] >= 1.0, 'cornflowerblue'),
        ('Sub-bin (<1 bin)', lambda r: r['gt_width_bins'] <  1.0, 'lightcoral'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(suptitle, fontsize=13)

    col_ymaxes = [0.0, 0.0]

    for row_i, (row_label, matched) in enumerate(row_data):
        for col_i, (col_label, pred, base_colour) in enumerate(col_colours):

            if row_i == 0: # light colour for uncropped
                colour = base_colour

            else: # dark colour for cropped
                rgb = np.array(matplotlib.colors.to_rgb(base_colour)) * 255
                darken_amount = 50
                colour = matplotlib.colors.to_hex((np.clip(rgb - darken_amount, 0, 255) / 255))

            ax = axes[row_i][col_i]
            errs = np.array([r['err_mev'] * 1000 for r in matched if pred(r)])

            if len(errs) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{row_label} — {col_label}')
                continue

            lo = max(errs.min(), 0.1)
            hi = errs.max()
            bins = np.logspace(np.log10(lo), np.log10(hi), 60) if hi > lo else 30
            ax.hist(errs, bins=bins, color=colour, alpha=0.75)
            ax.axvline(errs.mean(), color='red', linestyle='--', label=f'Mean = {errs.mean():.1f} keV')
            ax.axvline(np.median(errs), color='gold', linestyle='--', label=f'Median = {np.median(errs):.1f} keV')
            ax.set_xscale('log')
            ax.set_xlabel('Energy Error (keV)')
            ax.set_ylabel('Count')
            ax.set_title(f'{row_label} — {col_label}  n={len(errs)}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            col_ymaxes[col_i] = max(col_ymaxes[col_i], ax.get_ylim()[1])

    for col_i in range(2):
        for row_i in range(2):
            axes[row_i][col_i].set_ylim(0, col_ymaxes[col_i])

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, filename), dpi=150)
    plt.close(fig)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=str)
    parser.add_argument('--confidence-threshold', type=float, default=0.5)
    parser.add_argument('--width-tolerance', type=float, default=0.5)
    parser.add_argument('--floor', type=int, default=3)
    args = parser.parse_args()

    matched_u, unmatched_u = _collect_records(args.run_dir, False, args.width_tolerance)
    matched_c, unmatched_c = _collect_records(args.run_dir, True,  args.width_tolerance)

    floors = [1, 2, 3, 4, 5]

    sweep_u, sweep_c = {}, {}
    for f in floors:
        sweep_u[f] = _compute_metrics(matched_u, unmatched_u, args.confidence_threshold, args.width_tolerance, f)
        sweep_c[f] = _compute_metrics(matched_c, unmatched_c, args.confidence_threshold, args.width_tolerance, f)

    tp_u = _filter_tp(matched_u, args.confidence_threshold, args.width_tolerance, args.floor)
    tp_c = _filter_tp(matched_c, args.confidence_threshold, args.width_tolerance, args.floor)

    matplotlib.use('agg')

    out_dir = os.path.join(args.run_dir, 'analysis')
    os.makedirs(out_dir, exist_ok=True)

    _plot_sweep(args.run_dir, sweep_u, sweep_c, floors, matched_u, matched_c)
    _plot_energy_errors(out_dir, tp_u, tp_c,
                        'detection_errors.png',
                        f'Detection Energy Error (TPs only, floor={args.floor}/512, {_floor_to_kev(args.floor):.0f} keV)')
