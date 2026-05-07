import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

import process.data
from model.detr import DETR_Model
from process.header import Header

N_ANGLES = 16
N_Y = 512
CLAMP = 1e-8
LOG_FLOOR = np.log10(CLAMP)
CONF_THRESHOLD = 0.5

EXP_SAMPLE_PATH = 'data/exp/o16_exp_elastic_only_GLOBAL.pt'
BEST_CHECKPOINT = 'checkpoint.pt'
CHECKPOINT_GLOB = 'checkpoint_epoch*.pt'
OUT_DIR = 'out/predictions'
FULL_OUT_DIR = 'out/predictions_full'

def _jpi_nchannels(header):
    return {
        (float(s['j']), '+' if s['parity'] > 0 else '-'): len(s['channels'])
        for s in header.jpi_sets
    }

def _active_energy_window(tensor, e_min, e_max):
    mask = tensor[1] > 0 if tensor.ndim == 3 and tensor.shape[0] == 2 else tensor > LOG_FLOOR
    active_rows = (mask.sum(dim=1) > 0).nonzero(as_tuple=True)[0]
    if active_rows.numel() == 0:
        return None

    row_lo = int(active_rows[0].item())
    row_hi = int(active_rows[-1].item()) + 1
    E = int(mask.shape[0])
    return {
        'row_start': row_lo,
        'row_end_exclusive': row_hi,
        'norm_min': row_lo / E,
        'norm_max': row_hi / E,
        'energy_min': e_min + (row_lo / E) * (e_max - e_min),
        'energy_max': e_min + (row_hi / E) * (e_max - e_min),
    }

def _predict_checkpoint(model, header, checkpoint_path, tensor, e_min, e_max, active_window, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = {k.removeprefix('_orig_mod.'): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(state_dict)
    model.eval()

    print(f'\nCheckpoint: {checkpoint_path.name}')

    with torch.no_grad():
        preds = model(tensor.unsqueeze(0).to(device))

    class_logits = preds['class'][0]
    energy_norm = preds['energy'][0].squeeze(-1).cpu()
    j_logits = preds['j'][0]
    pi_logits = preds['pi'][0]
    gamma_norm = preds['gamma'][0].cpu()

    confidences = class_logits.softmax(-1)[:, 1].cpu()
    j_pred = j_logits.argmax(dim=-1).cpu()
    pi_pred = pi_logits.argmax(dim=-1).cpu()

    gamma_log = gamma_norm * (process.data.GAMMA_LOG_MAX - process.data.GAMMA_LOG_MIN) + process.data.GAMMA_LOG_MIN
    gamma_mev = 10.0 ** gamma_log
    gamma_kev = gamma_mev * 1000.0

    jpi_nchannels = _jpi_nchannels(header)
    max_channels = gamma_mev.shape[1]

    results = []
    filtered_outside_active = 0
    for q in range(len(confidences)):

        conf = confidences[q].item()
        if conf < CONF_THRESHOLD:
            continue

        e_norm = energy_norm[q].item()
        if active_window is not None and not (active_window['norm_min'] <= e_norm < active_window['norm_max']):
            filtered_outside_active += 1
            continue

        e_mev = e_min + e_norm * (e_max - e_min)
        j_val = j_pred[q].item() / 2.0
        pi_sym = '+' if pi_pred[q].item() == 1 else '-'
        n_ch = jpi_nchannels.get((j_val, pi_sym), max_channels)

        results.append({
            'energy': e_mev,
            'conf': conf,
            'j': j_val,
            'parity': pi_sym,
            'partial_widths': gamma_kev[q, :n_ch].tolist(),
            'total_width': float(gamma_mev[q, :n_ch].sum()) * 1000.0,
        })

    results.sort(key=lambda r: r['energy'])
    return results, filtered_outside_active

def _out_dir(exp_sample_path):
    sample_name = os.path.splitext(os.path.basename(exp_sample_path))[0]
    if sample_name == 'o16_exp_all_GLOBAL':
        return FULL_OUT_DIR
    return OUT_DIR

def main():
    run_dir = sys.argv[1]
    exp_sample_path = sys.argv[2] if len(sys.argv) > 2 else EXP_SAMPLE_PATH

    with open(os.path.join(run_dir, 'params.json')) as f:
        params = json.load(f)

    exp_dict = torch.load(exp_sample_path, weights_only=False)
    tensor = exp_dict['tensor']
    e_min = float(exp_dict['e_min'])
    e_max = float(exp_dict['e_max'])
    active_window = _active_energy_window(tensor, e_min, e_max)

    process.data.MAX_RESONANCES = params['max_resonances']
    header = Header(params['header'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DETR_Model(header, params)
    model.to(device)

    checkpoint_paths = [Path(run_dir) / BEST_CHECKPOINT]
    checkpoint_paths += sorted(Path(run_dir).glob(CHECKPOINT_GLOB))
    checkpoint_paths = [path for path in checkpoint_paths if path.exists()]
    if not checkpoint_paths:
        raise FileNotFoundError(f'No checkpoints found in {run_dir}')

    print(f'Using device: {device}')
    print(f'Loaded {run_dir}')
    print(f'Found {len(checkpoint_paths)} checkpoint(s)')

    run_id = os.path.normpath(run_dir).split(os.sep)[-1]
    sample_name = os.path.splitext(os.path.basename(exp_sample_path))[0]
    out_dir = _out_dir(exp_sample_path)

    for checkpoint_path in checkpoint_paths:
        results, filtered_outside_active = _predict_checkpoint(
            model, header, checkpoint_path, tensor, e_min, e_max, active_window, device,
        )

        print(f'Energy range: {e_min:.3f} - {e_max:.3f} MeV  |  '
              f'{len(results)} resonance(s) above threshold {CONF_THRESHOLD}')
        if active_window is not None:
            print(f"  Active window: {active_window['energy_min']:.3f} - "
                  f"{active_window['energy_max']:.3f} MeV "
                  f"(rows {active_window['row_start']}..{active_window['row_end_exclusive'] - 1})")
            print(f'  Filtered outside active window: {filtered_outside_active}')

        if not results:
            print('  (none)')
        else:
            print(f"  {'Energy (MeV)':>14}  {'Conf':>6}  {'Jpi':<6}  {'Total G (keV)':>14}  Partial G (keV)")
            print(f"  {'-'*14}  {'-'*6}  {'-'*6}  {'-'*14}  {'-'*30}")

            for r in results:
                jpi = f"{r['j']:.1f}{r['parity']}"
                pw = '  '.join(f"{w:.3f}" for w in r['partial_widths'])
                print(f"  {r['energy']:>14.4f}  {r['conf']:>6.3f}  {jpi:<6}"
                      f"  {r['total_width']:>14.3f}  {pw}")

        ckpt_name = checkpoint_path.stem
        ckpt_suffix = '' if ckpt_name == 'checkpoint' else f'__{ckpt_name}'
        out_path = os.path.join(out_dir, f'{run_id}__{sample_name}{ckpt_suffix}.json')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump({
                'run_dir': run_id,
                'checkpoint': checkpoint_path.name,
                'sample_path': exp_sample_path,
                'sample_name': sample_name,
                'e_min': e_min,
                'e_max': e_max,
                'active_energy_window': active_window,
                'filtered_outside_active': filtered_outside_active,
                'predictions': results,
            }, f, indent=2)
        print(f'Wrote predictions to {out_path}')

if __name__ == '__main__':
    main()
