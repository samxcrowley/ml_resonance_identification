import gzip
import json
import os
import sys
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

def main():

    run_dir = sys.argv[1]
    with open(os.path.join(run_dir, 'params.json')) as f:
        params = json.load(f)

    if len(sys.argv) > 2:
        exp_sample_path = sys.argv[2]
    else:
        exp_sample_path = 'data/exp/o16_exp_aa_elastic_only.pt'

    exp_dict = torch.load(exp_sample_path)
    tensor = exp_dict['tensor']
    e_min = exp_dict['e_min']
    e_max = exp_dict['e_max']

    process.data.MAX_RESONANCES = params['max_resonances']
    header = Header(params['header'])

    model = DETR_Model(header, params)
    checkpoint = torch.load(os.path.join(run_dir, 'checkpoint.pt'), weights_only=False)
    state_dict = {k.removeprefix('_orig_mod.'): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(state_dict)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f'Using device: {device}')
    print(f'Loaded {run_dir}')

    with torch.no_grad():
        preds = model(tensor.unsqueeze(0).to(device))

    class_logits = preds['class'][0]
    energy_norm  = preds['energy'][0].squeeze(-1).cpu()
    j_logits = preds['j'][0]
    pi_logits = preds['pi'][0]
    gamma_norm = preds['gamma'][0].cpu()

    confidences = class_logits.softmax(-1)[:, 1].cpu()
    j_pred  = j_logits.argmax(dim=-1).cpu()
    pi_pred = pi_logits.argmax(dim=-1).cpu()

    gamma_log = gamma_norm * (process.data.GAMMA_LOG_MAX - process.data.GAMMA_LOG_MIN) + process.data.GAMMA_LOG_MIN
    gamma_mev = 10.0 ** gamma_log
    gamma_kev = gamma_mev * 1000.0

    jpi_nchannels = {
        (float(s['j']), '+' if s['parity'] > 0 else '-'): len(s['channels'])
        for s in header.jpi_sets
    }
    max_channels = gamma_mev.shape[1]

    results = []
    for q in range(len(confidences)):

        conf = confidences[q].item()
        if conf < CONF_THRESHOLD:
            continue

        e_kev = e_min + energy_norm[q].item() * (e_max - e_min)
        j_val = j_pred[q].item() / 2.0
        pi_sym = '+' if pi_pred[q].item() == 1 else '-'
        n_ch = jpi_nchannels.get((j_val, pi_sym), max_channels)

        results.append({
            'energy': e_kev,
            'conf': conf,
            'j': j_val,
            'parity': pi_sym,
            'partial_widths': gamma_kev[q, :n_ch].tolist(),
            'total_width': float(gamma_mev[q, :n_ch].sum()) * 1000.0,
        })

    results.sort(key=lambda r: r['energy'])

    print(f'\nEnergy range: {e_min:.3f} – {e_max:.3f} kev  |  '
          f'{len(results)} resonance(s) above threshold {CONF_THRESHOLD}')

    if not results:
        print('  (none)')
        return

    print(f"  {'Energy (keV)':>14}  {'Conf':>6}  {'Jpi':<6}  {'Total G (keV)':>14}  Partial G (keV)")
    print(f"  {'-'*14}  {'-'*6}  {'-'*6}  {'-'*14}  {'-'*30}")

    for r in results:
        jpi = f"{r['j']:.1f}{r['parity']}"
        pw  = '  '.join(f"{w:.3f}" for w in r['partial_widths'])
        print(f"  {r['energy']:>14.4f}  {r['conf']:>6.3f}  {jpi:<6}"
              f"  {r['total_width']:>14.3f}  {pw}")

    # save to json file
    run_dir = run_dir[9:]
    out_path = f'out/predictions/{run_dir}.json'
    with open(out_path, 'w') as f:
        json.dump({'run_dir': run_dir, 'e_min': e_min, 'e_max': e_max, 'predictions': results}, f, indent=2)
    print(f'Wrote predictions to {out_path}')

if __name__ == '__main__':
    main()