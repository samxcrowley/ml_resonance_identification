import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import process.data as data
from process.data import ResonanceDataset
from model.detr import DETR_Model, DETR_Loss, HungarianMatcher
from model.set_detr import SetDETR_Model
from process.header import Header

SAMPLE_IDX = 242
DATASET_PATH = 'data/preprocessed/nlevels_20_test.pt'
CHECKPOINT_PATH = 'out/runs/0320_1622_set_detr_baseline_floorpaddingmask/checkpoint.pt'
PARAMS_PATH = 'params/set_detr.json'
CONFIDENCE_THRESHOLD = 0.5

header = Header()
with open(PARAMS_PATH, 'r') as f:
    params = json.load(f)

d = ResonanceDataset(path=DATASET_PATH)

model = SetDETR_Model(header, params)
checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
model.load_state_dict(checkpoint['model'])
model.eval()

tensor, target = d[SAMPLE_IDX]

with torch.no_grad():
    preds = model(tensor.unsqueeze(0))

target_mask = target['class'][:, 1] == 1.0
n_targets = target_mask.sum().item()
target_energies = target['energy'][target_mask].squeeze(1)
target_gammas = target['gamma'][target_mask]
target_gamma_masks = target['gamma_mask'][target_mask]
target_jpi = target['jpi_index'][target_mask].squeeze(1).long()

confidences = preds['class'][0].softmax(-1)[:, 1]
confident_mask = confidences > CONFIDENCE_THRESHOLD
n_preds = confident_mask.sum().item()
pred_energies = preds['energy'][0, confident_mask].squeeze(1)
pred_gammas = preds['gamma'][0, confident_mask]
pred_jpi_probs = preds['jpi_index'][0, confident_mask]
pred_jpi = pred_jpi_probs.argmax(-1)
pred_confidences = confidences[confident_mask]

prepared_targets = [{
    'class': torch.ones(n_targets, dtype=torch.long),
    'energy': target['energy'][target_mask],
    'gamma': target_gammas,
    'gamma_mask': target_gamma_masks,
    'jpi_index': target['jpi_index'][target_mask],
}]

prepared_preds = {
    'class': preds['class'][0, confident_mask].unsqueeze(0),
    'energy': preds['energy'][0, confident_mask].unsqueeze(0),
    'gamma': preds['gamma'][0, confident_mask].unsqueeze(0),
    'jpi_index': preds['jpi_index'][0, confident_mask].unsqueeze(0),
}

loss_fn = DETR_Loss(header, params)
matcher = HungarianMatcher()

indices = matcher(prepared_preds, prepared_targets)
pred_idx, target_idx = indices[0]

matched_pred_set = set(pred_idx.tolist())
matched_target_set = set(target_idx.tolist())

jpi_labels = [f"{int(s['j'])}{'+' if s['parity'] == 1 else '-'}" for s in header.jpi_sets]

print(f'\nTargets: {n_targets} resonances')
print(f'Predictions: {n_preds} confident (>{CONFIDENCE_THRESHOLD})')
print(f'Matched: {len(pred_idx)}')

# sort by target energy
order = target_energies[target_idx].argsort()
for idx in range(len(order)):

    i = order[idx]
    pi = pred_idx[i].item()
    ti = target_idx[i].item()

    te = target_energies[ti].item()
    pe = pred_energies[pi].item()

    tj = target_jpi[ti].item()
    pj = pred_jpi[pi].item()

    gamma_mask = target_gamma_masks[ti]
    valid_ch = gamma_mask.nonzero(as_tuple=True)[0]
    tg_str = '  '.join(f'ch{c.item()}: {target_gammas[ti, c].item():.3f}' for c in valid_ch)
    pg_str = '  '.join(f'ch{c.item()}: {pred_gammas[pi, c].item():.3f}' for c in valid_ch)

    print(f'\n#{idx+1} | energy: {te:.4f} > {pe:.4f}  jpi: {jpi_labels[tj]} > {jpi_labels[pj]}')
    print(f'target gamma: {tg_str}')
    print(f'  pred gamma: {pg_str}')

print()