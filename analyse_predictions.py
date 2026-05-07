import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch

KNOWN_LEVELS_PATH = 'data/exp/o16_known_levels.json'
PREDICTIONS_DIR = 'out/predictions'
PLOT_DIR = 'out/predictions'
PLOT_PATH = f'{PLOT_DIR}/cluster_plot.png'
EXP_SAMPLE_PATH = 'data/exp/o16_exp_elastic_only_GLOBAL.pt'
FULL_PREDICTIONS_DIR = 'out/predictions_full'
FULL_PLOT_DIR = 'out/predictions_full'
FULL_EXP_SAMPLE_PATH = 'data/exp/o16_exp_all_GLOBAL.pt'

SHOW_PREDICTIONS = True
SHOW_KNOWN_ENERGY_LINES = False
EPOCH_MIN = 450
EPOCH_MAX = 600
LEGEND_FONTSIZE = 16
LEGEND_TITLE_FONTSIZE = 18
LEGEND_MARKERSIZE = 18

MODEL_RUNS = {
    'A': [
        'A1_seed22',
        'A2_seed8',
    ],
    'B': [
        'B1_seed37',
    ],
}

MODEL_COLOR_RAMPS = {
    'A': [
        ['#08306b', '#08519c', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef'],
        ['#00441b', '#006d2c', '#238b45', '#41ab5d', '#74c476', '#a1d99b', '#c7e9c0'],
    ],
    'B': [
        ['#67000d', '#a50f15', '#cb181d', '#ef3b2c', '#fb6a4a', '#fc9272', '#fcbba1'],
        ['#3f007d', '#54278f', '#6a51a3', '#807dba', '#9e9ac8', '#bcbddc', '#dadaeb'],
    ],
}

def _checkpoint_epoch(checkpoint):
    stem = Path(checkpoint).stem
    if stem.startswith('checkpoint_epoch'):
        return int(stem.replace('checkpoint_epoch', ''))
    return 999999

def _model_info(run):
    for label, patterns in MODEL_RUNS.items():
        for i, pattern in enumerate(patterns):
            if pattern in run:
                return label, i
    return None, None

def _ramp_color(colors, n, i):
    if n <= 1:
        return colors[len(colors) // 2]
    pos = round((n - 1 - i) * (len(colors) - 1) / (n - 1))
    return colors[pos]

def _all_run_labels(model_labels=None):
    labels = []
    for model_label, ramps in MODEL_COLOR_RAMPS.items():
        if model_labels is not None and model_label not in model_labels:
            continue
        for run_idx in range(len(ramps)):
            labels.append(f'{model_label}{run_idx + 1}')
    return labels

def _jpi_label(j, pi):
    j = float(j)
    j_text = f'{int(j)}' if j.is_integer() else f'{j:g}'
    return f'{j_text}{pi}'

def _active_energy_range(exp_sample_path):
    saved = torch.load(exp_sample_path, weights_only=False)
    tensor = saved['tensor']
    mask = tensor[1] > 0 if tensor.ndim == 3 and tensor.shape[0] == 2 else tensor > -7.9

    rows = (mask.sum(dim=1) > 0).nonzero(as_tuple=True)[0]
    e_min = float(saved['e_min'])
    e_max = float(saved['e_max'])
    if rows.numel() == 0:
        return e_min, e_max

    row_start = int(rows[0].item())
    row_end = int(rows[-1].item()) + 1
    E = int(mask.shape[0])

    return (
        e_min + (row_start / E) * (e_max - e_min),
        e_min + (row_end / E) * (e_max - e_min),
    )

def load_all(predictions_dir=PREDICTIONS_DIR, exp_sample_path=EXP_SAMPLE_PATH,
             include_best=False, best_only=False, model_labels=None,
             epoch_min=EPOCH_MIN, epoch_max=EPOCH_MAX, epoch_only=None):
    e_min_plot, e_max_plot = _active_energy_range(exp_sample_path)

    levels = [l for l in json.load(open(KNOWN_LEVELS_PATH))
              if e_min_plot <= l['energy_mev'] <= e_max_plot]
    levels.sort(key=lambda l: l['energy_mev'])

    preds = []
    run_to_info = {}
    for f in sorted(Path(predictions_dir).glob('*.json')):
        d = json.load(open(f))
        checkpoint = d.get('checkpoint', 'checkpoint.pt')
        is_best = checkpoint == 'checkpoint.pt'
        if best_only and not is_best:
            continue
        if not best_only and is_best and not include_best:
            continue

        run = d['run_dir'].rstrip('/').split('/')[-1]
        model_label, run_idx = _model_info(run)
        if model_label is None:
            model_label = chr(ord('A') + len(run_to_info))
            run_idx = 0
        run_to_info.setdefault(run, (model_label, run_idx))
        model_label, run_idx = run_to_info[run]
        if model_labels is not None and model_label not in model_labels:
            continue

        epoch = _checkpoint_epoch(checkpoint)
        if not is_best:
            if epoch_only is not None and epoch != epoch_only:
                continue
            if epoch_only is None and (epoch < epoch_min or epoch > epoch_max):
                continue

        label = f'{model_label}{run_idx + 1}'
        run_key = f'{run}|{checkpoint}'
        for p in d['predictions']:
            if e_min_plot <= p['energy'] <= e_max_plot:
                preds.append({
                    **p,
                    'run': run_key,
                    'model_label': model_label,
                    'run_idx': run_idx,
                    'run_label': label,
                    'checkpoint_epoch': epoch,
                })

    return levels, preds, (e_min_plot, e_max_plot)

def make_cluster_plot(preds, levels, energy_range, plot_path=PLOT_PATH,
                      title_extra=None, model_labels=None, epoch_title=None):
    e_min_plot, e_max_plot = energy_range

    all_jpis = {(p['j'], p['parity']) for p in preds} | {(l['j'], l['parity']) for l in levels}
    jpi_sorted = sorted(all_jpis, key=lambda jp: (jp[0], 0 if jp[1] == '+' else 1))
    jpi_to_y = {jp: i for i, jp in enumerate(jpi_sorted)}
    jpi_labels = [_jpi_label(j, pi) for j, pi in jpi_sorted]

    plot_labels = sorted(
        {p['run_label'] for p in preds},
        key=lambda label: (
            next(p['model_label'] for p in preds if p['run_label'] == label),
            next(p['run_idx'] for p in preds if p['run_label'] == label),
        ),
    )
    legend_labels = _all_run_labels(model_labels)

    run_colors = {}
    legend_colors = {}
    run_y_offsets = {}
    for model_label, ramps in MODEL_COLOR_RAMPS.items():
        for run_idx, colors in enumerate(ramps):
            label = f'{model_label}{run_idx + 1}'
            legend_colors[label] = _ramp_color(colors, 1, 0)

    for label in plot_labels:
        label_preds = [p for p in preds if p['run_label'] == label]
        model_label = label_preds[0]['model_label']
        run_idx = label_preds[0]['run_idx']
        epochs = sorted({p['checkpoint_epoch'] for p in label_preds})
        ramps = MODEL_COLOR_RAMPS.get(model_label, MODEL_COLOR_RAMPS['A'])
        colors = ramps[run_idx % len(ramps)]
        for i, epoch in enumerate(epochs):
            run_colors[(label, epoch)] = _ramp_color(colors, len(epochs), i)
            run_y_offsets[(label, epoch)] = (i % 7 - 3) * 0.04
        legend_colors[label] = _ramp_color(colors, len(epochs), len(epochs) // 2)

    fig, ax = plt.subplots(figsize=(15, max(6.5, 0.65 * len(jpi_sorted) + 3.0)))

    if SHOW_KNOWN_ENERGY_LINES:
        for lvl in levels:
            ax.axvline(lvl['energy_mev'], color='#012a4a', linestyle=':',
                       linewidth=1.1, alpha=0.52, zorder=1)

    if SHOW_PREDICTIONS:
        for label_idx, label in enumerate(plot_labels):
            rps = [p for p in preds if p['run_label'] == label]
            xs = [p['energy'] for p in rps]
            ys = [
                jpi_to_y[(p['j'], p['parity'])] + run_y_offsets[(label, p['checkpoint_epoch'])]
                for p in rps
            ]
            sizes = [40 + 200 * p['conf'] for p in rps]
            colors = [run_colors[(label, p['checkpoint_epoch'])] for p in rps]
            ax.scatter(xs, ys, s=sizes, color=colors, alpha=0.9,
                       edgecolors='none', linewidths=0.0, zorder=3)

    for lvl in levels:
        y = jpi_to_y[(lvl['j'], lvl['parity'])]
        ax.scatter(lvl['energy_mev'], y, marker='X', color='black',
                   s=240, linewidths=1.5, zorder=10, edgecolors='white')

    ax.set_yticks(list(jpi_to_y.values()))
    ax.set_yticklabels(jpi_labels)
    ax.set_xlim(e_min_plot, e_max_plot)
    ax.set_ylim(-0.6, len(jpi_sorted) - 0.4)
    ax.set_xlabel('Energy (MeV)')
    ax.set_ylabel(r'$J^\pi$')
    ax.grid(False)

    if SHOW_PREDICTIONS:
        title = f'Predictions vs known O16 levels ({e_min_plot:.3f}-{e_max_plot:.3f} MeV)'
        if title_extra:
            title += f' - {title_extra}'
        ax.set_title(title)
        if legend_labels:
            handles = [
                Line2D([0], [0], marker='o', linestyle='', markersize=LEGEND_MARKERSIZE,
                       markerfacecolor=legend_colors[label], markeredgecolor='none',
                       label=label)
                for label in legend_labels
            ]
            if epoch_title is None:
                epoch_title = f'Epoch {EPOCH_MIN}-{EPOCH_MAX}: light to dark'
            ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1.0),
                      fontsize=LEGEND_FONTSIZE,
                      title=f'Model run\n{epoch_title}',
                      title_fontsize=LEGEND_TITLE_FONTSIZE, frameon=False,
                      labelspacing=0.8, handletextpad=0.6)
    else:
        ax.set_title(f'Known O16 levels ({e_min_plot:.3f}-{e_max_plot:.3f} MeV)')

    fig.subplots_adjust(right=0.76)
    plt.savefig(plot_path, dpi=130, bbox_inches='tight')
    plt.close(fig)

def make_plot_bundle(predictions_dir, plot_dir, exp_sample_path):
    plot_jobs = [
        {
            'plot_path': f'{plot_dir}/cluster_plot.png',
            'title_extra': f'epochs {EPOCH_MIN}-{EPOCH_MAX}',
            'epoch_title': f'Epoch {EPOCH_MIN}-{EPOCH_MAX}: light to dark',
        },
        {
            'plot_path': f'{plot_dir}/cluster_plot_best.png',
            'title_extra': 'best checkpoints',
            'best_only': True,
            'epoch_title': 'Best checkpoint',
        },
        {
            'plot_path': f'{plot_dir}/cluster_plot_A.png',
            'title_extra': f'model A, epochs {EPOCH_MIN}-{EPOCH_MAX}',
            'model_labels': ['A'],
            'epoch_title': f'Epoch {EPOCH_MIN}-{EPOCH_MAX}: light to dark',
        },
        {
            'plot_path': f'{plot_dir}/cluster_plot_B.png',
            'title_extra': f'model B, epochs {EPOCH_MIN}-{EPOCH_MAX}',
            'model_labels': ['B'],
            'epoch_title': f'Epoch {EPOCH_MIN}-{EPOCH_MAX}: light to dark',
        },
        {
            'plot_path': f'{plot_dir}/cluster_plot_epoch0600.png',
            'title_extra': f'epoch {EPOCH_MAX}',
            'epoch_only': EPOCH_MAX,
            'epoch_title': f'Epoch {EPOCH_MAX}',
        },
    ]

    for job in plot_jobs:
        levels, preds, energy_range = load_all(
            predictions_dir=predictions_dir,
            exp_sample_path=exp_sample_path,
            best_only=job.get('best_only', False),
            model_labels=job.get('model_labels'),
            epoch_only=job.get('epoch_only'),
        )
        make_cluster_plot(
            preds, levels, energy_range,
            plot_path=job['plot_path'],
            title_extra=job.get('title_extra'),
            model_labels=job.get('model_labels'),
            epoch_title=job.get('epoch_title'),
        )
        print(f'Wrote {job["plot_path"]}')

if __name__ == '__main__':
    make_plot_bundle(PREDICTIONS_DIR, PLOT_DIR, EXP_SAMPLE_PATH)
    if Path(FULL_PREDICTIONS_DIR).exists() and list(Path(FULL_PREDICTIONS_DIR).glob('*.json')):
        make_plot_bundle(FULL_PREDICTIONS_DIR, FULL_PLOT_DIR, FULL_EXP_SAMPLE_PATH)
