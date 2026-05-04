import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

E_WINDOW_KEV = 150
KNOWN_LEVELS_PATH = 'data/exp/o16_known_levels.json'
PREDICTIONS_DIR = 'out/predictions'
PLOT_PATH = 'out/predictions/cluster_plot.png'
E_MIN_MEV, E_MAX_MEV = 8.5, 12.2

def load_all():

    levels = [l for l in json.load(open(KNOWN_LEVELS_PATH))
              if E_MIN_MEV <= l['energy_mev'] <= E_MAX_MEV]
    levels.sort(key=lambda l: l['energy_mev'])

    preds = []
    for f in sorted(Path(PREDICTIONS_DIR).rglob('*.json')):
        d = json.load(open(f))
        run = d['run_dir'].rstrip('/').split('/')[-1]
        for p in d['predictions']:
            if E_MIN_MEV <= p['energy'] <= E_MAX_MEV:
                preds.append({**p, 'run': run})

    return levels, preds

def make_cluster_plot(preds, levels):

    all_jpis = {(p['j'], p['parity']) for p in preds} | {(l['j'], l['parity']) for l in levels}
    jpi_sorted = sorted(all_jpis, key=lambda jp: (jp[0], 0 if jp[1] == '+' else 1))
    jpi_to_y = {jp: i for i, jp in enumerate(jpi_sorted)}
    jpi_labels = [f"{int(j)}{pi}" for j, pi in jpi_sorted]

    runs = sorted({p['run'] for p in preds})
    cmap = plt.get_cmap('tab10' if len(runs) <= 10 else 'tab20')
    run_colors = {r: cmap(i % cmap.N) for i, r in enumerate(runs)}

    fig, ax = plt.subplots(figsize=(15, max(6, 0.6 * len(jpi_sorted) + 3)))

    for lvl in levels:
        y = jpi_to_y[(lvl['j'], lvl['parity'])]
        rect = mpatches.Rectangle(
            (lvl['energy_mev'] - E_WINDOW_KEV / 1000, y - 0.35),
            2 * E_WINDOW_KEV / 1000, 0.7,
            facecolor='lightgray', alpha=0.4, edgecolor='none', zorder=1,
        )
        ax.add_patch(rect)

    for r in runs:
        rps = [p for p in preds if p['run'] == r]
        xs = [p['energy'] for p in rps]
        ys = [jpi_to_y[(p['j'], p['parity'])] + (hash(r) % 7 - 3) * 0.04 for p in rps]
        sizes = [40 + 200 * p['conf'] for p in rps]
        ax.scatter(xs, ys, s=sizes, color=run_colors[r], alpha=0.65,
                   edgecolors='white', linewidths=0.5, zorder=3, label=r[:40])

    for lvl in levels:
        y = jpi_to_y[(lvl['j'], lvl['parity'])]
        ax.scatter(lvl['energy_mev'], y, marker='X', color='black',
                   s=240, linewidths=1.5, zorder=10, edgecolors='white')

    ax.set_yticks(list(jpi_to_y.values()))
    ax.set_yticklabels(jpi_labels)
    ax.set_xlim(E_MIN_MEV, E_MAX_MEV)
    ax.set_ylim(-0.6, len(jpi_sorted) - 0.4)
    ax.set_xlabel('Energy (MeV)')
    ax.set_ylabel(r'$J^\pi$')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_title(f'Predictions vs known O16 levels ({E_MIN_MEV:.2f}–{E_MAX_MEV:.2f} MeV) '
                 f'— X = known level, gray band = ±{E_WINDOW_KEV} keV match window')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=8, title='Run', frameon=False)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=130, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    levels, preds = load_all()
    make_cluster_plot(preds, levels)