import train
import json
import os
from process.plotting import plot_results

with open('training_params.json', 'r') as f:
    params = json.load(f)

print(f'\nConfig `{params["config"]}` loaded.\n')

print(f'Starting run `{params["run_name"]}`.\n')

run_id = train.train(params)

run_dir = os.path.join('out', 'runs', run_id)

plot_results(
    path=os.path.join(run_dir, 'results.csv'),
    title=run_id,
    out_path=os.path.join(run_dir, 'plot.png')
)