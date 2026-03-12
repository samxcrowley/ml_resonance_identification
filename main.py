import train
import json
import os
import sys
from process.plotting import plot_results

params_file = 'training_params.json'
if len(sys.argv) == 2:
    params_file = sys.argv[1]

with open(params_file, 'r') as f:
    params = json.load(f)

print(f'\nModel `{params["model"]}` loaded.\n')

print(f'Starting run `{params["run_name"]}`.\n')

run_id = train.train(params)

run_dir = os.path.join('out', 'runs', run_id)

plot_results(
    path=os.path.join(run_dir, 'results.csv'),
    title=run_id,
    out_path=os.path.join(run_dir, 'train_results.png')
)