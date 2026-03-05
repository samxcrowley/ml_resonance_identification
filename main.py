import train
import json
import os
from plotting import plot_results

with open('params.json', 'r') as f:
    params = json.load(f)

print(f'Config `{params["config"]}` loaded.\n')

run_id = train.train(params)

run_dir = os.path.join('out', 'runs', run_id)

subtitle = (
    f'max_resonances={params["max_resonances"]}'
)

plot_results(
    path=os.path.join(run_dir, 'results.csv'),
    title=run_id,
    out_path=os.path.join(run_dir, 'plot.png'),
    subtitle=subtitle,
)