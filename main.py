import train
import json

with open('params.json', 'r') as f:
    params = json.load(f)

config = params['config']

print(f'Config `{config}` loaded.\n')

# train the model and output training results
train.train(params)