import gzip
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# x, y are the axes and z is the value at each point
x_key = 'theta_cm_out'
y_key = 'cn_ex'
z_key = 'dsdO'

# output sequence shape:
# [n, n_A, n_E]
# output target shape:
# [n]
def get_cx_sequence(path, log_cx=True, compressed=True):

    if compressed:
        with gzip.open(path, 'rb') as f:
            json_bytes = f.read()
            json_str = json_bytes.decode()
            data = json.loads(json_str)
    else:
        with open(path, 'r') as f:
            data = json.load(f)

    n = len(data)

    tensors = []
    
    for i in range(n):
        
        points = data[i]['observable_sets'][0]['points']

        xs = sorted(set(p[x_key] for p in points))
        ys = sorted(set(p[y_key] for p in points))

        x_idx = {x: i for i, x in enumerate(xs)}
        y_idx = {y: i for i, y in enumerate(ys)}
        
        tensor = torch.zeros(len(xs), len(ys))

        for p in points:

            if log_cx:
                z = np.log10(p[z_key])
            else:
                z = p[z_key]

            x = x_idx[p[x_key]]
            y = y_idx[p[y_key]]
            tensor[x, y] = z

        tensors.append(tensor)

    return torch.stack(tensors)