import json
import torch
from dataclasses import dataclass

class Header:

    def __init__(self, filename='o16_header'):

        self.filename = filename

        with open(f'data/header/{filename}.json') as f:
            data = json.load(f)

        self.jpi_sets = data['jpi_sets']
        self.n_jpi_sets = len(self.jpi_sets)

        self.max_channels = self.get_max_channels(data)

    def get_max_channels(self, data):

        max_channels = 0

        for set in self.jpi_sets:

            if len(set['channels']) > max_channels:
                max_channels = len(set['channels'])

        return max_channels

    def get_set_n_channels(self, set_index):

        return len(self.jpi_sets[set_index]['channels'])

    def get_jpi_to_j_index(self):
        return torch.tensor([int(s['j'] * 2) for s in self.jpi_sets], dtype=torch.long)

    # 0: -, 1: +
    def get_jpi_to_pi_index(self):
        return torch.tensor([0 if s['parity'] < 0 else 1 for s in self.jpi_sets], dtype=torch.long)