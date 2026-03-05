import json
from dataclasses import dataclass

class Header:

    def __init__(self, filename='o16_header'):

        self.filename = filename

        with open(f'data/header/{filename}.json') as f:
            data = json.load(f)

        self.jpi_sets = data['jpi_sets']
        self.n_jpi_sets = len(jpi_sets)