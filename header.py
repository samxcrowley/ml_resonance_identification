import json

class Header():

    def __init__(self, filename):

        self.filename = filename

        self.read_header(self.filename)

    def read_header(self, filename):

        with open(f'data/header/{filename}') as f:
            data = json.load(f)

        cn = data['cn']

        print(cn)

        pp = data['pp']

        print(pp)

        jpi_sets = data['jpi_sets']

        print(jpi_sets)

    # TODO: is a transform?
    # probably should be a transform (embedding), as different models might want
    # different embeddings or ways of including the header data
    # transform is a block that combines the header and main data together,
    # sort of like the combining blocks in the medical imaging paper
    def to_tensor(self):
        return torch.randn(10, 10)