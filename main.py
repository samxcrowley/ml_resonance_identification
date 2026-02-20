import torch
import train
import model

params = {

    'seed': 22,

    'data_path': 'data/10.json',
    'data_is_compressed': False,

    'num_workers': 4,
    'batch_size': 1,
    'n_epochs': 100,
    'lr': 1e-3,
    'weight_decay': 0.0

}

d_model = 16
pool_kernel_size = 2
n_hidden = d_model * 4
n_head = 4
n_layers = 6
dropout_p = 0.0

model = model.encoder.Encoder(d_model, pool_kernel_size, n_hidden, n_head, n_layers, dropout_p)
# model = model.torch_encoder.TorchEncoder(d_model, pool_kernel_size, n_head, n_layers)

if __name__ == "__main__":
    train.train(params, model)