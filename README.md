# Model

A [DETR (DEtection TRansformer)](https://arxiv.org/pdf/2005.12872) model for resonance prediction in nuclear scattering cross-section data.

Currently predicts energy level and total width (gamma total).

# Data

## Training data

Raw data comes as `json` files compressed into `.gz` files. They store the cross-section and target data for a number of samples.

## Output

Results of training runs are saved in `out/runs`, identified by timestamp and the config selected for that run. A csv file of results, along with plots and parameters used are saved.