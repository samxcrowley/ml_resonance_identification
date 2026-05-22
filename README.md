# DETR Nuclear Scattering Resonance Identification

This repository trains and applies a DETR-style model for identifying nuclear
resonances in scattering cross-section data. The model predicts resonance
energy, partial widths, and spin-parity assignments from log-scaled
cross-section tensors.

## Setup

Install the Python dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data Format

Preprocessed training datasets are saved as `.pt` files containing:

- `tensors`: log-scaled cross-section tensors with shape `[N, E, C]`.
- `targets`: padded resonance labels for training.
- `metadata`: channel and angle metadata.

Here `E` is the number of energy grid points and `C` is the number of observable
channels. During training, random cropping converts each sample into a model
input with shape `[2, E, C]`:

- channel `0`: log-scaled cross-section data.
- channel `1`: visibility mask for available data.

Prediction samples should already be on the same grid/channel layout used for
training. A single prediction sample is a `.pt` dict with:

- `tensor`: one sample, either `[2, E, C]` or `[E, C]`.
- `e_min`: lower energy bound in MeV.
- `e_max`: upper energy bound in MeV.

## Preprocessing

Raw synthetic samples are converted into train/test (90\%/10\% split) `.pt` files with:

```bash
python process/preprocess.py <pattern>
```

The script searches `data/raw/` for files matching `<pattern>` and writes:

```text
data/preprocessed/<pattern>_train.pt
data/preprocessed/<pattern>_test.pt
```

The preprocessing stage builds the uniform energy/channel tensors, target
arrays, metadata, and channel-weight information used by training.

## Training

Training is controlled by a JSON params file:

```bash
python main.py params/detr.json
```

The params file specifies the dataset path, nuclear header, model size,
optimizer settings, cropping settings, and checkpoint behaviour. Training writes
outputs under `out/runs/<run_id>/`, including:

- `params.json`: resolved run configuration.
- `checkpoint.pt`: best model checkpoint.
- `checkpoint_epoch*.pt`: optional snapshot checkpoints.
- `train_results.csv` and `train_results.png`: training curves.

## Prediction

Run predictions for one trained checkpoint and one preprocessed experimental
sample with:

```bash
python predict.py out/runs/<run_id>/checkpoint.pt data/exp/<sample>.pt --output out/predictions/<sample>.json
```

By default, `predict.py` loads `params.json` from the same directory as the
checkpoint. If the params file is somewhere else, pass it explicitly:

```bash
python predict.py checkpoint.pt sample.pt --params path/to/params.json --output predictions.json
```

The script loads the model, runs the single sample, filters predictions using
`CONF_THRESHOLD = 0.5`, prints a table, and optionally writes JSON containing
the predicted energies, confidences, spin-parity assignments, partial widths,
and total widths.