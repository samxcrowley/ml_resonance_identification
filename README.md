# DETR Nuclear Scattering Resonance Identification

This repository trains and applies a DETR-style model for identifying nuclear
resonances in scattering cross-section data.

The model operates on log-scaled differential cross-section tensors with a
mask channel for missing or cropped regions. It predicts resonance
properties such as energy, widths, and spin-parity assignments.

## Workflow

The project is organized around three main stages: data processing, training,
and evaluation. Prediction and plotting scripts then apply trained models to
experimental-style inputs and summarize their outputs.

## Data Processing

The data-processing module converts raw resonance samples into tensors that can
be inputted to the model. It handles loading, channel metadata, tensor shaping,
cropping, masking, and augmentation.

Before cropping, each sample is stored as a log-scaled cross-section tensor with
shape `[E, C]`, where:

- `E` is the number of energy grid points.
- `C` is the number of observable channels.
- `C = n_pp_combos * n_angles`, with one block of angle bins per particle-pair
  combination.

After cropping, each model input has shape `[2, E, C]`:

- Channel `0` is the log-scaled cross-section data.
- Channel `1` is the visibility mask, where valid tensor regions are marked as
  visible, and cropped or missing regions are masked out.

During training, these are batched as `[N, 2, E, C]`.

Targets are padded to a fixed maximum number of resonances per sample:

- `class`: `[R, 2]`, no-resonance/resonance labels.
- `energy`: `[R, 1]`, normalized resonance energy.
- `gamma`: `[R, G]`, normalized partial widths.
- `gamma_mask`: `[R, G]`, valid-width mask.
- `jpi_index`: `[R, 1]`, spin-parity assignment index.

Here `R` is `max_resonances`, usually 20, and `G` is the maximum number of gamma channels
defined by the nuclear header.

### Cropping

Cropping simulates incomplete experimental coverage by removing regions from the
input tensor and updating the visibility mask. The different cropping parameters
are used to sample a range of experimental-like scenarios, as model learning
is very sensitive to the exact pattern of available channels, angles, and energy
coverage. The main crop parameters are:

- `crop_energy`: maximum fraction of the energy axis that may be removed from a
  kept channel.
- `min_angles`: minimum number of angle bins to keep for each kept particle-pair
  combination.
- `min_pp_combos`: minimum number of particle-pair combinations to keep.
- `max_pp_combos`: maximum number of particle-pair combinations to keep.
- `elastic_max_pp_combos`: optional stricter maximum when an elastic-only crop is
  sampled.
- `contiguous_angle_crop_p`: probability that kept angles are one contiguous
  angular window instead of a random subset.
- `shared_energy_crop_p`: probability that kept channels share one energy window
  instead of being cropped independently.
- `inelastic_dropout_p`: probability of keeping only elastic particle-pair
  combinations.
- `min_kept_channel_weight`: minimum resonance channel weight required for a resonance to
  remain a target after cropping.

Additional augmentations can add log-space noise, amplitude scaling, or blur to
the visible regions.

## Training

Training is configuration-driven. A params file specifies the model settings,
data paths, optimizer settings, checkpoint behaviour, and run metadata.

Training writes run outputs under `out/runs/`, including the resolved
configuration, checkpoints, training metrics, and diagnostic plots.

The training module coordinates dataset loading, model construction,
optimization, checkpointing, and metric logging. The model module contains the
DETR architecture, prediction heads, matching logic, and loss computation.

A params file is a flat JSON object. Its structure is roughly:

```jsonc
{
  "model": "detr",
  "run_name": "...",
  "seed": 22,

  "data_path": "...",
  "header": "...",
  "max_resonances": 20,
  "n_entrances": 3,
  "n_exits": 3,
  "n_angles": 16,

  "n_epochs": 600,
  "batch_size": 256,
  "lr": 1e-3,
  "weight_decay": 1e-4,
  "scheduler": "cosine",
  "warmup_epochs": 10,
  "eval_every_n": 10,
  "best_after_epoch": 120,
  "early_stop_patience": 80,
  "snapshot_every": 25,

  "d_transformer": 256,
  "n_queries": 25,
  "n_hidden": 512,
  "n_head": 8,
  "n_layers": 6,
  "dropout_p": 0.1,
  "norm": "batch",

  "cost_class": 2.5,
  "cost_energy": 2.5,
  "cost_gamma": 4.0,
  "cost_j": 0.5,
  "cost_pi": 0.5,
  "class_weights": [0.5, 1.0],

  "crop_energy": 0.5,
  "min_angles": 8,
  "min_pp_combos": 1,
  "max_pp_combos": 9,
  "elastic_max_pp_combos": 1,
  "contiguous_angle_crop_p": 0.5,
  "shared_energy_crop_p": 0.5,
  "inelastic_dropout_p": 0.33,
  "min_kept_channel_weight": 0.25,
  "curriculum_epochs": 120,

  "noise_sigma_log10": 0.2,
  "amplitude_scale": 0.3,
  "gaussian_blur_sigma": 0.5,

  "num_workers": 8,
  "compile": true,
  "grad_clip_norm": 1.0
}
```

The most important groups are:

- Dataset fields define the processed tensor source, nuclear header, target
  padding, and channel geometry.
- Optimizer fields control run length, batch size, learning rate, weight decay,
  warmup, and scheduling.
- Evaluation and checkpoint fields control validation frequency, when best-model
  selection starts, early stopping, and snapshot checkpoint cadence.
- Model fields control the DETR hidden size, number of learned prediction
  queries, transformer depth, attention heads, dropout, and normalization.
- Loss fields weight the matching and training losses for class, energy, gamma,
  spin, and parity predictions.
- Cropping and augmentation fields control how much data is hidden during
  training and how noisy or blurred the visible regions can become.
- `curriculum_epochs` ramps crop severity from easy to full-strength over the
  early part of training.