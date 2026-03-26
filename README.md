# Nuclear Scattering Resonance Identification with DETR

*Note this project is still in-progress, expected to finish in June 2026.*

A [DETR (DEtection TRansformer)](https://arxiv.org/pdf/2005.12872) model for identifying nuclear resonances from scattering cross-section data. The model takes differential cross-section spectra as input and predicts resonance properties: energy, partial widths (gamma), and quantum numbers (J^pi).

Currently targeting the O-16 compound nucleus with 3 particle pairs and 12 J^pi sets.

## Architecture

The model follows the DETR detection pipeline:

1. **2D CNN backbone** -- extracts features from the `[2, 512, n_channels]` input (data + crop mask), producing spatial tokens and a downsampled attention mask
2. **Transformer encoder** (6 layers) -- self-attention over spatial tokens with key padding mask to ignore cropped regions
3. **Transformer decoder** (6 layers) -- learned query embeddings cross-attend to encoder output
4. **Prediction heads** -- per-query MLPs for classification, energy, J^pi index, and partial widths

Bipartite matching (Hungarian algorithm) assigns predictions to targets, and the loss combines focal loss (classification), L1 (energy), cross-entropy (J^pi), and masked MSE (gamma).

## Data

### Input

Each sample is a `[512, n_channels]` tensor of log10 differential cross-sections on a uniform energy grid. Channels are organised as `n_pp_combos * n_angles` (e.g. 9 x 16 = 144 for O-16).

Raw data comes as `.gz` or `.jsonl.gz` files and is preprocessed into `.pt` files via `process/preprocess.py`.

### Targets

Per-sample targets:
- **class** -- binary (resonance / no resonance) per query slot
- **energy** -- normalised resonance energy in [0, 1]
- **gamma** -- log-normalised partial widths per channel
- **jpi_index** -- index into the 12 J^pi sets
- **info_weight** -- per-resonance weight based on data availability after cropping

## Data Augmentation

### Cropping

Three independent cropping transforms simulate incomplete experimental data:

- **Energy cropping** -- randomly removes up to `crop_energy` fraction of the energy range from either end
- **Angle cropping** -- drops random angle bins across all channels (minimum `min_angles` kept)
- **Channel cropping** -- drops entire entrance channels (minimum `min_channels` kept)

Cropped regions are zeroed in the data and marked in the mask channel. The mask is downsampled through the backbone and passed as `key_padding_mask` to the transformer, so attention only operates on valid positions.

### Information weighting

Per-resonance loss weights are computed based on how much data remains visible near each resonance after cropping, accounting for partial width branching across particle pairs.

### Other augmentations

- **Gaussian noise** in log10 space (`noise_sigma_log10`)
- **Amplitude scaling** -- random additive offset in log10 space (`amplitude_scale`)

## Curriculum Learning

When `curriculum_epochs > 0`, crop severity ramps linearly from zero to full over that many epochs:

- `crop_energy`: 0 -> max
- `min_angles`: n_angles (16) -> final min (default is 3)
- `min_channels`: n_entrances (3) -> final min (default is 1)

This lets the model learn the base task first, then gradually adapt to missing data. Validation always uses uncropped data for a stable comparison metric.

## Usage

### Training

```bash
python main.py params/[params_name].json
```

Configuration is set in the JSON params file. See `params/detr.json` for the base template.

### Evaluation

```bash
python test.py out/runs/<run_id> <crop_strength> <confidence_threshold>
```

### Output

Results are saved in `out/runs/<timestamp>_<model>_<run_name>/`:
- `params.json` -- training configuration
- `checkpoint.pt` -- best model weights (by validation loss)
- `train_results.csv` -- per-epoch loss breakdown
- `train_results.png` -- loss curves

## Project Structure

```
model/
  detr.py              -- DETR model, loss function, Hungarian matcher
  transformer_encoder.py
  transformer_decoder.py
  layer/
    backbone.py         -- 2D CNN backbone with mask downsampling
    encoder_layer.py    -- self-attention + FFN
    decoder_layer.py    -- self-attention + cross-attention + FFN
    positional_encoding.py

process/
  data.py              -- dataset class, preprocessing
  transforms.py        -- cropping, augmentation, info weighting
  header.py            -- nuclear data header (J^pi sets, channels)
  preprocess.py        -- raw data files to .pt conversion

train.py               -- training loop with curriculum scheduling
test.py                -- evaluation metrics and visualisation
main.py                -- CLI entry point
```