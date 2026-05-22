import argparse
import json
from pathlib import Path
import numpy as np
import torch
import process.data
from model.detr import DETR_Model
from process.header import Header

CLAMP = 1e-8
LOG_FLOOR = np.log10(CLAMP)
CONF_THRESHOLD = 0.5

def load_params(checkpoint_path, params_path=None):

    if params_path is None:
        params_path = checkpoint_path.parent / "params.json"
    else:
        params_path = Path(params_path)

    if not params_path.exists():
        raise FileNotFoundError(
            f"Could not find params file at {params_path}. "
            "Pass one explicitly with --params."
        )

    with params_path.open("r", encoding="utf-8") as f:
        return json.load(f), params_path

def load_sample(sample_path):

    sample = torch.load(sample_path, map_location="cpu", weights_only=False)

    if isinstance(sample, dict) and "tensor" in sample:
        tensor = sample["tensor"]
        e_min = sample.get("e_min")
        e_max = sample.get("e_max")
    else:
        raise ValueError("Sample file must be a .pt dict containing 'tensor'.")

    if e_min is None or e_max is None:
        raise ValueError("Sample file must include e_min and e_max values.")

    if torch.is_tensor(e_min):
        e_min = e_min.item()
    if torch.is_tensor(e_max):
        e_max = e_max.item()

    tensor = tensor.float()
    if tensor.ndim == 2:
        mask = (tensor > LOG_FLOOR).float()
        tensor = torch.stack([tensor, mask], dim=0)
    elif tensor.ndim != 3 or tensor.shape[0] != 2:
        raise ValueError(
            "Sample tensor must have shape [E, C] or model-ready shape [2, E, C]."
        )

    return tensor, float(e_min), float(e_max)

def load_model(checkpoint_path, params, device):

    if params.get("model", "detr") != "detr":
        raise ValueError(f"Unsupported model type: {params['model']}")

    process.data.MAX_RESONANCES = params["max_resonances"]
    header = Header(params["header"])
    model = DETR_Model(header, params).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    return model, header

def jpi_channel_counts(header):

    return {
        (float(s["j"]), "+" if float(s["parity"]) > 0 else "-"): len(s["channels"])
        for s in header.jpi_sets
    }

def predict(model, header, tensor, e_min, e_max, device):

    with torch.no_grad():
        output = model(tensor.unsqueeze(0).to(device))

    class_logits = output["class"][0]
    energy_norm = output["energy"][0].squeeze(-1).cpu()
    gamma_norm = output["gamma"][0].cpu()
    j_pred = output["j"][0].argmax(dim=-1).cpu()
    pi_pred = output["pi"][0].argmax(dim=-1).cpu()
    confidences = class_logits.softmax(-1)[:, 1].cpu()

    gamma_log = (
        gamma_norm * (process.data.GAMMA_LOG_MAX - process.data.GAMMA_LOG_MIN)
        + process.data.GAMMA_LOG_MIN
    )
    gamma_kev = (10.0 ** gamma_log) * 1000.0

    channel_counts = jpi_channel_counts(header)
    max_channels = gamma_kev.shape[1]

    predictions = []
    for query_index, confidence in enumerate(confidences):
        confidence = float(confidence.item())
        if confidence < CONF_THRESHOLD:
            continue

        energy = e_min + float(energy_norm[query_index].item()) * (e_max - e_min)
        j = float(j_pred[query_index].item()) / 2.0
        parity = "+" if int(pi_pred[query_index].item()) == 1 else "-"
        n_channels = channel_counts.get((j, parity), max_channels)
        partial_widths = gamma_kev[query_index, :n_channels].tolist()

        predictions.append(
            {
                "energy": energy,
                "conf": confidence,
                "j": j,
                "parity": parity,
                "partial_widths": partial_widths,
                "total_width": float(sum(partial_widths)),
            }
        )

    return sorted(predictions, key=lambda pred: pred["energy"])

def print_predictions(predictions):

    if not predictions:
        print("No predictions above threshold.")
        return

    print(f"{'Energy (MeV)':>14}  {'Conf':>6}  {'Jpi':<6}  {'Total G (keV)':>14}  Partial G (keV)")
    print(f"{'-' * 14}  {'-' * 6}  {'-' * 6}  {'-' * 14}  {'-' * 30}")
    for pred in predictions:
        jpi = f"{pred['j']:.1f}{pred['parity']}"
        partials = "  ".join(f"{width:.3f}" for width in pred["partial_widths"])
        print(
            f"{pred['energy']:>14.4f}  {pred['conf']:>6.3f}  {jpi:<6}"
            f"  {pred['total_width']:>14.3f}  {partials}"
        )

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("sample")
    parser.add_argument("--params", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    sample_path = Path(args.sample)

    params, params_path = load_params(checkpoint_path, args.params)
    tensor, e_min, e_max = load_sample(sample_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, header = load_model(checkpoint_path, params, device)

    predictions = predict(
        model=model,
        header=header,
        tensor=tensor,
        e_min=e_min,
        e_max=e_max,
        device=device,
    )

    print(f"Using device: {device}")
    print(f"Energy range: {e_min:.3f} - {e_max:.3f} MeV")
    print_predictions(predictions)

    if args.output:
        output = {
            "checkpoint": str(checkpoint_path),
            "params": str(params_path),
            "sample": str(sample_path),
            "e_min": e_min,
            "e_max": e_max,
            "threshold": CONF_THRESHOLD,
            "predictions": predictions,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"Wrote predictions to {output_path}")

if __name__ == "__main__":
    main()