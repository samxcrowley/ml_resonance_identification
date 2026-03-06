import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_results(path, title, out_path):

    df = pd.read_csv(path)

    epochs = df["epoch"]

    fig = plt.figure(figsize=(11, 9))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3, top=0.91)

    # loss
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_loss.plot(epochs, df["train_total_loss"], label="Training")
    ax_loss.plot(epochs, df["val_total_loss"], label="Validation")
    ax_loss.set_title("Total Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # precision
    ax_prec = fig.add_subplot(gs[1, 0])
    ax_prec.plot(epochs, df["val_precision"], color="tab:orange", label="Precision")
    ax_prec.set_ylim(0, 1.05)
    ax_prec.set_title("Precision")
    ax_prec.set_xlabel("Epoch")
    ax_prec.set_ylabel("Precision")
    ax_prec.grid(True, alpha=0.3)

    # recall
    ax_rec = fig.add_subplot(gs[1, 1])
    ax_rec.plot(epochs, df["val_recall"], color="tab:green", label="Recall")
    ax_rec.set_ylim(0, 1.05)
    ax_rec.set_title("Recall")
    ax_rec.set_xlabel("Epoch")
    ax_rec.set_ylabel("Recall")
    ax_rec.grid(True, alpha=0.3)

    # crop strength schedule
    if 'crop_strength' in df.columns:
        ax_crop = fig.add_subplot(gs[0, 1])
        ax_crop.plot(epochs, df['crop_strength'], color='tab:red')
        ax_crop.set_title('Maximum Cropping Strength')
        ax_crop.set_xlabel('Epoch')
        ax_crop.set_ylabel('Maximum Strength')
        ax_crop.set_ylim(0, max(df['crop_strength'].max() * 1.1, 0.1))
        ax_crop.grid(True, alpha=0.3)
    else:
        fig.add_subplot(gs[0, 1]).set_visible(False)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plots to {out_path}.")

    plt.close(fig)

# display a cross-section sample with red lines at each resonance energy
# tensor shape: [2, H, W] where channel 0 is data and channel 1 is the visibility mask
def display_tensor_with_targets(tensor, target, name):

    data = tensor[0].numpy()
    mask = tensor[1].numpy()

    plt.figure(figsize=(10, 6))

    plt.imshow(data, cmap='viridis', aspect='auto', origin='lower', interpolation='nearest')
    plt.colorbar()

    # overlay cropped pixels
    cropped_overlay = np.zeros((*data.shape, 4))
    cropped_overlay[mask == 0] = [0, 0, 0, 1]
    plt.imshow(cropped_overlay, aspect='auto', origin='lower', interpolation='nearest')

    n_energy = data.shape[0]
    for energy in target['energy']:
        y = energy * n_energy
        plt.axhline(y=y, color='red', linestyle=':')

    plt.title('Data (black = cropped)')

    plt.tight_layout()
    plt.savefig(f'out/tensor/{name}')
    plt.close()

# display a tensor of shape [H, W]
def display_tensor(tensor, name):

    plt.figure(figsize=(10, 6))
    plt.imshow(tensor.numpy(), cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.savefig(f'out/tensor/{name}')

# display an RGB image
def display_image(img, name):

    plt.figure(figsize=(10, 6))
    plt.imshow(img.permute(1, 2, 0).numpy(), aspect='auto')
    plt.axis('off')
    plt.savefig(f'out/image/{name}')