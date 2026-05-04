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

    # draw resonance lines
    n_energy = data.shape[0]
    for n in range(target['class'].shape[0]):
        if target['class'][n, 1] == 1.0:
            y = target['energy'][n] * n_energy
            plt.axhline(y=y, color='red', linestyle=':')

    plt.title('Cross Section')

    plt.tight_layout()
    plt.savefig(f'out/tensor/{name}')
    plt.close()

# display tensor as a heatmap with channels as columns
# tensor shape: [n_y, n_channels]
def display_tensor(tensor, name):

    mask = None
    if len(tensor.shape) == 3:
        mask = tensor[1].numpy()
        tensor = tensor[0]

    grid = tensor.numpy()

    plt.figure(figsize=(12, 6))
    plt.imshow(grid, cmap='viridis', origin='lower', interpolation='nearest', aspect='auto')
    
    if mask is not None:
        black = np.zeros((*mask.shape, 4))
        black[..., 3] = 1.0 - mask
        plt.imshow(black, origin='lower', interpolation='nearest', aspect='auto')

    plt.colorbar(label='log10(cross section)')
    plt.xlabel('Channel')
    plt.ylabel('Energy bin')
    plt.title('Cross Section (all channels)')

    plt.tight_layout()
    plt.savefig(f'out/tensor/{name}')
    plt.close()

# display tensor as a grid of line plots, one per channel
# tensor shape: [n_y, n_channels] where n_channels = n_pp_combos * n_angles
def display_tensor_grid(tensor, name, n_pp_combos=None, n_angles=None):

    grid = tensor.numpy()
    n_y = grid.shape[0]

    fig, axes = plt.subplots(n_pp_combos, n_angles, figsize=(3 * n_angles, 2 * n_pp_combos),
                             sharex=True, sharey=True)

    for pp_idx in range(n_pp_combos):
        for ang_idx in range(n_angles):

            ch_idx = pp_idx * n_angles + ang_idx
            ax = axes[pp_idx, ang_idx]
            ax.plot(np.arange(n_y), grid[:, ch_idx], linewidth=0.8)

            if pp_idx == 0:
                ax.set_title(f'ang {ang_idx}', fontsize=8)
            if ang_idx == 0:
                ax.set_ylabel(f'pp {pp_idx}', fontsize=8)

            ax.tick_params(labelsize=6)

    fig.suptitle('Cross Section per Channel', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'out/tensor/{name}', dpi=150)
    plt.close()

# display an RGB image
def display_image(img, name):

    plt.figure(figsize=(10, 6))
    plt.imshow(img.permute(1, 2, 0).numpy(), aspect='auto')
    plt.axis('off')
    plt.savefig(f'out/image/{name}')