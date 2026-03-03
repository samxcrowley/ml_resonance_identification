import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_results(path, title, out_path, subtitle=None):

    df = pd.read_csv(path)

    epochs = df["epoch"]

    fig = plt.figure(figsize=(11, 9))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    if subtitle:
        fig.text(0.5, 0.955, subtitle, ha="center", fontsize=10, color="gray")
        
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
    ax_prec.set_ylim(0, 1)
    ax_prec.set_title("Precision")
    ax_prec.set_xlabel("Epoch")
    ax_prec.set_ylabel("Precision")
    ax_prec.grid(True, alpha=0.3)

    # recall
    ax_rec = fig.add_subplot(gs[1, 1])
    ax_rec.plot(epochs, df["val_recall"], color="tab:green", label="Recall")
    ax_rec.set_ylim(0, 1)
    ax_rec.set_title("Recall")
    ax_rec.set_xlabel("Epoch")
    ax_rec.set_ylabel("Recall")
    ax_rec.grid(True, alpha=0.3)

    # blank fourth cell
    fig.add_subplot(gs[0, 1]).set_visible(False)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plots to {out_path}.")

    plt.close(fig)