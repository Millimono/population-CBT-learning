# ============================================================
# run.py — Point d'entrée unique
# ============================================================

import torch
import gc
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from data  import load_ddsm
from train import run_experiment

# ============================================================
# CONFIG
# ============================================================
TRAIN_DIR   = "data/miniddsm/train"
VAL_DIR     = "data/miniddsm/val"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 2
EPOCHS      = 40
LR          = 0.1
NUM_CELLS   = 1600
PATCH_SIZE  = (5, 5)
THETA_INIT  = 0.5
SEED        = 42

# ============================================================
def set_seed(seed=42):
    import random, numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def plot_learning_curve(history, name, oracle_acc, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs  = range(1, len(history) + 1)

    ax.plot(epochs, history, color="#2196F3", linewidth=2, label="Notre méthode")
    ax.axhline(y=oracle_acc, color="red", linestyle="--",
               linewidth=1.5, label=f"Oracle kNN ({oracle_acc:.2%})")
    ax.axhline(y=max(history), color="#2196F3", linestyle=":",
               linewidth=1.5, label=f"Best ({max(history):.2%})")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy (validation)", fontsize=12)
    ax.set_title(name, fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0.3, 1.0)
    ax.set_xlim(1, len(history))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"✅ Figure sauvegardée : {save_path}")


def plot_prototypes(pop, class_names, patch_size, save_path, n_proto=12):
    fig, axes = plt.subplots(
        len(class_names), n_proto,
        figsize=(n_proto * 1.2, len(class_names) * 1.4)
    )

    for c, cls_name in enumerate(class_names):
        mask     = (pop.proto_class == c)
        protos_c = pop.prototypes[mask].cpu().detach()
        norms    = protos_c.norm(dim=1)
        top_idx  = norms.argsort(descending=True)[:n_proto]
        protos_c = protos_c[top_idx]

        for j in range(n_proto):
            ax = axes[c][j] if len(class_names) > 1 else axes[j]
            if j < len(protos_c):
                patch = protos_c[j].reshape(patch_size, patch_size).numpy()
                patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-6)
                ax.imshow(patch, cmap="viridis", vmin=0, vmax=1)
            ax.axis("off")

        axes[c][0].set_ylabel(cls_name, fontsize=11,
                              rotation=90, labelpad=10, va="center")

    plt.suptitle("Prototypes appris par classe", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"✅ Figure sauvegardée : {save_path}")


# ============================================================
if __name__ == "__main__":

    set_seed(SEED)
    torch.cuda.empty_cache()
    gc.collect()
    os.makedirs("figs", exist_ok=True)
    print(f"Device: {DEVICE}")

    # Charger les données
    train_images, train_labels, val_images, val_labels = load_ddsm(
        TRAIN_DIR, VAL_DIR
    )

    # Entraîner
    acc, pop, trainer, history = run_experiment(
        train_images, train_labels,
        val_images,   val_labels,
        name        = "MiniDDSM — K=1, patch=5x5, theta=0.5",
        num_classes = NUM_CLASSES,
        epochs      = EPOCHS,
        lr          = LR,
        num_cells   = NUM_CELLS,
        patch_size  = PATCH_SIZE,
        theta_init  = THETA_INIT,
        device      = DEVICE
    )

    print(f"\nRésultat final : {acc:.4f}")

    # ============================================================
    # Générer les figures
    # ============================================================

    # Figure 1 — Courbe d'apprentissage
    plot_learning_curve(
        history    = history,
        name       = "MiniDDSM — Cancer vs Normal",
        oracle_acc = 0.7779,
        save_path  = "figs/learning_curve_ddsm.png"
    )

    # Figure 2 — Prototypes par classe
    plot_prototypes(
        pop         = pop,
        class_names = ["Cancer", "Normal"],
        patch_size  = PATCH_SIZE[0],
        save_path   = "figs/prototypes_ddsm.png"
    )

    print("\n✅ Toutes les figures générées dans figs/")
    print(f"{'='*50}")
    print(f"RÉSUMÉ FINAL")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Figures  : figs/")
    print(f"{'='*50}")