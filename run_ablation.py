# ============================================================
# run_ablation.py — Ablation study complet
# ============================================================

import torch
import gc
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data  import load_ddsm
from train import run_experiment
from run   import set_seed, TRAIN_DIR, VAL_DIR, DEVICE, NUM_CLASSES, LR

ABLATION_SEED   = 42
ABLATION_EPOCHS = 10
BEST_CELLS      = 6400

# ============================================================
if __name__ == "__main__":
    print("v2")
    torch.cuda.empty_cache()
    gc.collect()
    os.makedirs("figs", exist_ok=True)

    set_seed(ABLATION_SEED)
    train_images, train_labels, val_images, val_labels = load_ddsm(
        TRAIN_DIR, VAL_DIR
    )

    results = []

    # ============================================================
    # ABLATION K
    # ============================================================
    print("\n=== ABLATION K ===")
    for K in [1, 2, 3, 5, 10, 20]:
        set_seed(ABLATION_SEED)
        acc, _, _, _ = run_experiment(
            train_images, train_labels, val_images, val_labels,
            name        = f"K={K}",
            num_classes = NUM_CLASSES,
            epochs      = ABLATION_EPOCHS,
            lr          = LR,
            num_cells   = BEST_CELLS,
            patch_size  = (5, 5),
            theta_init  = 0.5,
            device      = DEVICE,
            K           = K
        )
        print(f"  K={K} → {acc:.4f}")
        results.append({"param": "K", "value": K, "acc": acc})

    # ============================================================
    # ABLATION num_cells
    # ============================================================
    print("\n=== ABLATION num_cells ===")
    for num_cells in [200, 400, 800, 1600, 3200, 6400, 8000]:
        set_seed(ABLATION_SEED)
        acc, _, _, _ = run_experiment(
            train_images, train_labels, val_images, val_labels,
            name        = f"num_cells={num_cells}",
            num_classes = NUM_CLASSES,
            epochs      = ABLATION_EPOCHS,
            lr          = LR,
            num_cells   = num_cells,
            patch_size  = (5, 5),
            theta_init  = 0.5,
            device      = DEVICE,
            K           = 1
        )
        print(f"  num_cells={num_cells} → {acc:.4f}")
        results.append({"param": "num_cells", "value": num_cells, "acc": acc})

    # ============================================================
    # ABLATION patch_size
    # ============================================================
    print("\n=== ABLATION patch_size ===")
    for ps in [3, 5, 7, 9, 11]:
        set_seed(ABLATION_SEED)
        acc, _, _, _ = run_experiment(
            train_images, train_labels, val_images, val_labels,
            name        = f"patch={ps}x{ps}",
            num_classes = NUM_CLASSES,
            epochs      = ABLATION_EPOCHS,
            lr          = LR,
            num_cells   = BEST_CELLS,
            patch_size  = (ps, ps),
            theta_init  = 0.5,
            device      = DEVICE,
            K           = 1
        )
        print(f"  patch={ps}x{ps} → {acc:.4f}")
        results.append({"param": "patch_size", "value": ps, "acc": acc})

    # ============================================================
    # ABLATION theta
    # ============================================================
    print("\n=== ABLATION theta ===")
    for theta in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        set_seed(ABLATION_SEED)
        acc, _, _, _ = run_experiment(
            train_images, train_labels, val_images, val_labels,
            name        = f"theta={theta}",
            num_classes = NUM_CLASSES,
            epochs      = ABLATION_EPOCHS,
            lr          = LR,
            num_cells   = BEST_CELLS,
            patch_size  = (5, 5),
            theta_init  = theta,
            device      = DEVICE,
            K           = 1
        )
        print(f"  theta={theta} → {acc:.4f}")
        results.append({"param": "theta", "value": theta, "acc": acc})

    # ============================================================
    # ABLATION LR
    # ============================================================
    print("\n=== ABLATION LR ===")
    for lr in [0.05, 0.1, 0.2, 0.3]:
        set_seed(ABLATION_SEED)
        acc, _, _, _ = run_experiment(
            train_images, train_labels, val_images, val_labels,
            name        = f"lr={lr}",
            num_classes = NUM_CLASSES,
            epochs      = ABLATION_EPOCHS,
            lr          = lr,
            num_cells   = BEST_CELLS,
            patch_size  = (5, 5),
            theta_init  = 0.5,
            device      = DEVICE,
            K           = 1
        )
        print(f"  lr={lr} → {acc:.4f}")
        results.append({"param": "lr", "value": lr, "acc": acc})

    # ============================================================
    # Résumé + CSV + Figure
    # ============================================================
    df = pd.DataFrame(results)
    print(f"\n{'='*40}")
    print("RÉSUMÉ ABLATION")
    print(f"{'='*40}")
    for param in df["param"].unique():
        sub = df[df["param"] == param].sort_values("value")
        print(f"\n{param}:")
        for _, row in sub.iterrows():
            marker = " ← best" if row["acc"] == sub["acc"].max() else ""
            print(f"  {str(row['value']):>8} → {row['acc']:.4f}{marker}")

    df.to_csv("figs/ablation_results.csv", index=False)
    print("\n[OK] figs/ablation_results.csv")

    # ── Figure résumé ─────────────────────────────────────────
    params = df["param"].unique()
    fig, axes = plt.subplots(1, len(params), figsize=(5 * len(params), 4))

    for ax, param in zip(axes, params):
        sub      = df[df["param"] == param].sort_values("value")
        best_val = sub.loc[sub["acc"].idxmax(), "value"]
        colors   = ["#F44336" if v == best_val else "#90CAF9"
                    for v in sub["value"]]
        ax.bar(sub["value"].astype(str), sub["acc"], color=colors)
        ax.set_title(param, fontsize=12)
        ax.set_xlabel("Valeur")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.4, 0.85)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle(
        f"Ablation study — MiniDDSM "
        f"({ABLATION_EPOCHS} epochs, seed={ABLATION_SEED})",
        fontsize=13)
    plt.tight_layout()
    plt.savefig("figs/ablation_study.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("[OK] figs/ablation_study.png")