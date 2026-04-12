# ============================================================
# run.py — Point d'entrée unique
# ============================================================

import torch
import gc
from data  import load_ddsm
from train import run_experiment

# ============================================================
# CONFIG
# ============================================================
TRAIN_DIR   = "/content/PhD_AI_Grad_CAM_CAS_1_BAselines/miniddsm/train"
VAL_DIR     = "/content/PhD_AI_Grad_CAM_CAS_1_BAselines/miniddsm/val"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 2
EPOCHS      = 40
LR          = 0.1
NUM_CELLS   = 1600
PATCH_SIZE  = (5, 5)
THETA_INIT  = 0.5

# ============================================================
# LANCEMENT
# ============================================================
if __name__ == "__main__":

    # Libérer mémoire
    torch.cuda.empty_cache()
    gc.collect()
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