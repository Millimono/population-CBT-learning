# ============================================================
# data.py — Chargement cache 256×256
# ============================================================

# import os
# import torch

# CACHE_PATH = "/content/drive/MyDrive/CBIS-DDSM/cbis_cache_256.pt"

# def load_ddsm(train_dir=None, val_dir=None, img_size=256, use_mask=False):
#     """Charge depuis cache 256×256."""
#     if not os.path.exists(CACHE_PATH):
#         raise FileNotFoundError(f"Cache non trouvé : {CACHE_PATH}")
    
#     print(f"[OK] Chargement depuis cache : {CACHE_PATH}")
#     data = torch.load(CACHE_PATH)
    
#     train_images = data["train_images"]
#     train_labels = data["train_labels"]
#     val_images   = data["val_images"]
#     val_labels   = data["val_labels"]
    
#     # Mélanger train
#     perm = torch.randperm(len(train_images))
#     train_images = [train_images[i] for i in perm]
#     train_labels = [train_labels[i] for i in perm]
    
#     # Stats
#     n_train_cancer = sum(1 for l in train_labels if l == 0)
#     n_train_normal = sum(1 for l in train_labels if l == 1)
#     n_val_cancer   = sum(1 for l in val_labels   if l == 0)
#     n_val_normal   = sum(1 for l in val_labels   if l == 1)
    
#     print(f"Train: {len(train_images)} ({n_train_cancer} Cancer, {n_train_normal} Normal)")
#     print(f"Val  : {len(val_images)} ({n_val_cancer} Cancer, {n_val_normal} Normal)")
#     print(f"Shape: {train_images[0].shape}")
    
#     return train_images, train_labels, val_images, val_labels

# load_cbis_ddsm = load_ddsm

# ============================================================
# data.py — Chargement cache MiniDDSM 128×128
# ============================================================

import os
import torch

CACHE_PATH = "/content/drive/MyDrive/MiniDDSM/miniddsm_cache_128.pt"

def load_ddsm(train_dir=None, val_dir=None, img_size=128, use_mask=False):
    """Charge depuis cache MiniDDSM 128×128."""
    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(f"Cache non trouvé : {CACHE_PATH}")
    
    print(f"[OK] Chargement depuis cache : {CACHE_PATH}")
    data = torch.load(CACHE_PATH)
    
    train_images = data["train_images"]
    train_labels = data["train_labels"]
    val_images   = data["val_images"]
    val_labels   = data["val_labels"]
    
    # Mélanger train
    perm = torch.randperm(len(train_images))
    train_images = [train_images[i] for i in perm]
    train_labels = [train_labels[i] for i in perm]
    
    # Stats
    n_train_cancer = sum(1 for l in train_labels if l == 0)
    n_train_normal = sum(1 for l in train_labels if l == 1)
    n_val_cancer   = sum(1 for l in val_labels   if l == 0)
    n_val_normal   = sum(1 for l in val_labels   if l == 1)
    
    print(f"Train: {len(train_images)} ({n_train_cancer} Cancer, {n_train_normal} Normal)")
    print(f"Val  : {len(val_images)} ({n_val_cancer} Cancer, {n_val_normal} Normal)")
    print(f"Shape: {train_images[0].shape}")
    
    return train_images, train_labels, val_images, val_labels

load_cbis_ddsm = load_ddsm