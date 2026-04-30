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

# import os
# import torch

# CACHE_PATH = "/content/drive/MyDrive/MiniDDSM/miniddsm_cache_128.pt"

# def load_ddsm(train_dir=None, val_dir=None, img_size=128, use_mask=False):
#     """Charge depuis cache MiniDDSM 128×128."""
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

import os
import torch

CACHE_PATH = "/content/drive/MyDrive/MiniDDSM/miniddsm_cache_128.pt"

def load_ddsm(train_dir=None, val_dir=None, img_size=128, use_mask=False):
    """Charge depuis cache MiniDDSM 128×128 avec initialisation équilibrée."""
    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(f"Cache non trouvé : {CACHE_PATH}")
    
    print(f"[OK] Chargement depuis cache : {CACHE_PATH}")
    data = torch.load(CACHE_PATH)
    
    train_images = data["train_images"]
    train_labels = data["train_labels"]
    val_images   = data["val_images"]
    val_labels   = data["val_labels"]
    
    # ✅ NOUVEAU : Séparer par classe
    cancer_idx = [i for i, l in enumerate(train_labels) if l == 0]
    normal_idx = [i for i, l in enumerate(train_labels) if l == 1]
    
    # ✅ NOUVEAU : Mélanger CHAQUE classe séparément
    cancer_perm = torch.randperm(len(cancer_idx))
    normal_perm = torch.randperm(len(normal_idx))
    
    cancer_shuffled = [cancer_idx[i] for i in cancer_perm]
    normal_shuffled = [normal_idx[i] for i in normal_perm]
    
    # ✅ NOUVEAU : Entrelacer Cancer/Normal pour alternance parfaite
    train_images_balanced = []
    train_labels_balanced = []
    
    max_len = max(len(cancer_shuffled), len(normal_shuffled))
    for i in range(max_len):
        if i < len(cancer_shuffled):
            train_images_balanced.append(train_images[cancer_shuffled[i]])
            train_labels_balanced.append(0)
        if i < len(normal_shuffled):
            train_images_balanced.append(train_images[normal_shuffled[i]])
            train_labels_balanced.append(1)
    
    train_images = train_images_balanced
    train_labels = train_labels_balanced
    
    # Stats
    n_train_cancer = sum(1 for l in train_labels if l == 0)
    n_train_normal = sum(1 for l in train_labels if l == 1)
    n_val_cancer   = sum(1 for l in val_labels if l == 0)
    n_val_normal   = sum(1 for l in val_labels if l == 1)
    
    print(f"Train: {len(train_images)} ({n_train_cancer} Cancer, {n_train_normal} Normal) [entrelacé]")
    print(f"Val  : {len(val_images)} ({n_val_cancer} Cancer, {n_val_normal} Normal)")
    print(f"Shape: {train_images[0].shape}")
    
    return train_images, train_labels, val_images, val_labels

load_cbis_ddsm = load_ddsm
