# ============================================================
# data.py — Chargement des données
# ============================================================

# import os
# import torch
# import torchvision.transforms as T
# from PIL import Image


# def load_binary(root_dir, transform,
#                 classes=["Cancer", "Normal"],
#                 max_per_class=1924):
#     images, labels = [], []
#     for idx, cls in enumerate(classes):
#         cls_dir = os.path.join(root_dir, cls)
#         files = [f for f in os.listdir(cls_dir)
#                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:max_per_class]
#         for f in files:
#             img = Image.open(os.path.join(cls_dir, f))
#             t   = transform(img)
#             if t.dim() == 3:
#                 t = t.mean(0)  # → (H, W)
#             images.append(t)
#             labels.append(idx)
#     return images, labels


# def get_transforms():
#     transform_train = T.Compose([
#         T.Resize((64, 64)),
#         T.RandomHorizontalFlip(p=0.5),
#         T.RandomRotation(degrees=15),
#         T.ToTensor(),
#     ])
#     transform_val = T.Compose([
#         T.Resize((64, 64)),
#         T.ToTensor(),
#     ])
#     return transform_train, transform_val


# def load_ddsm(train_dir, val_dir):
#     transform_train, transform_val = get_transforms()
#     train_images, train_labels = load_binary(train_dir, transform_train)
#     val_images,   val_labels   = load_binary(val_dir,   transform_val)

#     perm = torch.randperm(len(train_images))
#     train_images = [train_images[i] for i in perm]
#     train_labels = [train_labels[i] for i in perm]

#     print(f"Train: {len(train_images)} | Val: {len(val_images)}")
#     print(f"Shape: {train_images[0].shape}")
#     return train_images, train_labels, val_images, val_labels


# ============================================================
# data.py — version CBIS-DDSM (sans masque Otsu)
# ============================================================

# import os
# import torch
# import numpy as np
# import torchvision.transforms as T
# from PIL import Image


# def load_binary(root_dir, transform,
#                 classes=["Cancer", "Normal"],
#                 max_per_class=None):
#     """
#     Charge les images depuis root_dir/{Cancer,Normal}/.
#     Pas de masque — les images CBIS-DDSM sont déjà des ROI cropped.
#     """
#     images, labels = [], []
#     for idx, cls in enumerate(classes):
#         cls_dir = os.path.join(root_dir, cls)
#         files   = sorted([
#             f for f in os.listdir(cls_dir)
#             if f.lower().endswith(('.png', '.jpg', '.jpeg'))
#         ])
#         if max_per_class:
#             files = files[:max_per_class]
#         for f in files:
#             img = Image.open(os.path.join(cls_dir, f))
#             t   = transform(img)
#             if t.dim() == 3:
#                 t = t.mean(0)
#             images.append(t)
#             labels.append(idx)
#     return images, labels


# def get_transforms():
#     transform_train = T.Compose([
#         T.Resize((64, 64)),
#         T.RandomHorizontalFlip(p=0.5),
#         T.RandomRotation(degrees=15),
#         T.ToTensor(),
#     ])
#     transform_val = T.Compose([
#         T.Resize((64, 64)),
#         T.ToTensor(),
#     ])
#     return transform_train, transform_val


# def load_cbis_ddsm(train_dir, val_dir):
#     """
#     Charge CBIS-DDSM ROI cropped — Cancer (Malignant) vs Normal (Benign).
#     """
#     transform_train, transform_val = get_transforms()
#     train_images, train_labels = load_binary(train_dir, transform_train)
#     val_images,   val_labels   = load_binary(val_dir,   transform_val)

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


# # Garder load_ddsm pour compatibilité MiniDDSM
# def load_ddsm(train_dir, val_dir, use_mask=False):
#     """Alias générique — pas de masque par défaut."""
#     return load_cbis_ddsm(train_dir, val_dir)


# ============================================================
# data.py — Chargement depuis cache pré-traité
# ============================================================

import os
import torch
import torchvision.transforms as T


CACHE_PATH = "/content/drive/MyDrive/CBIS-DDSM/cbis_cache_128.pt"


def load_ddsm(train_dir=None, val_dir=None, img_size=128, use_mask=False):
    """
    Charge depuis le cache pré-traité.
    train_dir/val_dir ignorés — les chemins sont dans le cache.
    """
    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(
            f"Cache non trouvé : {CACHE_PATH}\n"
            "Lance d'abord la cellule de préprocessing dans Colab."
        )
    
    print(f"[OK] Chargement depuis cache : {CACHE_PATH}")
    data = torch.load(CACHE_PATH)
    
    train_images = data["train_images"]
    train_labels = data["train_labels"]
    val_images   = data["val_images"]
    val_labels   = data["val_labels"]
    
    # Mélanger train (différent à chaque seed grâce au generator)
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


# Aliases pour compatibilité
load_cbis_ddsm = load_ddsm