# ============================================================
# data.py — CBIS-DDSM images complètes
# Préprocessing : crop bordures noires + windowing + resize 128×128
# ============================================================

import os
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image


# ============================================================
# Préprocessing mammographique
# ============================================================
def preprocess_mammogram(img_pil):
    """
    1. Convertir en niveaux de gris
    2. Crop des bordures noires (bounding box des pixels non-noirs)
    3. Windowing : étirer le contraste sur les percentiles 1%-99%
    """
    # Convertir en niveaux de gris
    img = np.array(img_pil.convert("L"))

    # 1. Crop bordures noires
    mask = img > 10
    if mask.any():
        coords  = np.argwhere(mask)
        y0, x0  = coords.min(axis=0)
        y1, x1  = coords.max(axis=0) + 1
        img     = img[y0:y1, x0:x1]

    # 2. Windowing — étirer le contraste
    p1, p99 = np.percentile(img, [1, 99])
    if p99 > p1:
        img = np.clip(img, p1, p99)
        img = ((img - p1) / (p99 - p1) * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    return Image.fromarray(img)


# ============================================================
# Chargement des images
# ============================================================
def load_binary(root_dir, transform,
                classes=["Cancer", "Normal"],
                max_per_class=None):
    """
    Charge les images depuis root_dir/{Cancer,Normal}/.
    Applique preprocess_mammogram avant la transformation.
    """
    images, labels = [], []
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(root_dir, cls)
        files   = sorted([
            f for f in os.listdir(cls_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        if max_per_class:
            files = files[:max_per_class]

        for f in files:
            img = Image.open(os.path.join(cls_dir, f))
            img = preprocess_mammogram(img)   # crop + windowing
            t   = transform(img)
            if t.dim() == 3:
                t = t.mean(0)
            images.append(t)
            labels.append(idx)
    return images, labels


# ============================================================
# Transformations
# ============================================================
def get_transforms(img_size=128):
    transform_train = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ToTensor(),
    ])
    transform_val = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
    ])
    return transform_train, transform_val


# ============================================================
# Chargeur principal — CBIS-DDSM
# ============================================================
def load_ddsm(train_dir, val_dir, img_size=128, use_mask=False):
    """
    Charge CBIS-DDSM avec images complètes prétraitées.
    
    use_mask : ignoré (gardé pour compatibilité)
    img_size : taille de redimensionnement (par défaut 128×128)
    """
    transform_train, transform_val = get_transforms(img_size=img_size)
    train_images, train_labels = load_binary(train_dir, transform_train)
    val_images,   val_labels   = load_binary(val_dir,   transform_val)

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


# Alias pour compatibilité
load_cbis_ddsm = load_ddsm


# ============================================================
# Chargeur générique (datasets non médicaux)
# ============================================================
def load_generic(train_dir, val_dir, classes,
                 max_per_class=None, img_size=64):
    """
    Chargeur générique pour datasets non médicaux.
    Pas de prétraitement mammographique.
    """
    transform_train = T.Compose([
        T.Resize((img_size, img_size)),
        T.Grayscale(num_output_channels=1),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ToTensor(),
    ])
    transform_val = T.Compose([
        T.Resize((img_size, img_size)),
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
    ])

    # Pour ce chargeur, pas de preprocess_mammogram
    train_images, train_labels = [], []
    val_images,   val_labels   = [], []

    for split_dir, imgs, labels, transform in [
        (train_dir, train_images, train_labels, transform_train),
        (val_dir,   val_images,   val_labels,   transform_val)
    ]:
        for idx, cls in enumerate(classes):
            cls_dir = os.path.join(split_dir, cls)
            files = sorted([f for f in os.listdir(cls_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if max_per_class:
                files = files[:max_per_class]
            for f in files:
                img = Image.open(os.path.join(cls_dir, f))
                t   = transform(img)
                if t.dim() == 3:
                    t = t.mean(0)
                imgs.append(t)
                labels.append(idx)

    perm = torch.randperm(len(train_images))
    train_images = [train_images[i] for i in perm]
    train_labels = [train_labels[i] for i in perm]

    print(f"Train: {len(train_images)} | Val: {len(val_images)}")
    print(f"Shape: {train_images[0].shape}")
    print(f"Classes : {classes}")
    return train_images, train_labels, val_images, val_labels