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


import os
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from skimage.filters import threshold_otsu

def apply_breast_mask(tensor):
    img_np = tensor.numpy()
    if img_np.max() == 0:
        return tensor
    thresh = threshold_otsu(img_np)
    mask   = (img_np > thresh * 0.5).astype(np.float32)
    return tensor * torch.from_numpy(mask)


def load_binary(root_dir, transform,
                classes=["Cancer", "Normal"],
                max_per_class=1924,
                use_mask=False):        # ← nouveau paramètre
    images, labels = [], []
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(root_dir, cls)
        files   = [f for f in os.listdir(cls_dir)
                   if f.lower().endswith(
                       ('.png', '.jpg', '.jpeg'))][:max_per_class]
        for f in files:
            img = Image.open(os.path.join(cls_dir, f))
            t   = transform(img)
            if t.dim() == 3:
                t = t.mean(0)
            # ── Masque mammaire optionnel ─────────────────
            if use_mask:
                t = apply_breast_mask(t)
            # ─────────────────────────────────────────────
            images.append(t)
            labels.append(idx)
    return images, labels


def load_ddsm(train_dir, val_dir, use_mask=True):    # ← médical = True par défaut
    transform_train, transform_val = get_transforms()
    train_images, train_labels = load_binary(
        train_dir, transform_train, use_mask=use_mask)
    val_images, val_labels     = load_binary(
        val_dir, transform_val, use_mask=use_mask)

    perm = torch.randperm(len(train_images))
    train_images = [train_images[i] for i in perm]
    train_labels = [train_labels[i] for i in perm]

    print(f"Train: {len(train_images)} | Val: {len(val_images)}")
    print(f"Shape: {train_images[0].shape}")
    print(f"Masque Otsu : {'activé' if use_mask else 'désactivé'}")
    return train_images, train_labels, val_images, val_labels


def load_generic(train_dir, val_dir,
                 classes,
                 max_per_class=None,
                 img_size=64):
    """
    Chargeur générique pour n'importe quel dataset.
    Pas de masque — pour données non médicales.
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

    train_images, train_labels = load_binary(
        train_dir, transform_train,
        classes       = classes,
        max_per_class = max_per_class or 9999,
        use_mask      = False              # ← pas de masque
    )
    val_images, val_labels = load_binary(
        val_dir, transform_val,
        classes       = classes,
        max_per_class = max_per_class or 9999,
        use_mask      = False
    )

    perm = torch.randperm(len(train_images))
    train_images = [train_images[i] for i in perm]
    train_labels = [train_labels[i] for i in perm]

    print(f"Train: {len(train_images)} | Val: {len(val_images)}")
    print(f"Shape: {train_images[0].shape}")
    print(f"Classes : {classes}")
    return train_images, train_labels, val_images, val_labels


def get_transforms():
    transform_train = T.Compose([
        T.Resize((64, 64)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ToTensor(),
    ])
    transform_val = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
    ])
    return transform_train, transform_val

