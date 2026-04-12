# ============================================================
# data.py — Chargement des données
# ============================================================

import os
import torch
import torchvision.transforms as T
from PIL import Image


def load_binary(root_dir, transform,
                classes=["Cancer", "Normal"],
                max_per_class=1924):
    images, labels = [], []
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(root_dir, cls)
        files = [f for f in os.listdir(cls_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:max_per_class]
        for f in files:
            img = Image.open(os.path.join(cls_dir, f))
            t   = transform(img)
            if t.dim() == 3:
                t = t.mean(0)  # → (H, W)
            images.append(t)
            labels.append(idx)
    return images, labels


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


def load_ddsm(train_dir, val_dir):
    transform_train, transform_val = get_transforms()
    train_images, train_labels = load_binary(train_dir, transform_train)
    val_images,   val_labels   = load_binary(val_dir,   transform_val)

    perm = torch.randperm(len(train_images))
    train_images = [train_images[i] for i in perm]
    train_labels = [train_labels[i] for i in perm]

    print(f"Train: {len(train_images)} | Val: {len(val_images)}")
    print(f"Shape: {train_images[0].shape}")
    return train_images, train_labels, val_images, val_labels