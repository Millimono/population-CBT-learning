# interpretability.py

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import os

def explain_prediction(pop, img, label, class_names, patch_size, device):
    """
    Pour une image donnée, montre quelles cellules s'activent
    et où elles se trouvent dans l'image.
    """
    img_t = img.unsqueeze(0).to(device)
    activated, z = pop.process_batch(img_t)
    activated = activated[0]  # (B,)

    # Classes des cellules actives
    active_classes = pop.proto_class[activated]
    active_classes = active_classes[active_classes >= 0]

    if len(active_classes) == 0:
        print("Aucune cellule active")
        return

    pred = torch.mode(active_classes).values.item()

    # Trouver les patches les plus activés
    # Recalculer les distances pour localisation
    patches_t   = pop.extract_patches_batch(img_t)         # (1, P, D)
    patches_std = pop.preprocess_patches(patches_t)
    protos      = pop.preprocess_patches(
        pop.prototypes.unsqueeze(0)).squeeze(0)

    P = patches_std.shape[1]
    D = pop.D
    ps = patch_size

    patches_sq = (patches_std**2).sum(dim=-1)
    protos_sq  = (protos**2).sum(dim=-1)
    dot        = torch.einsum("npd,bd->nbp", patches_std, protos)
    dists_sq   = (patches_sq.unsqueeze(1) +
                  protos_sq.view(1,-1,1) - 2*dot).clamp(min=0)

    # Top cellules actives par classe
    H = W = img.shape[-1]
    n_patches_w = W - ps + 1

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Image originale
    axes[0].imshow(img.numpy(), cmap="gray")
    axes[0].set_title(f"Image originale\nVraie classe : {class_names[label]}", fontsize=11)
    axes[0].axis("off")

    # Heatmap d'activation
    heatmap = torch.zeros(H, W, device=device)

    for cell_idx in activated.nonzero(as_tuple=True)[0][:50]:  # top 50 cellules
        cell_cls = pop.proto_class[cell_idx].item()
        if cell_cls < 0:
            continue

        # Patch le plus proche de cette cellule
        best_patch_idx = dists_sq[0, cell_idx].argmin().item()
        row = best_patch_idx // n_patches_w
        col = best_patch_idx  % n_patches_w

        # Ajouter à la heatmap
        heatmap[row:row+ps, col:col+ps] += 1.0

    heatmap = heatmap.cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)

    axes[1].imshow(img.numpy(), cmap="gray", alpha=0.5)
    axes[1].imshow(heatmap, cmap="hot", alpha=0.5)
    axes[1].set_title("Zones activées par les cellules B", fontsize=11)
    axes[1].axis("off")

    # Distribution des cellules actives par classe
    class_votes = [(pop.proto_class[activated] == c).sum().item()
                   for c in range(pop.num_classes)]
    colors = ["#F44336", "#2196F3"][:len(class_names)]
    axes[2].bar(class_names, class_votes, color=colors)
    axes[2].set_title(
        f"Vote des cellules actives\nPrédiction : {class_names[pred]}", fontsize=11)
    axes[2].set_ylabel("Nombre de cellules")
    axes[2].grid(True, axis="y", alpha=0.3)

    plt.suptitle(
        f"Explication de la décision — {'[OK] Correct' if pred == label else '[ERREUR] Incorrect'}",
        fontsize=13)
    plt.tight_layout()
    return fig


def plot_interpretability_examples(pop, val_images, val_labels,
                                    class_names, patch_size,
                                    device, save_path, n_examples=4):
    """
    Montre n_examples exemples d'explication par classe
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, all_axes = plt.subplots(
        n_examples, 3,
        figsize=(15, n_examples * 4)
    )

    # Prendre des exemples corrects et incorrects
    correct_idx   = []
    incorrect_idx = []

    preds = []
    for img in val_images:
        img_t      = img.unsqueeze(0).to(device)
        activated, _ = pop.process_batch(img_t)
        act        = activated[0]
        ac         = pop.proto_class[act]
        ac         = ac[ac >= 0]
        p = torch.mode(ac).values.item() if len(ac) > 0 else -1
        preds.append(p)

    for i, (p, l) in enumerate(zip(preds, val_labels)):
        if p == l and len(correct_idx) < n_examples // 2:
            correct_idx.append(i)
        if p != l and len(incorrect_idx) < n_examples // 2:
            incorrect_idx.append(i)

    selected = correct_idx + incorrect_idx

    for row, idx in enumerate(selected[:n_examples]):
        img   = val_images[idx]
        label = val_labels[idx]
        pred  = preds[idx]

        img_t       = img.unsqueeze(0).to(device)
        activated, _ = pop.process_batch(img_t)
        act         = activated[0]

        # Heatmap
        H = W = img.shape[-1]
        ps = patch_size
        n_patches_w = W - ps + 1

        patches_t   = pop.extract_patches_batch(img_t)
        patches_std = pop.preprocess_patches(patches_t)
        protos      = pop.preprocess_patches(
            pop.prototypes.unsqueeze(0)).squeeze(0)

        patches_sq = (patches_std**2).sum(dim=-1)
        protos_sq  = (protos**2).sum(dim=-1)
        dot        = torch.einsum("npd,bd->nbp", patches_std, protos)
        dists_sq   = (patches_sq.unsqueeze(1) +
                      protos_sq.view(1,-1,1) - 2*dot).clamp(min=0)

        heatmap = torch.zeros(H, W, device=device)
        for cell_idx in act.nonzero(as_tuple=True)[0][:50]:
            cell_cls = pop.proto_class[cell_idx].item()
            if cell_cls < 0:
                continue
            best_patch_idx = dists_sq[0, cell_idx].argmin().item()
            r = best_patch_idx // n_patches_w
            c = best_patch_idx  % n_patches_w
            heatmap[r:r+ps, c:c+ps] += 1.0

        heatmap = heatmap.cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)

        # Image originale
        all_axes[row][0].imshow(img.numpy(), cmap="gray")
        all_axes[row][0].set_title(
            f"Classe réelle : {class_names[label]}", fontsize=10)
        all_axes[row][0].axis("off")

        # Heatmap
        all_axes[row][1].imshow(img.numpy(), cmap="gray", alpha=0.5)
        all_axes[row][1].imshow(heatmap, cmap="hot", alpha=0.5)
        status = "[OK]" if pred == label else "[ERREUR]"
        all_axes[row][1].set_title(
            f"{status} Prédit : {class_names[pred]}", fontsize=10)
        all_axes[row][1].axis("off")

        # Votes
        class_votes = [(pop.proto_class[act] == c).sum().item()
                       for c in range(pop.num_classes)]
        colors = ["#F44336", "#2196F3"]
        all_axes[row][2].bar(class_names, class_votes, color=colors)
        all_axes[row][2].set_ylabel("Cellules actives")
        all_axes[row][2].grid(True, axis="y", alpha=0.3)

    plt.suptitle(
        "Interprétabilité — Explication des décisions (PopulationB)",
        fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[OK] Figure interprétabilité : {save_path}")