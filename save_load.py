
# ============================================================
# save_load.py — Sauvegarde, chargement et prédiction interprétable
# ============================================================

import torch
import numpy as np
import os
import matplotlib.pyplot as plt


def save_model(pop, path="populationb_model.pt"):
    """
    Sauvegarde le modèle PopulationB.
    """
    state = {
        "prototypes":   pop.prototypes.cpu(),
        "proto_class":  pop.proto_class.cpu(),
        "class_counts": pop.class_counts.cpu(),
        "num_cells":    pop.B,
        "patch_size":   pop.patch_size,
        "theta_init":   pop.theta_init,
        "beta":         pop.beta,
        "num_classes":  pop.num_classes,
        "K":            pop.K,
    }
    torch.save(state, path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"[OK] Modèle sauvegardé : {path} ({size_mb:.2f} MB)")


def load_model(path="populationb_model.pt", device="cpu"):
    """
    Charge un modèle PopulationB depuis un fichier .pt
    """
    from model import PopulationBFastExact, TrainerFastExact

    state = torch.load(path, map_location=device)

    pop = PopulationBFastExact(
        num_cells  = state["num_cells"],
        patch_size = state["patch_size"],
        theta_init = state["theta_init"],
        beta       = state["beta"],
        num_classes= state["num_classes"],
        K          = state["K"],
        device     = device
    )

    pop.prototypes   = state["prototypes"].to(device)
    pop.proto_class  = state["proto_class"].to(device)
    pop.class_counts = state["class_counts"].to(device)

    trainer = TrainerFastExact(
        population  = pop,
        num_classes = state["num_classes"],
        device      = device
    )

    print(f"[OK] Modèle chargé : {path}")
    print(f"     Prototypes assignés : "
          f"{(pop.proto_class >= 0).sum().item()}/{pop.B}")
    return pop, trainer


def predict_and_explain(pop, img_tensor,
                        class_names=["Cancer", "Normal"],
                        device="cpu",
                        save_path=None,
                        true_label=None):
    """
    Prédit la classe d'une image ET génère l'explication visuelle.

    Paramètres :
        pop         : modèle PopulationBFastExact chargé
        img_tensor  : tensor (H, W) — image prétraitée
        class_names : liste des noms de classes
        device      : "cpu" ou "cuda"
        save_path   : chemin pour sauvegarder la figure (None = afficher)
        true_label  : label réel si connu (None = inconnu)

    Retourne :
        pred   : classe prédite (int)
        votes  : scores de vote par classe (tensor)
        heatmap: carte d'activation (numpy array)
    """
    pop.prototypes   = pop.prototypes.to(device)
    pop.proto_class  = pop.proto_class.to(device)
    pop.class_counts = pop.class_counts.to(device)

    img_t = img_tensor.unsqueeze(0).to(device)

    # ── Étape 1 : activation des prototypes ──────────────────
    activated, z = pop.process_batch(img_t)
    act = activated[0]  # (B,) booléen

    valid = act & (pop.proto_class >= 0)

    # ── Étape 2 : vote pondéré ────────────────────────────────
    weights, freq = pop.get_vote_weights()
    votes = torch.zeros(pop.num_classes, device=device)

    if valid.any():
        active_freq    = freq[valid]
        active_weights = weights[valid]
        votes = (active_freq * active_weights.unsqueeze(1)).sum(dim=0)

    if votes.sum() == 0:
        # Fallback mode
        active_classes = pop.proto_class[valid]
        pred = torch.mode(active_classes).values.item() \
               if len(active_classes) > 0 else 0
    else:
        pred = votes.argmax().item()

    # ── Étape 3 : heatmap des zones responsables ──────────────
    H = W = img_tensor.shape[-1]
    ps = pop.patch_size[0]
    n_patches_w = W - ps + 1

    patches_t   = pop.extract_patches_batch(img_t)
    patches_std = pop.preprocess_patches(patches_t)
    protos      = pop.preprocess_patches(
                      pop.prototypes.unsqueeze(0)).squeeze(0)

    patches_sq = (patches_std ** 2).sum(dim=-1)
    protos_sq  = (protos ** 2).sum(dim=-1)
    dot        = torch.einsum("npd,bd->nbp", patches_std, protos)
    dists_sq   = (patches_sq.unsqueeze(1) +
                  protos_sq.view(1, -1, 1) - 2 * dot).clamp(min=0)

    # Heatmap pondérée par exclusivité — zones les plus discriminantes
    heatmap = torch.zeros(H, W, device=device)

    for cell_idx in valid.nonzero(as_tuple=True)[0]:
        cell_cls = pop.proto_class[cell_idx].item()
        if cell_cls != pred:
            continue  # on ne montre que les prototypes qui ont voté pour pred

        # Poids d'exclusivité du prototype
        w_i = weights[cell_idx].item()

        # Localisation du patch déclencheur
        best_patch_idx = dists_sq[0, cell_idx].argmin().item()
        r = best_patch_idx // n_patches_w
        c = best_patch_idx  % n_patches_w
        heatmap[r:r+ps, c:c+ps] += w_i  # pondéré par exclusivité

    heatmap_np = heatmap.cpu().numpy()
    if heatmap_np.max() > 0:
        heatmap_np = (heatmap_np - heatmap_np.min()) / \
                     (heatmap_np.max() - heatmap_np.min() + 1e-6)

    # ── Étape 4 : figure ─────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Colonne 1 — image originale
    axes[0].imshow(img_tensor.cpu().numpy(), cmap="gray")
    if true_label is not None:
        axes[0].set_title(
            f"Image originale\nClasse réelle : {class_names[true_label]}",
            fontsize=11)
    else:
        axes[0].set_title("Image originale", fontsize=11)
    axes[0].axis("off")

    # Colonne 2 — heatmap des zones responsables
    axes[1].imshow(img_tensor.cpu().numpy(), cmap="gray", alpha=0.6)
    axes[1].imshow(heatmap_np, cmap="hot", alpha=0.5)
    n_active = valid.sum().item()
    axes[1].set_title(
        f"Zones responsables de la décision\n"
        f"{n_active} prototypes actifs — pondérés par exclusivité",
        fontsize=10)
    axes[1].axis("off")

    # Colonne 3 — votes pondérés par classe
    vote_vals = votes.cpu().detach().numpy()
    colors    = ["#F44336" if i == pred else "#90CAF9"
                 for i in range(pop.num_classes)]
    bars = axes[2].bar(class_names, vote_vals, color=colors)
    axes[2].set_title(
        f"Vote pondéré par classe\nPrédiction : {class_names[pred]}",
        fontsize=11)
    axes[2].set_ylabel("Score de vote (Σ fᵢ · wᵢ)")
    axes[2].grid(True, axis="y", alpha=0.3)

    # Annoter les barres
    for bar, val in zip(bars, vote_vals):
        axes[2].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.001,
                     f"{val:.3f}",
                     ha="center", va="bottom", fontsize=10)

    # Titre global
    if true_label is not None:
        status = "Correct" if pred == true_label else "Incorrect"
        fig.suptitle(
            f"Explication PopulationB — [{status}]  "
            f"Réel : {class_names[true_label]}  →  "
            f"Prédit : {class_names[pred]}",
            fontsize=13)
    else:
        fig.suptitle(
            f"Explication PopulationB — Prédit : {class_names[pred]}",
            fontsize=13)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path)
                    else ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"[OK] Explication sauvegardée : {save_path}")
    else:
        plt.show()

    return pred, votes.cpu(), heatmap_np


def predict_from_path(pop, img_path,
                      class_names=["Cancer", "Normal"],
                      device="cpu",
                      save_path=None,
                      true_label=None):
    """
    Prédit et explique depuis le chemin d'une image.
    """
    from PIL import Image
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
    ])

    img = Image.open(img_path)
    t   = transform(img)
    if t.dim() == 3:
        t = t.mean(0)  # → (H, W)

    pred, votes, heatmap = predict_and_explain(
        pop         = pop,
        img_tensor  = t,
        class_names = class_names,
        device      = device,
        save_path   = save_path,
        true_label  = true_label
    )

    print(f"\n{'='*40}")
    print(f"  Prédiction : {class_names[pred]}")
    print(f"  Scores     : "
          f"{class_names[0]}={votes[0]:.4f}  "
          f"{class_names[1]}={votes[1]:.4f}")
    print(f"{'='*40}")

    return pred, votes, heatmap


def predict_single(pop, trainer, img, class_names=["Cancer", "Normal"], device="cpu"):
    """
    Prédit la classe d'une seule image.
    img : tensor (H, W) — image déjà prétraitée (resize + ToTensor)
    """
    pred = trainer.predict_batch([img], batch_size=1)[0]

    if pred is None:
        print("Aucun prototype actif — prédiction impossible")
        return None

    print(f"Prédiction : {class_names[pred]}")
    return pred





