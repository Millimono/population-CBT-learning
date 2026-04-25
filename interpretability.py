# # interpretability.py

# import torch
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import numpy as np

# import os

# def explain_prediction(pop, img, label, class_names, patch_size, device):
#     """
#     Pour une image donnée, montre quelles cellules s'activent
#     et où elles se trouvent dans l'image.
#     """
#     img_t = img.unsqueeze(0).to(device)
#     activated, z = pop.process_batch(img_t)
#     activated = activated[0]  # (B,)

#     # Classes des cellules actives
#     active_classes = pop.proto_class[activated]
#     active_classes = active_classes[active_classes >= 0]

#     if len(active_classes) == 0:
#         print("Aucune cellule active")
#         return

#     pred = torch.mode(active_classes).values.item()

#     # Trouver les patches les plus activés
#     # Recalculer les distances pour localisation
#     patches_t   = pop.extract_patches_batch(img_t)         # (1, P, D)
#     patches_std = pop.preprocess_patches(patches_t)
#     protos      = pop.preprocess_patches(
#         pop.prototypes.unsqueeze(0)).squeeze(0)

#     P = patches_std.shape[1]
#     D = pop.D
#     ps = patch_size

#     patches_sq = (patches_std**2).sum(dim=-1)
#     protos_sq  = (protos**2).sum(dim=-1)
#     dot        = torch.einsum("npd,bd->nbp", patches_std, protos)
#     dists_sq   = (patches_sq.unsqueeze(1) +
#                   protos_sq.view(1,-1,1) - 2*dot).clamp(min=0)

#     # Top cellules actives par classe
#     H = W = img.shape[-1]
#     n_patches_w = W - ps + 1

#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))

#     # Image originale
#     axes[0].imshow(img.numpy(), cmap="gray")
#     axes[0].set_title(f"Image originale\nVraie classe : {class_names[label]}", fontsize=11)
#     axes[0].axis("off")

#     # Heatmap d'activation
#     heatmap = torch.zeros(H, W, device=device)

#     for cell_idx in activated.nonzero(as_tuple=True)[0][:50]:  # top 50 cellules
#         cell_cls = pop.proto_class[cell_idx].item()
#         if cell_cls < 0:
#             continue

#         # Patch le plus proche de cette cellule
#         best_patch_idx = dists_sq[0, cell_idx].argmin().item()
#         row = best_patch_idx // n_patches_w
#         col = best_patch_idx  % n_patches_w

#         # Ajouter à la heatmap
#         heatmap[row:row+ps, col:col+ps] += 1.0

#     heatmap = heatmap.cpu().numpy()
#     heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)

#     axes[1].imshow(img.numpy(), cmap="gray", alpha=0.5)
#     axes[1].imshow(heatmap, cmap="hot", alpha=0.5)
#     axes[1].set_title("Zones activées par les cellules B", fontsize=11)
#     axes[1].axis("off")

#     # Distribution des cellules actives par classe
#     class_votes = [(pop.proto_class[activated] == c).sum().item()
#                    for c in range(pop.num_classes)]
#     colors = ["#F44336", "#2196F3"][:len(class_names)]
#     axes[2].bar(class_names, class_votes, color=colors)
#     axes[2].set_title(
#         f"Vote des cellules actives\nPrédiction : {class_names[pred]}", fontsize=11)
#     axes[2].set_ylabel("Nombre de cellules")
#     axes[2].grid(True, axis="y", alpha=0.3)

#     plt.suptitle(
#         f"Explication de la décision — {'[OK] Correct' if pred == label else '[ERREUR] Incorrect'}",
#         fontsize=13)
#     plt.tight_layout()
#     return fig


# def plot_interpretability_examples(pop, val_images, val_labels,
#                                     class_names, patch_size,
#                                     device, save_path, n_examples=4):
#     """
#     Montre n_examples exemples d'explication par classe
#     """
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)

#     fig, all_axes = plt.subplots(
#         n_examples, 3,
#         figsize=(15, n_examples * 4)
#     )

#     # Prendre des exemples corrects et incorrects
#     correct_idx   = []
#     incorrect_idx = []

#     preds = []
#     for img in val_images:
#         img_t      = img.unsqueeze(0).to(device)
#         activated, _ = pop.process_batch(img_t)
#         act        = activated[0]
#         ac         = pop.proto_class[act]
#         ac         = ac[ac >= 0]
#         p = torch.mode(ac).values.item() if len(ac) > 0 else -1
#         preds.append(p)

#     for i, (p, l) in enumerate(zip(preds, val_labels)):
#         if p == l and len(correct_idx) < n_examples // 2:
#             correct_idx.append(i)
#         if p != l and len(incorrect_idx) < n_examples // 2:
#             incorrect_idx.append(i)

#     selected = correct_idx + incorrect_idx

#     for row, idx in enumerate(selected[:n_examples]):
#         img   = val_images[idx]
#         label = val_labels[idx]
#         pred  = preds[idx]

#         img_t       = img.unsqueeze(0).to(device)
#         activated, _ = pop.process_batch(img_t)
#         act         = activated[0]

#         # Heatmap
#         H = W = img.shape[-1]
#         ps = patch_size
#         n_patches_w = W - ps + 1

#         patches_t   = pop.extract_patches_batch(img_t)
#         patches_std = pop.preprocess_patches(patches_t)
#         protos      = pop.preprocess_patches(
#             pop.prototypes.unsqueeze(0)).squeeze(0)

#         patches_sq = (patches_std**2).sum(dim=-1)
#         protos_sq  = (protos**2).sum(dim=-1)
#         dot        = torch.einsum("npd,bd->nbp", patches_std, protos)
#         dists_sq   = (patches_sq.unsqueeze(1) +
#                       protos_sq.view(1,-1,1) - 2*dot).clamp(min=0)

#         heatmap = torch.zeros(H, W, device=device)
#         for cell_idx in act.nonzero(as_tuple=True)[0][:50]:
#             cell_cls = pop.proto_class[cell_idx].item()
#             if cell_cls < 0:
#                 continue
#             best_patch_idx = dists_sq[0, cell_idx].argmin().item()
#             r = best_patch_idx // n_patches_w
#             c = best_patch_idx  % n_patches_w
#             heatmap[r:r+ps, c:c+ps] += 1.0

#         heatmap = heatmap.cpu().numpy()
#         heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)

#         # Image originale
#         all_axes[row][0].imshow(img.numpy(), cmap="gray")
#         all_axes[row][0].set_title(
#             f"Classe réelle : {class_names[label]}", fontsize=10)
#         all_axes[row][0].axis("off")

#         # Heatmap
#         all_axes[row][1].imshow(img.numpy(), cmap="gray", alpha=0.5)
#         all_axes[row][1].imshow(heatmap, cmap="hot", alpha=0.5)
#         status = "[OK]" if pred == label else "[ERREUR]"
#         all_axes[row][1].set_title(
#             f"{status} Prédit : {class_names[pred]}", fontsize=10)
#         all_axes[row][1].axis("off")

#         # Votes
#         class_votes = [(pop.proto_class[act] == c).sum().item()
#                        for c in range(pop.num_classes)]
#         colors = ["#F44336", "#2196F3"]
#         all_axes[row][2].bar(class_names, class_votes, color=colors)
#         all_axes[row][2].set_ylabel("Cellules actives")
#         all_axes[row][2].grid(True, axis="y", alpha=0.3)

#     plt.suptitle(
#         "Interprétabilité — Explication des décisions (PopulationB)",
#         fontsize=13)
#     plt.tight_layout()
#     plt.savefig(save_path, bbox_inches="tight", dpi=150)
#     plt.close()
#     print(f"[OK] Figure interprétabilité : {save_path}")

# ============================================================
# interpretability.py — Version cohérente avec predict_batch
# ============================================================

import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def _compute_heatmap_and_votes(pop, img_t, act, device):
    """
    Calcule la heatmap pondérée et les scores de vote réels.
    Cohérent avec predict_batch — utilise get_vote_weights().
    """
    H = W   = img_t.shape[-1]
    ps      = pop.patch_size[0]
    n_pw    = W - ps + 1

    # Distances patches ↔ prototypes
    patches_t   = pop.extract_patches_batch(img_t)
    patches_std = pop.preprocess_patches(patches_t)
    protos      = pop.preprocess_patches(
                      pop.prototypes.unsqueeze(0)).squeeze(0)

    patches_sq = (patches_std ** 2).sum(dim=-1)
    protos_sq  = (protos ** 2).sum(dim=-1)
    dot        = torch.einsum("npd,bd->nbp", patches_std, protos)
    dists_sq   = (patches_sq.unsqueeze(1) +
                  protos_sq.view(1, -1, 1) - 2 * dot).clamp(min=0)

    # Poids d'exclusivité — identiques à predict_batch
    weights, freq = pop.get_vote_weights()

    # Prototypes valides (actifs + assignés)
    valid = act & (pop.proto_class >= 0)

    # ── Vote pondéré réel ─────────────────────────────────────
    votes = torch.zeros(pop.num_classes, device=device)
    if valid.any():
        active_freq    = freq[valid]
        active_weights = weights[valid]
        votes = (active_freq * active_weights.unsqueeze(1)).sum(dim=0)

    if votes.sum() == 0:
        # Fallback identique à predict_batch
        ac   = pop.proto_class[valid]
        pred = torch.mode(ac).values.item() if len(ac) > 0 else 0
    else:
        pred = votes.argmax().item()

    # ── Heatmap pondérée par wᵢ ───────────────────────────────
    # Seuls les prototypes ayant voté pour la classe prédite
    heatmap = torch.zeros(H, W, device=device)
    for cell_idx in valid.nonzero(as_tuple=True)[0]:
        if pop.proto_class[cell_idx].item() != pred:
            continue
        w_i            = weights[cell_idx].item()
        best_patch_idx = dists_sq[0, cell_idx].argmin().item()
        r = best_patch_idx // n_pw
        c = best_patch_idx  % n_pw
        heatmap[r:r+ps, c:c+ps] += w_i

    heatmap_np = heatmap.cpu().numpy()
    if heatmap_np.max() > 0:
        heatmap_np = (heatmap_np - heatmap_np.min()) / \
                     (heatmap_np.max() - heatmap_np.min() + 1e-6)

    return pred, votes.cpu(), heatmap_np


def explain_prediction(pop, img, label, class_names, patch_size, device):
    """
    Explique la décision pour une seule image.
    Retourne la figure matplotlib.
    """
    img_t            = img.unsqueeze(0).to(device)
    activated, _     = pop.process_batch(img_t)
    act              = activated[0]

    pred, votes, heatmap = _compute_heatmap_and_votes(
        pop, img_t, act, device)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Colonne 1 — image originale
    axes[0].imshow(img.cpu().numpy(), cmap="gray")
    axes[0].set_title(
        f"Image originale\nClasse réelle : {class_names[label]}",
        fontsize=11)
    axes[0].axis("off")

    # Colonne 2 — heatmap pondérée
    axes[1].imshow(img.cpu().numpy(), cmap="gray", alpha=0.6)
    axes[1].imshow(heatmap, cmap="hot", alpha=0.5)
    n_active = (act & (pop.proto_class >= 0)).sum().item()
    axes[1].set_title(
        f"Zones responsables de la décision\n"
        f"{n_active} prototypes actifs (pondérés par wᵢ)",
        fontsize=10)
    axes[1].axis("off")

    # Colonne 3 — vote pondéré réel Σ fᵢ · wᵢ
    vote_vals = votes.numpy()
    colors    = ["#F44336" if i == pred else "#90CAF9"
                 for i in range(pop.num_classes)]
    bars = axes[2].bar(class_names, vote_vals, color=colors)
    for bar, val in zip(bars, vote_vals):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    axes[2].set_title(
        f"Vote pondéré (Σ fᵢ · wᵢ)\nPrédiction : {class_names[pred]}",
        fontsize=11)
    axes[2].set_ylabel("Score de vote")
    axes[2].grid(True, axis="y", alpha=0.3)

    status = "Correct" if pred == label else "Incorrect"
    plt.suptitle(
        f"Explication PopulationB — [{status}]  "
        f"Réel : {class_names[label]}  →  Prédit : {class_names[pred]}",
        fontsize=13)
    plt.tight_layout()
    return fig


def plot_interpretability_examples(pop, val_images, val_labels,
                                   class_names, patch_size,
                                   device, save_path, n_examples=4):
    """
    Génère une figure avec n_examples exemples
    (moitié corrects, moitié incorrects).
    """
    os.makedirs(
        os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
        exist_ok=True)

    # ── Prédictions sur tout le val set ──────────────────────
    preds = []
    for img in val_images:
        img_t        = img.unsqueeze(0).to(device)
        activated, _ = pop.process_batch(img_t)
        act          = activated[0]
        valid        = act & (pop.proto_class >= 0)
        weights, freq = pop.get_vote_weights()
        if valid.any():
            votes = (freq[valid] * weights[valid].unsqueeze(1)).sum(dim=0)
            p     = votes.argmax().item() if votes.sum() > 0 \
                    else torch.mode(pop.proto_class[valid]).values.item()
        else:
            p = -1
        preds.append(p)

    # ── Sélection des exemples ────────────────────────────────
    correct_idx, incorrect_idx = [], []
    for i, (p, l) in enumerate(zip(preds, val_labels)):
        if p == l  and len(correct_idx)   < n_examples // 2:
            correct_idx.append(i)
        if p != l  and len(incorrect_idx) < n_examples // 2:
            incorrect_idx.append(i)
    selected = (correct_idx + incorrect_idx)[:n_examples]

    # ── Figure ───────────────────────────────────────────────
    fig, all_axes = plt.subplots(
        n_examples, 3,
        figsize=(15, n_examples * 4))

    for row, idx in enumerate(selected):
        img   = val_images[idx]
        label = val_labels[idx]
        pred  = preds[idx]

        img_t        = img.unsqueeze(0).to(device)
        activated, _ = pop.process_batch(img_t)
        act          = activated[0]

        _, votes, heatmap = _compute_heatmap_and_votes(
            pop, img_t, act, device)

        # Colonne 1
        all_axes[row][0].imshow(img.cpu().numpy(), cmap="gray")
        all_axes[row][0].set_title(
            f"Classe réelle : {class_names[label]}", fontsize=10)
        all_axes[row][0].axis("off")

        # Colonne 2
        all_axes[row][1].imshow(img.cpu().numpy(), cmap="gray", alpha=0.6)
        all_axes[row][1].imshow(heatmap, cmap="hot", alpha=0.5)
        status = "[OK]" if pred == label else "[ERREUR]"
        all_axes[row][1].set_title(
            f"{status} Prédit : {class_names[pred]}", fontsize=10)
        all_axes[row][1].axis("off")

        # Colonne 3 — vote pondéré réel
        vote_vals = votes.numpy()
        colors    = ["#F44336" if i == pred else "#90CAF9"
                     for i in range(pop.num_classes)]
        bars = all_axes[row][2].bar(class_names, vote_vals, color=colors)
        for bar, val in zip(bars, vote_vals):
            all_axes[row][2].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        all_axes[row][2].set_ylabel("Score (Σ fᵢ · wᵢ)")
        all_axes[row][2].grid(True, axis="y", alpha=0.3)

    plt.suptitle(
        "Interprétabilité native — PopulationB",
        fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[OK] Figure interprétabilité : {save_path}")


def select_best_examples(pop, val_images, val_labels,
                         class_names, device,
                         n_candidates=30,
                         n_examples=4):
    """
    Analyse n_candidates images et sélectionne automatiquement
    les meilleures pour illustration.
    """
    candidates = []

    for idx in range(min(n_candidates, len(val_images))):
        img   = val_images[idx]
        label = val_labels[idx]

        img_t        = img.unsqueeze(0).to(device)
        activated, _ = pop.process_batch(img_t)
        act          = activated[0]

        pred, votes, heatmap = _compute_heatmap_and_votes(
            pop, img_t, act, device)

        vote_vals    = votes.numpy()
        n_active     = (act & (pop.proto_class >= 0)).sum().item()
        vote_margin  = abs(vote_vals[0] - vote_vals[1])
        correct      = (pred == label)

        candidates.append({
            "idx":         idx,
            "img":         img,
            "label":       label,
            "pred":        pred,
            "votes":       votes,
            "heatmap":     heatmap,
            "n_active":    n_active,
            "vote_margin": vote_margin,
            "correct":     correct
        })

    corrects   = [c for c in candidates
                  if c["correct"] and c["n_active"] > 0]
    incorrects = [c for c in candidates
                  if not c["correct"] and c["n_active"] > 0]

    corrects   = sorted(corrects,
                        key=lambda x: x["vote_margin"],
                        reverse=True)
    incorrects = sorted(incorrects,
                        key=lambda x: x["vote_margin"])

    n_correct   = n_examples // 2
    n_incorrect = n_examples - n_correct
    selected    = corrects[:n_correct] + incorrects[:n_incorrect]

    print(f"\n{'='*50}")
    print(f"SÉLECTION AUTOMATIQUE — {len(selected)} images")
    print(f"{'='*50}")
    for c in selected:
        status = "✅ Correct" if c["correct"] else "❌ Incorrect"
        print(f"  idx={c['idx']:4d} | {status} | "
              f"Réel={class_names[c['label']]:6s} | "
              f"Prédit={class_names[c['pred']]:6s} | "
              f"Margin={c['vote_margin']:.3f} | "
              f"Actifs={c['n_active']}")

    return selected


def plot_best_interpretability_examples(pop, val_images, val_labels,
                                        class_names, patch_size,
                                        device, save_path,
                                        n_candidates=30,
                                        n_examples=4):
    """
    Analyse n_candidates images, sélectionne les meilleures
    et génère la figure d'interprétabilité.
    """
    os.makedirs(
        os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
        exist_ok=True)

    selected = select_best_examples(
        pop, val_images, val_labels,
        class_names, device,
        n_candidates=n_candidates,
        n_examples=n_examples
    )

    fig, all_axes = plt.subplots(
        n_examples, 3,
        figsize=(15, n_examples * 4))

    for row, c in enumerate(selected):
        img      = c["img"]
        label    = c["label"]
        pred     = c["pred"]
        votes    = c["votes"]
        heatmap  = c["heatmap"]
        n_active = c["n_active"]

        # Colonne 1 — image originale
        all_axes[row][0].imshow(img.cpu().numpy(), cmap="gray")
        all_axes[row][0].set_title(
            f"Classe réelle : {class_names[label]}\n"
            f"idx={c['idx']}",
            fontsize=10)
        all_axes[row][0].axis("off")

        # Colonne 2 — heatmap pondérée
        all_axes[row][1].imshow(img.cpu().numpy(), cmap="gray", alpha=0.6)
        all_axes[row][1].imshow(heatmap, cmap="hot", alpha=0.5)
        status = "[OK]" if pred == label else "[ERREUR]"
        all_axes[row][1].set_title(
            f"{status} Prédit : {class_names[pred]}\n"
            f"{n_active} prototypes actifs",
            fontsize=10)
        all_axes[row][1].axis("off")

        # Colonne 3 — vote pondéré
        vote_vals = votes.numpy()
        colors    = ["#F44336" if i == pred else "#90CAF9"
                     for i in range(pop.num_classes)]
        bars = all_axes[row][2].bar(class_names, vote_vals, color=colors)
        for bar, val in zip(bars, vote_vals):
            all_axes[row][2].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        all_axes[row][2].set_ylabel("Score (Σ fᵢ · wᵢ)")
        all_axes[row][2].set_ylim(
            0, max(vote_vals) * 1.2 + 0.01)
        all_axes[row][2].grid(True, axis="y", alpha=0.3)

    plt.suptitle(
        f"Interprétabilité native — EpitopeNet\n"
        f"Sélection automatique sur {n_candidates} images",
        fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[OK] Figure interprétabilité : {save_path}")

def save_all_interpretability_examples(pop, val_images, val_labels,
                                       class_names, device,
                                       save_dir="figs/interpretability",
                                       n_candidates=30):
    """
    Génère et sauvegarde UNE figure par image — toutes les n_candidates.
    Chaque image est sauvegardée indépendamment.
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"GÉNÉRATION DE {n_candidates} FIGURES INDIVIDUELLES")
    print(f"{'='*50}")

    for idx in range(min(n_candidates, len(val_images))):
        img   = val_images[idx]
        label = val_labels[idx]

        img_t        = img.unsqueeze(0).to(device)
        activated, _ = pop.process_batch(img_t)
        act          = activated[0]

        pred, votes, heatmap = _compute_heatmap_and_votes(
            pop, img_t, act, device)

        vote_vals   = votes.numpy()
        n_active    = (act & (pop.proto_class >= 0)).sum().item()
        vote_margin = abs(vote_vals[0] - vote_vals[1])
        correct     = (pred == label)
        status      = "correct" if correct else "incorrect"

        # ── Figure individuelle ───────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Colonne 1 — image originale
        axes[0].imshow(img.cpu().numpy(), cmap="gray")
        axes[0].set_title(
            f"Classe réelle : {class_names[label]}",
            fontsize=11)
        axes[0].axis("off")

        # Colonne 2 — heatmap pondérée
        axes[1].imshow(img.cpu().numpy(), cmap="gray", alpha=0.6)
        axes[1].imshow(heatmap, cmap="hot", alpha=0.5)
        status_label = "[OK]" if correct else "[ERREUR]"
        axes[1].set_title(
            f"{status_label} Prédit : {class_names[pred]}\n"
            f"{n_active} prototypes actifs",
            fontsize=10)
        axes[1].axis("off")

        # Colonne 3 — vote pondéré
        colors = ["#F44336" if i == pred else "#90CAF9"
                  for i in range(pop.num_classes)]
        bars = axes[2].bar(class_names, vote_vals, color=colors)
        for bar, val in zip(bars, vote_vals):
            axes[2].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)
        axes[2].set_ylabel("Score (Σ fᵢ · wᵢ)")
        axes[2].set_ylim(0, max(vote_vals) * 1.2 + 0.01)
        axes[2].grid(True, axis="y", alpha=0.3)

        plt.suptitle(
            f"EpitopeNet — idx={idx} | [{status_label}] "
            f"Réel: {class_names[label]} → Prédit: {class_names[pred]} | "
            f"Margin: {vote_margin:.3f}",
            fontsize=12)
        plt.tight_layout()

        # Nom du fichier avec toutes les infos
        fname = (f"idx{idx:04d}"
                 f"_real{class_names[label]}"
                 f"_pred{class_names[pred]}"
                 f"_{status}"
                 f"_margin{vote_margin:.3f}"
                 f".png")
        save_path = os.path.join(save_dir, fname)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()

        print(f"  [{idx+1:2d}/{n_candidates}] {fname}")

    print(f"\n[OK] {n_candidates} figures sauvegardées dans : {save_dir}/")

def save_balanced_interpretability_examples(pop, val_images, val_labels,
                                            class_names, device,
                                            save_dir="figs/interpretability",
                                            n_per_class=15):
    """
    Génère n_per_class figures par classe — équilibré.
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"GÉNÉRATION ÉQUILIBRÉE — {n_per_class} images par classe")
    print(f"{'='*50}")

    # Séparer les indices par classe
    class_indices = {c: [] for c in range(len(class_names))}
    for idx, label in enumerate(val_labels):
        class_indices[label].append(idx)

    # Prendre n_per_class de chaque classe
    selected_indices = []
    for c in range(len(class_names)):
        selected_indices.extend(class_indices[c][:n_per_class])

    print(f"  Cancer  : {len(class_indices[0][:n_per_class])} images")
    print(f"  Normal  : {len(class_indices[1][:n_per_class])} images")
    print(f"  Total   : {len(selected_indices)} images\n")

    for count, idx in enumerate(selected_indices):
        img   = val_images[idx]
        label = val_labels[idx]

        img_t        = img.unsqueeze(0).to(device)
        activated, _ = pop.process_batch(img_t)
        act          = activated[0]

        pred, votes, heatmap = _compute_heatmap_and_votes(
            pop, img_t, act, device)

        vote_vals   = votes.numpy()
        n_active    = (act & (pop.proto_class >= 0)).sum().item()
        vote_margin = abs(vote_vals[0] - vote_vals[1])
        correct     = (pred == label)
        status      = "correct" if correct else "incorrect"
        status_label = "[OK]" if correct else "[ERREUR]"

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img.cpu().numpy(), cmap="gray")
        axes[0].set_title(
            f"Classe réelle : {class_names[label]}",
            fontsize=11)
        axes[0].axis("off")

        axes[1].imshow(img.cpu().numpy(), cmap="gray", alpha=0.6)
        axes[1].imshow(heatmap, cmap="hot", alpha=0.5)
        axes[1].set_title(
            f"{status_label} Prédit : {class_names[pred]}\n"
            f"{n_active} prototypes actifs",
            fontsize=10)
        axes[1].axis("off")

        colors = ["#F44336" if i == pred else "#90CAF9"
                  for i in range(pop.num_classes)]
        bars = axes[2].bar(class_names, vote_vals, color=colors)
        for bar, val in zip(bars, vote_vals):
            axes[2].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)
        axes[2].set_ylabel("Score (Σ fᵢ · wᵢ)")
        axes[2].set_ylim(0, max(vote_vals) * 1.2 + 0.01)
        axes[2].grid(True, axis="y", alpha=0.3)

        plt.suptitle(
            f"EpitopeNet — idx={idx} | {status_label} "
            f"Réel: {class_names[label]} → Prédit: {class_names[pred]} | "
            f"Margin: {vote_margin:.3f}",
            fontsize=12)
        plt.tight_layout()

        fname = (f"idx{idx:04d}"
                 f"_real{class_names[label]}"
                 f"_pred{class_names[pred]}"
                 f"_{status}"
                 f"_margin{vote_margin:.3f}"
                 f".png")
        save_path = os.path.join(save_dir, fname)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()

        print(f"  [{count+1:2d}/{len(selected_indices)}] {fname}")

    print(f"\n[OK] {len(selected_indices)} figures dans : {save_dir}/")