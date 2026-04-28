# # ============================================================
# # run.py — Point d'entrée unique
# # ============================================================

# import torch
# import gc
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from data  import load_ddsm
# from train import run_experiment
# from baselines import run_baselines

# from interpretability import plot_interpretability_examples

# # ============================================================
# # CONFIG
# # ============================================================
# TRAIN_DIR   = "/content/PhD_AI_Grad_CAM_CAS_1_BAselines/miniddsm/train"
# VAL_DIR     = "/content/PhD_AI_Grad_CAM_CAS_1_BAselines/miniddsm/val"
# DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# NUM_CLASSES = 2
# EPOCHS      = 40
# LR          = 0.1
# NUM_CELLS   = 1600
# PATCH_SIZE  = (5, 5)
# THETA_INIT  = 0.5
# SEEDS       = [42, 123, 456, 789, 1024]
# K = 1  

# # ============================================================
# def set_seed(seed):
#     import random
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True


# # def plot_learning_curve(history, name, save_path):
# #     fig, ax = plt.subplots(figsize=(8, 5))
# #     epochs  = range(1, len(history) + 1)
# #     ax.plot(epochs, history, color="#2196F3", linewidth=2, label="Notre méthode")
# #     ax.axhline(y=max(history), color="#2196F3", linestyle=":",
# #                linewidth=1.5, label=f"Best ({max(history):.2%})")
# #     ax.set_xlabel("Epoch", fontsize=12)
# #     ax.set_ylabel("Accuracy (validation)", fontsize=12)
# #     ax.set_title(name, fontsize=13)
# #     ax.legend(fontsize=10)
# #     ax.set_ylim(0.3, 1.0)
# #     ax.set_xlim(1, len(history))
# #     ax.grid(True, alpha=0.3)
# #     plt.tight_layout()
# #     plt.savefig(save_path, bbox_inches="tight", dpi=150)
# #     plt.close()
# #     print(f"✅ Courbe sauvegardée : {save_path}")

# def plot_learning_curve(history, name, save_path):
#     fig, ax = plt.subplots(figsize=(8, 5))
#     epochs  = range(1, len(history) + 1)
#     ax.plot(epochs, history, color="#2196F3", linewidth=2, label="Notre méthode")
#     ax.axhline(y=max(history), color="#2196F3", linestyle=":",
#                linewidth=1.5, label=f"Best ({max(history):.2%})")
#     ax.set_xlabel("Epoch", fontsize=12)
#     ax.set_ylabel("Accuracy (validation)", fontsize=12)
#     ax.set_title(name, fontsize=13)
#     ax.legend(fontsize=10)
#     ax.set_ylim(0.3, 1.0)
#     # ✅ Fix warning xlim
#     if len(history) > 1:
#         ax.set_xlim(1, len(history))
#     ax.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(save_path, bbox_inches="tight", dpi=150)
#     plt.close()
#     print(f"✅ Courbe sauvegardée : {save_path}")


# def plot_prototypes(pop, class_names, patch_size, save_path, n_proto=12):
#     fig, axes = plt.subplots(
#         len(class_names), n_proto,
#         figsize=(n_proto * 1.2, len(class_names) * 1.4)
#     )
#     for c, cls_name in enumerate(class_names):
#         mask     = (pop.proto_class == c)
#         protos_c = pop.prototypes[mask].cpu().detach()
#         norms    = protos_c.norm(dim=1)
#         top_idx  = norms.argsort(descending=True)[:n_proto]
#         protos_c = protos_c[top_idx]
#         for j in range(n_proto):
#             ax = axes[c][j] if len(class_names) > 1 else axes[j]
#             if j < len(protos_c):
#                 patch = protos_c[j].reshape(patch_size, patch_size).numpy()
#                 patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-6)
#                 ax.imshow(patch, cmap="viridis", vmin=0, vmax=1)
#             ax.axis("off")
#         axes[c][0].set_ylabel(cls_name, fontsize=11,
#                               rotation=90, labelpad=10, va="center")
#     plt.suptitle("Prototypes appris par classe", fontsize=13)
#     plt.tight_layout()
#     plt.savefig(save_path, bbox_inches="tight", dpi=150)
#     plt.close()
#     print(f"✅ Prototypes sauvegardés : {save_path}")


# # ============================================================
# if __name__ == "__main__":

#     torch.cuda.empty_cache()
#     gc.collect()
#     os.makedirs("figs", exist_ok=True)
#     print(f"Device: {DEVICE}")

#     # ============================================================
#     # Évaluation sur 5 seeds
#     # ============================================================
#     accs         = []
#     best_acc     = 0.0
#     best_pop     = None
#     best_history = None

#     # print("=== ÉVALUATION SUR 5 SEEDS ===\n")

#     import time

#     # ============================================================
#     # Dans la boucle seeds — mesurer le temps
#     # ============================================================
#     print("=== ÉVALUATION SUR 5 SEEDS ===\n")

#     train_times = []



#     for seed in SEEDS:
#         set_seed(seed)

#         train_images, train_labels, val_images, val_labels = load_ddsm(
#             TRAIN_DIR, VAL_DIR
#         )

#         start_time = time.time()  # ← début chrono


#         acc, pop, trainer, history = run_experiment(
#             train_images, train_labels,
#             val_images,   val_labels,
#             name        = f"MiniDDSM — seed={seed}",
#             num_classes = NUM_CLASSES,
#             epochs      = EPOCHS,
#             lr          = LR,
#             num_cells   = NUM_CELLS,
#             patch_size  = PATCH_SIZE,
#             theta_init  = THETA_INIT,
#             device      = DEVICE,
#             K           = K  # ← passé à run_experiment
#         )

#         elapsed = time.time() - start_time  # ← fin chrono
#         train_times.append(elapsed)

#         accs.append(acc)
#         # print(f"Seed {seed:5d} → {acc:.4f}\n")
#         print(f"Seed {seed:5d} → Acc: {acc:.4f} | Temps: {elapsed:.1f}s\n")


#         # Garder le meilleur run pour les figures
#         if acc > best_acc:
#             best_acc     = acc
#             best_pop     = pop
#             best_history = history

#     # ============================================================
#     # Résumé
#     # ============================================================
#     print(f"{'='*40}")
#     print(f"RÉSUMÉ SUR 5 SEEDS — MiniDDSM")
#     print(f"{'='*40}")
#     print(f"  Résultats : {[f'{a:.4f}' for a in accs]}")
#     print(f"  Moyenne   : {np.mean(accs):.4f}")
#     print(f"  Std       : {np.std(accs):.4f}")
#     print(f"  Min       : {np.min(accs):.4f}")
#     print(f"  Max       : {np.max(accs):.4f}")
  
#     print(f"  Accuracy  : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
#     print(f"  Temps     : {np.mean(train_times):.1f}s ± {np.std(train_times):.1f}s")
#     print(f"  Temps/run : {np.mean(train_times)/60:.1f} min")
#     print(f"{'='*40}")


#     # ============================================================
#     # Figures — basées sur le meilleur run
#     # ============================================================
#     plot_learning_curve(
#         history   = best_history,
#         name      = f"MiniDDSM — Cancer vs Normal (best={best_acc:.2%})",
#         save_path = "figs/learning_curve_ddsm.png"
#     )

#     plot_prototypes(
#         pop         = best_pop,
#         class_names = ["Cancer", "Normal"],
#         patch_size  = PATCH_SIZE[0],
#         save_path   = "figs/prototypes_ddsm.png"
#     )

    
#     baseline_results = run_baselines(
#         train_images, train_labels,
#         val_images,   val_labels
#     )

#     print("\n✅ Toutes les figures générées dans figs/")
#     print(f"{'='*50}")
#     print(f"RÉSUMÉ FINAL")
#     print(f"  Moyenne ± Std : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
#     print(f"  Figures       : figs/")
#     print(f"{'='*50}")


#     import os
#     from interpretability import plot_interpretability_examples

#     plot_interpretability_examples(
#         pop         = best_pop,
#         val_images  = val_images,
#         val_labels  = val_labels,
#         class_names = ["Cancer", "Normal"],
#         patch_size  = PATCH_SIZE[0],
#         device      = DEVICE,
#         save_path   = "figs/interpretability_ddsm.png",
#         n_examples  = 4
#     )

# ============================================================
# run.py — Point d'entrée unique
# ============================================================

import os
import gc
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from data            import load_ddsm
from train           import run_experiment
from baselines       import run_baselines
from interpretability import plot_interpretability_examples



# # ============================================================
# # CONFIG
# # ============================================================
# TRAIN_DIR   = "/content/PhD_AI_Grad_CAM_CAS_1_BAselines/miniddsm/train"
# VAL_DIR     = "/content/PhD_AI_Grad_CAM_CAS_1_BAselines/miniddsm/val"
# DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# NUM_CLASSES = 2
# EPOCHS      = 40
# LR          = 0.1
# NUM_CELLS   = 6400        # ← correct
# PATCH_SIZE  = (5, 5)
# THETA_INIT  = 0.5
# SEEDS       = [42, 123, 456, 789, 1024]
# K           = 1
# DATASET     = "ddsm"      # "ddsm" ou "generic"
# # ── CONFIG ────────────────────────────────────────────────

# ============================================================
# CONFIG
# ============================================================
# TRAIN_DIR   = "/content/cbis-ddsm-prepared/train"   # ← nouveau chemin
# VAL_DIR     = "/content/cbis-ddsm-prepared/val"     # ← nouveau chemin
TRAIN_DIR   = ""   # ← peut rester vide, ignoré
VAL_DIR     = ""
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 2
EPOCHS      = 40
LR          = 0.1
NUM_CELLS   = 6400
PATCH_SIZE  = (7, 7)
THETA_INIT  = 0.5
SEEDS       = [42, 123, 456, 789, 1024]
K           = 1

DATASET     = "ddsm"      # "ddsm" ou "generic"

# ============================================================
def set_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def plot_learning_curve(history, name, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs  = range(1, len(history) + 1)
    ax.plot(epochs, history, color="#2196F3", linewidth=2, label="Notre méthode")
    ax.axhline(y=max(history), color="#2196F3", linestyle=":",
               linewidth=1.5, label=f"Best ({max(history):.2%})")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy (validation)", fontsize=12)
    ax.set_title(name, fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0.3, 1.0)
    if len(history) > 1:
        ax.set_xlim(1, len(history))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[OK] Courbe sauvegardée : {save_path}")


def plot_prototypes(pop, class_names, patch_size, save_path, n_proto=12):
    fig, axes = plt.subplots(
        len(class_names), n_proto,
        figsize=(n_proto * 1.2, len(class_names) * 1.4)
    )
    for c, cls_name in enumerate(class_names):
        mask     = (pop.proto_class == c)
        protos_c = pop.prototypes[mask].cpu().detach()
        norms    = protos_c.norm(dim=1)
        top_idx  = norms.argsort(descending=True)[:n_proto]
        protos_c = protos_c[top_idx]
        for j in range(n_proto):
            ax = axes[c][j] if len(class_names) > 1 else axes[j]
            if j < len(protos_c):
                patch = protos_c[j].reshape(patch_size, patch_size).numpy()
                patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-6)
                ax.imshow(patch, cmap="viridis", vmin=0, vmax=1)
            ax.axis("off")
        axes[c][0].set_ylabel(cls_name, fontsize=11,
                              rotation=90, labelpad=10, va="center")
    plt.suptitle("Prototypes appris par classe", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[OK] Prototypes sauvegardés : {save_path}")


# ============================================================
if __name__ == "__main__":

    torch.cuda.empty_cache()
    gc.collect()
    os.makedirs("figs", exist_ok=True)
    print(f"Device: {DEVICE}")

    accs            = []
    f1s             = []
    aucs            = []
    train_times     = []
    best_acc        = 0.0
    best_pop        = None
    best_history    = None
    best_trainer    = None
    best_val_labels = None

    print("=== EVALUATION SUR 5 SEEDS ===\n")
    torch.cuda.empty_cache()
    for seed in SEEDS:
        set_seed(seed)
        if DATASET == "ddsm":
            train_images, train_labels, val_images, val_labels = load_ddsm(
                TRAIN_DIR, VAL_DIR, use_mask = True 
            )
        elif DATASET == "generic":
            from data import load_generic
            train_images, train_labels, val_images, val_labels = load_generic(
                train_dir     = "/chemin/vers/train",
                val_dir       = "/chemin/vers/val",
                classes       = ["ClasseA", "ClasseB"],
                max_per_class = 1000,
                img_size      = 64
            )

        start_time = time.time()

        acc, pop, trainer, history = run_experiment(
            train_images, train_labels,
            val_images,   val_labels,
            name        = f"MiniDDSM — seed={seed}",
            num_classes = NUM_CLASSES,
            epochs      = EPOCHS,
            lr          = LR,
            num_cells   = NUM_CELLS,
            patch_size  = PATCH_SIZE,
            theta_init  = THETA_INIT,
            device      = DEVICE,
            K           = K
        )

        elapsed = time.time() - start_time
        train_times.append(elapsed)
        accs.append(acc)

        # F1 et AUC
        preds       = trainer.predict_batch(val_images, batch_size=32)
        preds_clean = [p if p is not None else 0 for p in preds]
        f1  = f1_score(val_labels, preds_clean, average="macro")
        auc = roc_auc_score(val_labels, preds_clean)
        f1s.append(f1)
        aucs.append(auc)

        print(f"Seed {seed:5d} → Acc: {acc:.4f} | F1: {f1:.4f} | "
              f"AUC: {auc:.4f} | Temps: {elapsed:.1f}s\n")

        if acc > best_acc:
            best_acc        = acc
            best_pop        = pop
            best_history    = history
            best_trainer    = trainer
            best_val_labels = val_labels

    # ============================================================
    # Résumé
    # ============================================================
    print(f"{'='*40}")
    print(f"RÉSUMÉ SUR 5 SEEDS — MiniDDSM")
    print(f"{'='*40}")
    print(f"  Accuracy  : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  F1 macro  : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  AUC       : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  Temps/run : {np.mean(train_times)/60:.1f} min")
    print(f"{'='*40}")

    # Rapport détaillé meilleur run
    print("\n=== RAPPORT DÉTAILLÉ (meilleur run) ===")
    preds_best  = best_trainer.predict_batch(val_images, batch_size=32)
    preds_clean = [p if p is not None else 0 for p in preds_best]
    print(classification_report(
        best_val_labels, preds_clean,
        target_names=["Cancer", "Normal"]))

    from save_load import save_model

    # ── Sauvegarder après entraînement ──────────────────────────
    save_model(best_pop, path="figs/populationb_best.pt")


    # ============================================================
    # Figures
    # ============================================================
    plot_learning_curve(
        history   = best_history,
        name      = f"MiniDDSM — Cancer vs Normal (best={best_acc:.2%})",
        save_path = "figs/learning_curve_ddsm.png"
    )

    plot_prototypes(
        pop         = best_pop,
        class_names = ["Cancer", "Normal"],
        patch_size  = PATCH_SIZE[0],
        save_path   = "figs/prototypes_ddsm.png"
    )

    baseline_results = run_baselines(
        train_images, train_labels,
        val_images,   val_labels
    )

    plot_interpretability_examples(
        pop         = best_pop,
        val_images  = val_images,
        val_labels  = val_labels,
        class_names = ["Cancer", "Normal"],
        patch_size  = PATCH_SIZE[0],
        device      = DEVICE,
        save_path   = "figs/interpretability_ddsm.png",
        n_examples  = 4
    )

    print("\n[OK] Toutes les figures generees dans figs/")
    print(f"{'='*50}")
    print(f"RÉSUMÉ FINAL")
    print(f"  Accuracy  : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  F1 macro  : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  AUC       : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  Temps/run : {np.mean(train_times)/60:.1f} min")
    print(f"{'='*50}")