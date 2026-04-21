# # ============================================================
# # run_ablation.py — Ablation study séparé
# # ============================================================

# import torch
# import gc
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from data  import load_ddsm
# from train import run_experiment
# from run   import set_seed, TRAIN_DIR, VAL_DIR, DEVICE, NUM_CLASSES, EPOCHS, LR

# ABLATION_SEED = 42  # seed fixe pour toute l'ablation

# # ============================================================
# if __name__ == "__main__":

#     torch.cuda.empty_cache()
#     gc.collect()
#     os.makedirs("figs", exist_ok=True)

#     # Charger les données une seule fois
#     set_seed(ABLATION_SEED)
#     train_images, train_labels, val_images, val_labels = load_ddsm(
#         TRAIN_DIR, VAL_DIR
#     )

#     results = []

#     # ============================================================
#     # ABLATION K
#     # ============================================================
#     print("\n=== ABLATION K ===")
#     for K in [1, 3, 5, 10, 20]:
#         set_seed(ABLATION_SEED)

#         # Créer un modèle avec K variable
#         from model import PopulationBFastExact, TrainerFastExact
#         from train import init_prototypes_from_data
#         import torch.nn.functional as F

#         class PopK(PopulationBFastExact):
#             def process_batch(self, imgs):
#                 imgs        = imgs.to(self.device)
#                 patches     = self.extract_patches_batch(imgs)
#                 patches_std = self.preprocess_patches(patches)
#                 protos      = self.preprocess_patches(
#                     self.prototypes.unsqueeze(0)).squeeze(0)
#                 N, P, D = patches_std.shape
#                 B       = protos.shape[0]
#                 patches_sq = (patches_std**2).sum(dim=-1)
#                 protos_sq  = (protos**2).sum(dim=-1)
#                 dot        = torch.einsum("npd,bd->nbp", patches_std, protos)
#                 dists_sq   = (patches_sq.unsqueeze(1) +
#                               protos_sq.view(1,B,1) - 2*dot).clamp(min=0)
#                 topk_dists, topk_idx = dists_sq.topk(self.K, dim=2, largest=False)
#                 sim       = torch.exp(-topk_dists.mean(dim=2) / self.D**0.5)
#                 activated = (sim >= self.theta_init).bool()
#                 topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1,-1,-1,D)
#                 patches_exp  = patches_std.unsqueeze(1).expand(-1,B,-1,-1)
#                 z = patches_exp.gather(2, topk_idx_exp).mean(dim=2)
#                 return activated, z

#         pop     = PopK(num_cells=1600, patch_size=(5,5), theta_init=0.5,
#                        beta=5.0, num_classes=NUM_CLASSES, device=DEVICE)
#         pop.K   = K
#         trainer = TrainerFastExact(population=pop,
#                                    num_classes=NUM_CLASSES, device=DEVICE)
#         init_prototypes_from_data(pop, train_images, device=DEVICE, n_samples=200)

#         best = 0.0
#         for epoch in range(10):  # 10 epochs rapides pour ablation
#             lr_e = LR * (0.95**epoch)
#             trainer.train_batch(train_images, train_labels,
#                                 batch_size=32, lr=lr_e)
#             pop.reassign_proto_class(train_images, train_labels,
#                                      DEVICE, batch_size=64)
#             preds   = trainer.predict_batch(val_images, batch_size=64)
#             correct = sum(p == l for p, l in zip(preds, val_labels)
#                          if p is not None)
#             acc     = correct / len(val_images)
#             best    = max(best, acc)

#         print(f"  K={K} → {best:.4f}")
#         results.append({"param": "K", "value": K, "acc": best})

#     # ============================================================
#     # ABLATION num_cells
#     # ============================================================
#     print("\n=== ABLATION num_cells ===")
#     for num_cells in [200, 400, 800, 1600]:
#         set_seed(ABLATION_SEED)
#         acc, _, _, _ = run_experiment(
#             train_images, train_labels, val_images, val_labels,
#             name=f"num_cells={num_cells}", num_classes=NUM_CLASSES,
#             epochs=10, lr=LR, num_cells=num_cells,
#             patch_size=(5,5), theta_init=0.5, device=DEVICE
#         )
#         print(f"  num_cells={num_cells} → {acc:.4f}")
#         results.append({"param": "num_cells", "value": num_cells, "acc": acc})

#     # ============================================================
#     # ABLATION patch_size
#     # ============================================================
#     print("\n=== ABLATION patch_size ===")
#     for ps in [3, 5, 7, 9]:
#         set_seed(ABLATION_SEED)
#         acc, _, _, _ = run_experiment(
#             train_images, train_labels, val_images, val_labels,
#             name=f"patch={ps}x{ps}", num_classes=NUM_CLASSES,
#             epochs=10, lr=LR, num_cells=1600,
#             patch_size=(ps,ps), theta_init=0.5, device=DEVICE
#         )
#         print(f"  patch={ps}x{ps} → {acc:.4f}")
#         results.append({"param": "patch_size", "value": ps, "acc": acc})

#     # ============================================================
#     # ABLATION theta
#     # ============================================================
#     print("\n=== ABLATION theta ===")
#     for theta in [0.3, 0.4, 0.5, 0.6, 0.7]:
#         set_seed(ABLATION_SEED)
#         acc, _, _, _ = run_experiment(
#             train_images, train_labels, val_images, val_labels,
#             name=f"theta={theta}", num_classes=NUM_CLASSES,
#             epochs=10, lr=LR, num_cells=1600,
#             patch_size=(5,5), theta_init=theta, device=DEVICE
#         )
#         print(f"  theta={theta} → {acc:.4f}")
#         results.append({"param": "theta", "value": theta, "acc": acc})

#     # ============================================================
#     # Résumé
#     # ============================================================
#     df = pd.DataFrame(results)
#     print(f"\n{'='*40}")
#     print("RÉSUMÉ ABLATION")
#     print(f"{'='*40}")
#     for param in df["param"].unique():
#         sub = df[df["param"] == param].sort_values("value")
#         print(f"\n{param}:")
#         for _, row in sub.iterrows():
#             marker = " ← best" if row["acc"] == sub["acc"].max() else ""
#             print(f"  {str(row['value']):>8} → {row['acc']:.4f}{marker}")

#     # Sauvegarder CSV
#     df.to_csv("figs/ablation_results.csv", index=False)
#     print("\n✅ Résultats sauvegardés : figs/ablation_results.csv")

# ============================================================
# run_ablation.py — Ablation study
# ============================================================

import torch
import gc
import os
import numpy as np
import pandas as pd
from data  import load_ddsm
from train import run_experiment
from run   import set_seed, TRAIN_DIR, VAL_DIR, DEVICE, NUM_CLASSES, LR

ABLATION_SEED   = 42
ABLATION_EPOCHS = 10  # rapide — mentionné dans l'article

# ============================================================
if __name__ == "__main__":

    torch.cuda.empty_cache()
    gc.collect()
    os.makedirs("figs", exist_ok=True)

    set_seed(ABLATION_SEED)
    train_images, train_labels, val_images, val_labels = load_ddsm(
        TRAIN_DIR, VAL_DIR
    )

    results = []

    # ============================================================
    # ABLATION K
    # ============================================================
    print("\n=== ABLATION K ===")
    for K in [1, 3, 5, 10, 20]:
        set_seed(ABLATION_SEED)
        acc, _, _, _ = run_experiment(
            train_images, train_labels, val_images, val_labels,
            name        = f"K={K}",
            num_classes = NUM_CLASSES,
            epochs      = ABLATION_EPOCHS,
            lr          = LR,
            num_cells   = 1600,
            patch_size  = (5, 5),
            theta_init  = 0.5,
            device      = DEVICE,
            K           = K          # ← correct
        )
        print(f"  K={K} → {acc:.4f}")
        results.append({"param": "K", "value": K, "acc": acc})

    # ============================================================
    # ABLATION num_cells
    # ============================================================
    print("\n=== ABLATION num_cells ===")
    for num_cells in [200, 400, 800, 1600]:
        set_seed(ABLATION_SEED)
        acc, _, _, _ = run_experiment(
            train_images, train_labels, val_images, val_labels,
            name        = f"num_cells={num_cells}",
            num_classes = NUM_CLASSES,
            epochs      = ABLATION_EPOCHS,
            lr          = LR,
            num_cells   = num_cells,
            patch_size  = (5, 5),
            theta_init  = 0.5,
            device      = DEVICE,
            K           = 1
        )
        print(f"  num_cells={num_cells} → {acc:.4f}")
        results.append({"param": "num_cells", "value": num_cells, "acc": acc})

    # ============================================================
    # ABLATION patch_size
    # ============================================================
    print("\n=== ABLATION patch_size ===")
    for ps in [3, 5, 7, 9]:
        set_seed(ABLATION_SEED)
        acc, _, _, _ = run_experiment(
            train_images, train_labels, val_images, val_labels,
            name        = f"patch={ps}x{ps}",
            num_classes = NUM_CLASSES,
            epochs      = ABLATION_EPOCHS,
            lr          = LR,
            num_cells   = 1600,
            patch_size  = (ps, ps),
            theta_init  = 0.5,
            device      = DEVICE,
            K           = 1
        )
        print(f"  patch={ps}x{ps} → {acc:.4f}")
        results.append({"param": "patch_size", "value": ps, "acc": acc})

    # ============================================================
    # ABLATION theta
    # ============================================================
    print("\n=== ABLATION theta ===")
    for theta in [0.3, 0.4, 0.5, 0.6, 0.7]:
        set_seed(ABLATION_SEED)
        acc, _, _, _ = run_experiment(
            train_images, train_labels, val_images, val_labels,
            name        = f"theta={theta}",
            num_classes = NUM_CLASSES,
            epochs      = ABLATION_EPOCHS,
            lr          = LR,
            num_cells   = 1600,
            patch_size  = (5, 5),
            theta_init  = theta,
            device      = DEVICE,
            K           = 1
        )
        print(f"  theta={theta} → {acc:.4f}")
        results.append({"param": "theta", "value": theta, "acc": acc})

    # ============================================================
    # Résumé + CSV
    # ============================================================
    df = pd.DataFrame(results)
    print(f"\n{'='*40}")
    print("RÉSUMÉ ABLATION")
    print(f"{'='*40}")
    for param in df["param"].unique():
        sub = df[df["param"] == param].sort_values("value")
        print(f"\n{param}:")
        for _, row in sub.iterrows():
            best_marker = " ← best" if row["acc"] == sub["acc"].max() else ""
            print(f"  {str(row['value']):>8} → {row['acc']:.4f}{best_marker}")

    df.to_csv("figs/ablation_results.csv", index=False)
    print("\n[OK] figs/ablation_results.csv")