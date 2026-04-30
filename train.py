# # # ============================================================
# # # train.py — Fonctions d'entraînement
# # # ============================================================

# # import torch
# # from model import PopulationBFastExact, TrainerFastExact


# # # def init_prototypes_from_data(population, images, device, n_samples=200):
# # #     import torch.nn.functional as F
# # #     imgs_batch = torch.stack(images[:n_samples]).to(device)
# # #     patches    = population.extract_patches_batch(imgs_batch)
# # #     patches    = population.preprocess_patches(patches)
# # #     patches    = patches.reshape(-1, patches.shape[2])
# # #     idx = torch.randperm(patches.shape[0])[:population.B]
# # #     population.prototypes = patches[idx].clone().to(device)
# # #     print(f"Prototypes initialisés depuis {patches.shape[0]} patches réels")

# # # Dans train.py — remplacer init_prototypes_from_data

# # # def init_prototypes_from_data(population, images, device, n_samples=200):
# # #     """
# # #     Initialisation par K-means au lieu d'un échantillonnage aléatoire.
# # #     Les prototypes sont initialisés aux centroïdes des clusters de patches.
# # #     """
# # #     from sklearn.cluster import MiniBatchKMeans
# # #     import numpy as np

# # #     print("Extraction des patches pour K-means...")
# # #     imgs_batch = torch.stack(images[:n_samples]).to(device)
# # #     patches    = population.extract_patches_batch(imgs_batch)
# # #     patches    = population.preprocess_patches(patches)
# # #     patches    = patches.reshape(-1, patches.shape[2])  # (N*P, D)
# # #     patches_np = patches.cpu().numpy()

# # #     print(f"  {patches_np.shape[0]} patches extraits — lancement K-means (K={population.B})...")

# # #     # MiniBatchKMeans : beaucoup plus rapide que KMeans classique
# # #     kmeans = MiniBatchKMeans(
# # #         n_clusters  = population.B,
# # #         batch_size  = 4096,
# # #         n_init      = 3,
# # #         max_iter    = 100,
# # #         random_state= 42,
# # #         verbose     = 0
# # #     )
# # #     kmeans.fit(patches_np)

# # #     # Les centroïdes deviennent les prototypes
# # #     centroids = torch.tensor(kmeans.cluster_centers_,
# # #                              dtype=torch.float32, device=device)
# # #     population.prototypes = centroids
# # #     print(f"Prototypes initialisés par K-means depuis {patches_np.shape[0]} patches")



# # def init_prototypes_from_data(population, images, labels, device, n_samples=200):
# #     """
# #     Initialisation K-means séparé par classe.
# #     Chaque classe obtient B//num_classes prototypes.
# #     Les proto_class sont initialisés directement.
# #     """
# #     from sklearn.cluster import MiniBatchKMeans
# #     import numpy as np

# #     n_per_class = population.B // population.num_classes
# #     all_centroids = []

# #     for c in range(population.num_classes):
# #         # Images de cette classe uniquement
# #         class_idx  = [i for i, l in enumerate(labels) if l == c][:n_samples]
# #         class_imgs = torch.stack([images[i] for i in class_idx]).to(device)

# #         patches     = population.extract_patches_batch(class_imgs)
# #         patches_std = population.preprocess_patches(patches)
# #         patches_np  = patches_std.reshape(-1, patches_std.shape[2]).cpu().numpy()

# #         print(f"  Classe {c} : {patches_np.shape[0]} patches → K-means (K={n_per_class})")

# #         kmeans = MiniBatchKMeans(
# #             n_clusters   = n_per_class,
# #             batch_size   = 4096,
# #             n_init       = 3,
# #             max_iter     = 100,
# #             random_state = 42
# #         )
# #         kmeans.fit(patches_np)
# #         all_centroids.append(kmeans.cluster_centers_)

# #     # Concaténer tous les centroïdes
# #     centroids = np.vstack(all_centroids)
# #     population.prototypes = torch.tensor(
# #         centroids, dtype=torch.float32, device=device)

# #     # ✅ Initialiser proto_class directement par classe
# #     for c in range(population.num_classes):
# #         start = c * n_per_class
# #         end   = start + n_per_class
# #         population.proto_class[start:end] = c

# #     print(f"✅ Prototypes initialisés : {n_per_class} par classe")
# #     for c in range(population.num_classes):
# #         n = (population.proto_class == c).sum().item()
# #         print(f"   Classe {c} : {n} prototypes")


# # def run_experiment(train_images, train_labels, val_images, val_labels,
# #                    name, num_classes, epochs=40, lr=0.1,
# #                    num_cells=1600, patch_size=(5, 5),
# #                    theta_init=0.5, device="cuda" , K=1):

# #     print(f"\n{'='*50}")
# #     print(f"EXPÉRIENCE : {name}")
# #     print(f"{'='*50}")

# #     pop = PopulationBFastExact(
# #         num_cells=num_cells,
# #         patch_size=patch_size,
# #         theta_init=theta_init,
# #         beta=5.0,
# #         num_classes=num_classes,
# #         device=device,
# #         K=K 
# #     )
# #     trainer = TrainerFastExact(population=pop, num_classes=num_classes, device=device)
# #     # init_prototypes_from_data(pop, train_images, device, n_samples=200)
# #     init_prototypes_from_data(pop, train_images, train_labels, device, n_samples=200)


# #     best_acc     = 0.0
# #     best_protos  = pop.prototypes.clone()
# #     best_counts  = pop.class_counts.clone()
# #     best_classes = pop.proto_class.clone()
# #     patience, max_patience = 0, 7
# #     history = []

# #     for epoch in range(epochs):
# #         lr_epoch = lr * (0.95 ** epoch)

# #         freeze = (epoch < 5)
# #         trainer.train_batch(
# #         train_images, train_labels,
# #         batch_size=16, lr=lr_epoch,
# #         freeze_classes=freeze
# #         )
# #         pop.reassign_proto_class(train_images, train_labels, device)

        
# #         # trainer.train_batch(train_images, train_labels, batch_size=16, lr=lr_epoch)
# #         # pop.reassign_proto_class(train_images, train_labels, device)

# #         preds   = trainer.predict_batch(val_images, batch_size=32)
# #         correct = sum(p == l for p, l in zip(preds, val_labels) if p is not None)
# #         acc     = correct / len(val_images)
# #         history.append(acc)

# #         if acc > best_acc:
# #             best_acc     = acc
# #             best_protos  = pop.prototypes.clone()
# #             best_counts  = pop.class_counts.clone()
# #             best_classes = pop.proto_class.clone()
# #             patience     = 0
# #             marker       = "✅"
# #         elif acc < best_acc - 0.05:
# #             pop.prototypes   = best_protos.clone()
# #             pop.class_counts = best_counts.clone()
# #             pop.proto_class  = best_classes.clone()
# #             patience        += 1
# #             marker           = f"⚠️  restauré (patience {patience}/{max_patience})"
# #         else:
# #             patience += 1
# #             marker    = f"  (patience {patience}/{max_patience})"

# #         print(f"  Epoch {epoch+1:2d} | Acc: {acc:.4f} | Best: {best_acc:.4f} | "
# #               f"lr: {lr_epoch:.4f} {marker}")

# #         if patience >= max_patience:
# #             print(f"\n  Early stopping à l'epoch {epoch+1}")
# #             break

# #     pop.prototypes   = best_protos.clone()
# #     pop.class_counts = best_counts.clone()
# #     pop.proto_class  = best_classes.clone()
# #     print(f"\n>>> BEST ACCURACY [{name}]: {best_acc:.4f}")
# #     return best_acc, pop, trainer, history

# # ============================================================
# # train.py — Fonctions d'entraînement
# # ============================================================

# import torch
# from model import PopulationBFastExact, TrainerFastExact


# # def init_prototypes_from_data(population, images, device, n_samples=200):
# #     """Initialisation simple depuis patches réels."""
# #     imgs_batch = torch.stack(images[:n_samples]).to(device)
# #     patches    = population.extract_patches_batch(imgs_batch)
# #     patches    = population.preprocess_patches(patches)
# #     patches    = patches.reshape(-1, patches.shape[2])
# #     idx = torch.randperm(patches.shape[0])[:population.B]
# #     population.prototypes = patches[idx].clone().to(device)
# #     population.proto_class.fill_(-1)
# #     print(f"Prototypes initialisés depuis {patches.shape[0]} patches réels")

# def init_prototypes_from_data(population, images, device, n_samples=50):
#         """Initialisation depuis patches réels — par petits batches."""
#         all_patches = []
        
#         # Traiter par batch de 10 images
#         for i in range(0, min(n_samples, len(images)), 10):
#             batch = images[i:i+10]
#             imgs_batch = torch.stack(batch).to(device)
#             patches    = population.extract_patches_batch(imgs_batch)
#             patches    = population.preprocess_patches(patches)
#             patches    = patches.reshape(-1, patches.shape[2])
#             all_patches.append(patches.cpu())  # déplacer sur CPU
#             del imgs_batch, patches
#             torch.cuda.empty_cache()
        
#         # Concaténer sur CPU
#         all_patches = torch.cat(all_patches, dim=0)
        
#         # Échantillonner sur CPU
#         idx = torch.randperm(all_patches.shape[0])[:population.B]
#         population.prototypes = all_patches[idx].clone().to(device)
#         population.proto_class.fill_(-1)
        
#         print(f"Prototypes initialisés depuis {all_patches.shape[0]} patches réels")

# # def run_experiment(train_images, train_labels, val_images, val_labels,
# #                    name, num_classes, epochs=40, lr=0.1,
# #                    num_cells=1600, patch_size=(5, 5),
# #                    theta_init=0.5, K=1, device="cuda"):

# #     print(f"\n{'='*50}")
# #     print(f"EXPÉRIENCE : {name}")
# #     print(f"{'='*50}")

# #     pop = PopulationBFastExact(
# #         num_cells  = num_cells,
# #         patch_size = patch_size,
# #         theta_init = theta_init,
# #         beta       = 5.0,
# #         num_classes= num_classes,
# #         K          = K,
# #         device     = device
# #     )
# #     trainer = TrainerFastExact(population=pop, num_classes=num_classes, device=device)
# #     init_prototypes_from_data(pop, train_images, device, n_samples=200)  # ← sans labels

# #     best_acc     = 0.0
# #     best_protos  = pop.prototypes.clone()
# #     best_counts  = pop.class_counts.clone()
# #     best_classes = pop.proto_class.clone()
# #     patience, max_patience = 0, 7
# #     history = []

# #     for epoch in range(epochs):
# #         lr_epoch = lr * (0.95 ** epoch)
# #         trainer.train_batch(train_images, train_labels, batch_size=4, lr=lr_epoch)
# #         pop.reassign_proto_class(train_images, train_labels, device)

# #         preds   = trainer.predict_batch(val_images, batch_size=32)
# #         correct = sum(p == l for p, l in zip(preds, val_labels) if p is not None)
# #         acc     = correct / len(val_images)
# #         history.append(acc)

# #         if acc > best_acc:
# #             best_acc     = acc
# #             best_protos  = pop.prototypes.clone()
# #             best_counts  = pop.class_counts.clone()
# #             best_classes = pop.proto_class.clone()
# #             patience     = 0
# #             marker       = "✅"
# #         elif acc < best_acc - 0.05:
# #             pop.prototypes   = best_protos.clone()
# #             pop.class_counts = best_counts.clone()
# #             pop.proto_class  = best_classes.clone()
# #             patience        += 1
# #             marker           = f"restauré (patience {patience}/{max_patience})"
# #         else:
# #             patience += 1
# #             marker    = f"  (patience {patience}/{max_patience})"

# #         print(f"  Epoch {epoch+1:2d} | Acc: {acc:.4f} | Best: {best_acc:.4f} | "
# #               f"lr: {lr_epoch:.4f} {marker}")

# #         if patience >= max_patience:
# #             print(f"\n  Early stopping à l'epoch {epoch+1}")
# #             break

# #     pop.prototypes   = best_protos.clone()
# #     pop.class_counts = best_counts.clone()
# #     pop.proto_class  = best_classes.clone()
# #     print(f"\n>>> BEST ACCURACY [{name}]: {best_acc:.4f}")
# #     return best_acc, pop, trainer, history


# def run_experiment(train_images, train_labels, val_images, val_labels,
#                    name, num_classes, epochs=40, lr=0.1,
#                    num_cells=1600, patch_size=(5, 5),
#                    theta_init=0.5, K=1, device="cuda"):

#     print(f"\n{'='*50}")
#     print(f"EXPÉRIENCE : {name}")
#     print(f"{'='*50}")

#     pop = PopulationBFastExact(
#         num_cells  = num_cells,
#         patch_size = patch_size,
#         theta_init = theta_init,
#         beta       = 5.0,
#         num_classes= num_classes,
#         K          = K,
#         device     = device
#     )
#     trainer = TrainerFastExact(population=pop, num_classes=num_classes, device=device)
#     # init_prototypes_from_data(pop, train_images, device, n_samples=200)
#     init_prototypes_from_data(pop, train_images, device, n_samples=50)


#     best_acc     = 0.0
#     best_protos  = pop.prototypes.clone()
#     best_counts  = pop.class_counts.clone()
#     best_classes = pop.proto_class.clone()
#     patience, max_patience = 0, 7
#     history = []

#     for epoch in range(epochs):
#         lr_epoch = lr * (0.95 ** epoch)
        
#         # ── Batch_size réduit pour images 128×128 ─────────
#         trainer.train_batch(train_images, train_labels,
#                             batch_size=2, lr=lr_epoch)
#         pop.reassign_proto_class(train_images, train_labels, device,
#                                  batch_size=2)

#         preds = trainer.predict_batch(val_images, batch_size=2)
#         # ──────────────────────────────────────────────────
        
#         correct = sum(p == l for p, l in zip(preds, val_labels) if p is not None)
#         acc     = correct / len(val_images)
#         history.append(acc)

#         if acc > best_acc:
#             best_acc     = acc
#             best_protos  = pop.prototypes.clone()
#             best_counts  = pop.class_counts.clone()
#             best_classes = pop.proto_class.clone()
#             patience     = 0
#             marker       = "✅"
#         elif acc < best_acc - 0.05:
#             pop.prototypes   = best_protos.clone()
#             pop.class_counts = best_counts.clone()
#             pop.proto_class  = best_classes.clone()
#             patience        += 1
#             marker           = f"restauré (patience {patience}/{max_patience})"
#         else:
#             patience += 1
#             marker    = f"  (patience {patience}/{max_patience})"

#         print(f"  Epoch {epoch+1:2d} | Acc: {acc:.4f} | Best: {best_acc:.4f} | "
#               f"lr: {lr_epoch:.4f} {marker}")

#         if patience >= max_patience:
#             print(f"\n  Early stopping à l'epoch {epoch+1}")
#             break

#     pop.prototypes   = best_protos.clone()
#     pop.class_counts = best_counts.clone()
#     pop.proto_class  = best_classes.clone()
#     print(f"\n>>> BEST ACCURACY [{name}]: {best_acc:.4f}")
#     return best_acc, pop, trainer, history


# ============================================================
# train.py — Multi-scale avec visualisations
# ============================================================

import torch
from model import PopulationBMultiScale, TrainerMultiScale
from interpretability import save_epoch_visualizations


# def init_prototypes_from_data(population, images, device, n_samples=50):
#     """Initialisation multi-échelle."""
#     print(f"Initialisation prototypes multi-échelle (n_samples={n_samples})...")
    
#     for scale_idx, patch_size in enumerate(population.patch_sizes):
#         all_patches = []
        
#         for i in range(0, min(n_samples, len(images)), 10):
#             batch = images[i:i+10]
#             imgs_batch = torch.stack(batch).to(device)
#             patches = population.extract_patches_batch(imgs_batch, patch_size)
#             patches = population.preprocess_patches(patches)
#             patches = patches.reshape(-1, patches.shape[2])
#             all_patches.append(patches.cpu())
#             del imgs_batch, patches
#             torch.cuda.empty_cache()
        
#         all_patches = torch.cat(all_patches, dim=0)
#         B_scale = population.B_per_scale[scale_idx]
#         idx = torch.randperm(all_patches.shape[0])[:B_scale]
#         population.prototypes[scale_idx] = all_patches[idx].clone().to(device)
#         population.proto_class[scale_idx].fill_(-1)
        
#         print(f"  Échelle {scale_idx} ({patch_size[0]}×{patch_size[1]}) : "
#               f"{all_patches.shape[0]} patches → {B_scale} prototypes")


# def run_experiment(train_images, train_labels, val_images, val_labels,
#                    name, num_classes, epochs=40, lr=0.1,
#                    num_cells=6400, patch_sizes=[(5,5), (9,9), (13,13)],
#                    theta_init=0.5, K=1, device="cuda",
#                    save_viz_every=5, viz_dir="figs/epoch_viz"):
#     """
#     Entraînement avec visualisations périodiques.
    
#     Args:
#         save_viz_every: Sauvegarder visualisations tous les N epochs
#         viz_dir: Dossier de base pour visualisations
#     """
#     print(f"\n{'='*50}")
#     print(f"EXPÉRIENCE : {name}")
#     print(f"{'='*50}")

#     pop = PopulationBMultiScale(
#         num_cells   = num_cells,
#         patch_sizes = patch_sizes,
#         theta_init  = theta_init,
#         beta        = 5.0,
#         num_classes = num_classes,
#         K           = K,
#         device      = device
#     )
#     trainer = TrainerMultiScale(population=pop, num_classes=num_classes, device=device)
#     init_prototypes_from_data(pop, train_images, device, n_samples=50)

#     best_acc     = 0.0
#     best_protos  = [p.clone() for p in pop.prototypes]
#     best_counts  = [c.clone() for c in pop.class_counts]
#     best_classes = [c.clone() for c in pop.proto_class]
#     patience, max_patience = 0, 7
#     history = []

#     for epoch in range(epochs):
#         lr_epoch = lr * (0.95 ** epoch)
        
#         trainer.train_batch(train_images, train_labels, batch_size=2, lr=lr_epoch)
#         pop.reassign_proto_class(train_images, train_labels, device, batch_size=2)
#         preds = trainer.predict_batch(val_images, batch_size=4)
        
#         correct = sum(p == l for p, l in zip(preds, val_labels) if p is not None)
#         acc     = correct / len(val_images)
#         history.append(acc)

#         if acc > best_acc:
#             best_acc     = acc
#             best_protos  = [p.clone() for p in pop.prototypes]
#             best_counts  = [c.clone() for c in pop.class_counts]
#             best_classes = [c.clone() for c in pop.proto_class]
#             patience     = 0
#             marker       = "✅"
#         elif acc < best_acc - 0.05:
#             pop.prototypes   = [p.clone() for p in best_protos]
#             pop.class_counts = [c.clone() for c in best_counts]
#             pop.proto_class  = [c.clone() for c in best_classes]
#             patience        += 1
#             marker           = f"restauré (patience {patience}/{max_patience})"
#         else:
#             patience += 1
#             marker    = f"  (patience {patience}/{max_patience})"

#         print(f"  Epoch {epoch+1:2d} | Acc: {acc:.4f} | Best: {best_acc:.4f} | "
#               f"lr: {lr_epoch:.4f} {marker}")
        
#         # ── Sauvegarder visualisations périodiquement ─────
#         if (epoch + 1) % save_viz_every == 0 or epoch == 0:
#             save_epoch_visualizations(
#                 pop=pop,
#                 trainer=trainer,
#                 val_images=val_images,
#                 val_labels=val_labels,
#                 epoch=epoch + 1,
#                 class_names=["Cancer", "Normal"],
#                 save_dir=viz_dir,
#                 n_examples=5
#             )

#         if patience >= max_patience:
#             print(f"\n  Early stopping à l'epoch {epoch+1}")
#             break

#     pop.prototypes   = [p.clone() for p in best_protos]
#     pop.class_counts = [c.clone() for c in best_counts]
#     pop.proto_class  = [c.clone() for c in best_classes]
#     print(f"\n>>> BEST ACCURACY [{name}]: {best_acc:.4f}")
#     return best_acc, pop, trainer, history



# ============================================================

# ============================================================
# train.py — Multi-scale avec intensité
# ============================================================

import torch
from model import PopulationBMultiScale, TrainerMultiScale
from interpretability import save_epoch_visualizations


# def init_prototypes_from_data(population, images, device, n_samples=50):
#     """Initialisation multi-échelle AVEC intensité."""
#     print(f"Initialisation prototypes (intensité: {population.use_intensity}, n_samples={n_samples})...")
    
#     for scale_idx, patch_size in enumerate(population.patch_sizes):
#         all_patches = []
        
#         # Traiter par petits batches
#         for i in range(0, min(n_samples, len(images)), 10):
#             batch = images[i:i+10]
#             imgs_batch = torch.stack(batch).to(device)
#             patches = population.extract_patches_batch(imgs_batch, patch_size)
            
#             # ✅ CHANGEMENT : Préprocesser AVEC intensité
#             patches = population.preprocess_patches(patches, keep_intensity=True)
            
#             patches = patches.reshape(-1, patches.shape[2])
#             all_patches.append(patches.cpu())
#             del imgs_batch, patches
#             torch.cuda.empty_cache()
        
#         all_patches = torch.cat(all_patches, dim=0)
#         B_scale = population.B_per_scale[scale_idx]
#         idx = torch.randperm(all_patches.shape[0])[:B_scale]
#         population.prototypes[scale_idx] = all_patches[idx].clone().to(device)
#         population.proto_class[scale_idx].fill_(-1)
        
#         D_feat = all_patches.shape[1]
#         print(f"  Échelle {scale_idx} ({patch_size[0]}×{patch_size[1]}) : "
#               f"{all_patches.shape[0]} patches → {B_scale} prototypes, {D_feat} features")


# Dans train.py : init_prototypes_from_data
# Dans train.py : Modifier appel


# def init_prototypes_from_data(population, images, labels, device, n_samples=50):
#     """Initialisation ÉQUILIBRÉE 50/50 Cancer/Normal."""
#     print(f"Initialisation prototypes équilibrée (intensité: {population.use_intensity})...")
    
#     # ✅ Séparer images par classe
#     cancer_images = [img for img, lbl in zip(images, labels) if lbl == 0]
#     normal_images = [img for img, lbl in zip(images, labels) if lbl == 1]
    
#     for scale_idx, patch_size in enumerate(population.patch_sizes):
#         all_patches_cancer = []
#         all_patches_normal = []
        
#         # ✅ Extraire patches Cancer
#         for i in range(0, min(n_samples//2, len(cancer_images)), 10):
#             batch = cancer_images[i:i+10]
#             imgs_batch = torch.stack(batch).to(device)
#             patches = population.extract_patches_batch(imgs_batch, patch_size)
#             patches = population.preprocess_patches(patches, keep_intensity=True)
#             all_patches_cancer.append(patches.reshape(-1, patches.shape[2]).cpu())
#             del imgs_batch, patches
#             torch.cuda.empty_cache()
        
#         # ✅ Extraire patches Normal
#         for i in range(0, min(n_samples//2, len(normal_images)), 10):
#             batch = normal_images[i:i+10]
#             imgs_batch = torch.stack(batch).to(device)
#             patches = population.extract_patches_batch(imgs_batch, patch_size)
#             patches = population.preprocess_patches(patches, keep_intensity=True)
#             all_patches_normal.append(patches.reshape(-1, patches.shape[2]).cpu())
#             del imgs_batch, patches
#             torch.cuda.empty_cache()
        
#         all_patches_cancer = torch.cat(all_patches_cancer, dim=0)
#         all_patches_normal = torch.cat(all_patches_normal, dim=0)
        
#         B_scale = population.B_per_scale[scale_idx]
#         B_cancer = B_scale // 2
#         B_normal = B_scale - B_cancer
        
#         # ✅ Initialiser 50% Cancer, 50% Normal
#         idx_cancer = torch.randperm(all_patches_cancer.shape[0])[:B_cancer]
#         idx_normal = torch.randperm(all_patches_normal.shape[0])[:B_normal]
        
#         protos_init = torch.cat([
#             all_patches_cancer[idx_cancer],
#             all_patches_normal[idx_normal]
#         ], dim=0).to(device)
        
#         population.prototypes[scale_idx] = protos_init
#         population.proto_class[scale_idx].fill_(-1)
        
#         print(f"  Échelle {scale_idx} ({patch_size[0]}×{patch_size[1]}) : "
#               f"{B_cancer} Cancer + {B_normal} Normal = {B_scale} protos")


# def init_prototypes_from_data(population, images, labels, device, n_samples=50):
#     for scale_idx, patch_size in enumerate(population.patch_sizes):
#         all_patches_cancer = []
#         all_patches_normal = []
        
#         cancer_images = [img for img, lbl in zip(images, labels) if lbl == 0]
#         normal_images = [img for img, lbl in zip(images, labels) if lbl == 1]
        
#         for imgs, patches_list in [(cancer_images, all_patches_cancer), 
#                                      (normal_images, all_patches_normal)]:
#             for i in range(0, min(n_samples//2, len(imgs)), 10):
#                 batch = imgs[i:i+10]
#                 imgs_batch = torch.stack(batch).to(device)
#                 patches = population.extract_patches_batch(imgs_batch, patch_size)
                
#                 # ✅ FILTRER patches informatifs (variance > seuil)
#                 patches_raw = patches  # Avant normalisation
#                 variance = patches_raw.var(dim=-1)  # Variance par patch
                
#                 # Garder seulement patches avec texture (variance > 0.01)
#                 informative_mask = variance > 0.01
#                 patches_filtered = patches[informative_mask]
                
#                 if patches_filtered.shape[0] == 0:
#                     continue
                
#                 # Normaliser patches informatifs
#                 patches_std = population.preprocess_patches(
#                     patches_filtered.unsqueeze(0), keep_intensity=True
#                 ).squeeze(0)
                
#                 patches_list.append(patches_std.cpu())


# def init_prototypes_from_data(population, images, labels, device, n_samples=50):
#     """Initialisation ÉQUILIBRÉE + patches informatifs."""
#     print(f"Initialisation (intensité: {population.use_intensity}, filtre variance)...")
    
#     cancer_images = [img for img, lbl in zip(images, labels) if lbl == 0]
#     normal_images = [img for img, lbl in zip(images, labels) if lbl == 1]
    
#     for scale_idx, patch_size in enumerate(population.patch_sizes):
#         all_patches_cancer = []
#         all_patches_normal = []
        
#         for imgs, patches_list in [(cancer_images, all_patches_cancer), 
#                                      (normal_images, all_patches_normal)]:
#             for i in range(0, min(n_samples//2, len(imgs)), 10):
#                 batch = imgs[i:i+10]
#                 imgs_batch = torch.stack(batch).to(device)
#                 patches = population.extract_patches_batch(imgs_batch, patch_size)
                
#                 # ✅ Filtrer patches informatifs
#                 variance = patches.var(dim=-1)
#                 informative = variance > 0.01
#                 patches_filtered = patches[informative]
                
#                 if patches_filtered.shape[0] == 0:
#                     continue
                
#                 # Normaliser
#                 patches_std = population.preprocess_patches(
#                     patches_filtered.unsqueeze(0), keep_intensity=True
#                 ).squeeze(0)
                
#                 patches_list.append(patches_std.cpu())
#                 del imgs_batch, patches
#                 torch.cuda.empty_cache()
        
#         # ✅ Concaténer et initialiser 50/50
#         all_patches_cancer = torch.cat(all_patches_cancer, dim=0) if all_patches_cancer else torch.empty(0, 26)
#         all_patches_normal = torch.cat(all_patches_normal, dim=0) if all_patches_normal else torch.empty(0, 26)
        
#         B_scale = population.B_per_scale[scale_idx]
#         B_cancer = B_scale // 2
#         B_normal = B_scale - B_cancer
        
#         idx_cancer = torch.randperm(all_patches_cancer.shape[0])[:B_cancer] if all_patches_cancer.shape[0] > 0 else torch.empty(0, dtype=torch.long)
#         idx_normal = torch.randperm(all_patches_normal.shape[0])[:B_normal] if all_patches_normal.shape[0] > 0 else torch.empty(0, dtype=torch.long)
        
#         protos_init = torch.cat([
#             all_patches_cancer[idx_cancer] if len(idx_cancer) > 0 else torch.empty(0, all_patches_cancer.shape[1]),
#             all_patches_normal[idx_normal] if len(idx_normal) > 0 else torch.empty(0, all_patches_normal.shape[1])
#         ], dim=0).to(device)
        
#         population.prototypes[scale_idx] = protos_init
#         population.proto_class[scale_idx].fill_(-1)
        
#         print(f"  Échelle {scale_idx} ({patch_size[0]}×{patch_size[1]}) : "
#               f"{len(idx_cancer)} Cancer + {len(idx_normal)} Normal = {B_scale} protos")


def init_prototypes_from_data(population, images, labels, device, n_samples=50):
    """Init avec filtrage STRICT adaptatif."""
    print(f"Initialisation STRICTE (intensité: {population.use_intensity})...")
    
    cancer_images = [img for img, lbl in zip(images, labels) if lbl == 0]
    normal_images = [img for img, lbl in zip(images, labels) if lbl == 1]
    
    for scale_idx, patch_size in enumerate(population.patch_sizes):
        all_patches_cancer = []
        all_patches_normal = []
        
        for imgs, patches_list in [(cancer_images, all_patches_cancer), 
                                     (normal_images, all_patches_normal)]:
            for i in range(0, min(n_samples//2, len(imgs)), 10):
                batch = imgs[i:i+10]
                imgs_batch = torch.stack(batch).to(device)
                patches = population.extract_patches_batch(imgs_batch, patch_size)
                
                # ✅ Calculer variance
                variance = patches.var(dim=-1)
                
                # ✅ SEUIL ADAPTATIF : garder top 25% des patches les plus texturés
                if variance.numel() > 0:
                    threshold = torch.quantile(variance.flatten(), 0.75)
                    informative = variance > threshold
                else:
                    continue
                
                patches_filtered = patches[informative]
                
                if patches_filtered.shape[0] == 0:
                    continue
                
                # Normaliser
                patches_std = population.preprocess_patches(
                    patches_filtered.unsqueeze(0), keep_intensity=True
                ).squeeze(0)
                
                patches_list.append(patches_std.cpu())
                del imgs_batch, patches
                torch.cuda.empty_cache()
        
        # Concaténer
        all_patches_cancer = torch.cat(all_patches_cancer, dim=0) if all_patches_cancer else torch.empty(0, 26)
        all_patches_normal = torch.cat(all_patches_normal, dim=0) if all_patches_normal else torch.empty(0, 26)
        
        B_scale = population.B_per_scale[scale_idx]
        B_cancer = B_scale // 2
        B_normal = B_scale - B_cancer
        
        # Échantillonner
        idx_cancer = torch.randperm(all_patches_cancer.shape[0])[:B_cancer] if all_patches_cancer.shape[0] > 0 else torch.empty(0, dtype=torch.long)
        idx_normal = torch.randperm(all_patches_normal.shape[0])[:B_normal] if all_patches_normal.shape[0] > 0 else torch.empty(0, dtype=torch.long)
        
        protos_init = torch.cat([
            all_patches_cancer[idx_cancer] if len(idx_cancer) > 0 else torch.empty(0, all_patches_cancer.shape[1] if all_patches_cancer.shape[0] > 0 else 26),
            all_patches_normal[idx_normal] if len(idx_normal) > 0 else torch.empty(0, all_patches_normal.shape[1] if all_patches_normal.shape[0] > 0 else 26)
        ], dim=0).to(device)
        
        population.prototypes[scale_idx] = protos_init
        population.proto_class[scale_idx].fill_(-1)
        
        print(f"  Échelle {scale_idx} ({patch_size[0]}×{patch_size[1]}) : "
              f"{len(idx_cancer)} Cancer + {len(idx_normal)} Normal = {B_scale} protos [Top 25%]")


def run_experiment(train_images, train_labels, val_images, val_labels,
                   name, num_classes, epochs=40, lr=0.1,
                   num_cells=6400, patch_sizes=[(5,5), (9,9), (13,13)],
                   theta_init=0.5, K=1, device="cuda",
                   use_intensity=True,  # ← NOUVEAU
                   save_viz_every=5, viz_dir="figs/epoch_viz"):
    """
    Entraînement multi-échelle avec feature intensité.
    
    Args:
        use_intensity: Activer la feature intensité (texture + contraste)
    """

    # Sélectionner 10 images FIXES pour tout le training
    viz_indices = [0, 10, 20, 30, 40, 100, 150, 200, 250, 300]
    viz_images = [val_images[i] for i in viz_indices]
    viz_labels = [val_labels[i] for i in viz_indices]


    print(f"\n{'='*50}")
    print(f"EXPÉRIENCE : {name}")
    print(f"{'='*50}")

    pop = PopulationBMultiScale(
        num_cells     = num_cells,
        patch_sizes   = patch_sizes,
        theta_init    = theta_init,
        beta          = 5.0,
        num_classes   = num_classes,
        K             = K,
        use_intensity = use_intensity,  # ← ACTIVER ICI
        device        = device
    )
    trainer = TrainerMultiScale(population=pop, num_classes=num_classes, device=device)
    # init_prototypes_from_data(pop, train_images, device, n_samples=50)
    # Dans train.py : Modifier appel
    init_prototypes_from_data(pop, train_images, train_labels, device, n_samples=50)

    # ✅ Visualiser EPOCH 0 (juste après init, avant training)
    print("\n[Epoch 0] Visualisation AVANT entraînement...")
    save_epoch_visualizations(
        pop=pop, trainer=trainer,
        val_images=viz_images, val_labels=viz_labels,
        epoch=0, class_names=["Cancer", "Normal"],
        save_dir=viz_dir, n_examples=10
    )

    best_acc     = 0.0
    best_protos  = [p.clone() for p in pop.prototypes]
    best_counts  = [c.clone() for c in pop.class_counts]
    best_classes = [c.clone() for c in pop.proto_class]
    patience, max_patience = 0, 7
    history = []

    for epoch in range(epochs):
        lr_epoch = lr * (0.95 ** epoch)
        
        trainer.train_batch(train_images, train_labels, batch_size=2, lr=lr_epoch)
        pop.reassign_proto_class(train_images, train_labels, device, batch_size=2)
        preds = trainer.predict_batch(val_images, batch_size=4)
        
        correct = sum(p == l for p, l in zip(preds, val_labels) if p is not None)
        acc     = correct / len(val_images)
        history.append(acc)

        if acc > best_acc:
            best_acc     = acc
            best_protos  = [p.clone() for p in pop.prototypes]
            best_counts  = [c.clone() for c in pop.class_counts]
            best_classes = [c.clone() for c in pop.proto_class]
            patience     = 0
            marker       = "✅"
        elif acc < best_acc - 0.05:
            pop.prototypes   = [p.clone() for p in best_protos]
            pop.class_counts = [c.clone() for c in best_counts]
            pop.proto_class  = [c.clone() for c in best_classes]
            patience        += 1
            marker           = f"restauré (patience {patience}/{max_patience})"
        else:
            patience += 1
            marker    = f"  (patience {patience}/{max_patience})"

        print(f"  Epoch {epoch+1:2d} | Acc: {acc:.4f} | Best: {best_acc:.4f} | "
              f"lr: {lr_epoch:.4f} {marker}")
        
        # Visualisations périodiques
        if (epoch + 1) % save_viz_every == 0 or epoch == 0:
            save_epoch_visualizations(
                pop=pop,
                trainer=trainer,
                val_images=viz_images,
                val_labels=viz_labels,
                epoch=epoch + 1,
                class_names=["Cancer", "Normal"],
                save_dir=viz_dir,
                n_examples=10
            )

        if patience >= max_patience:
            print(f"\n  Early stopping à l'epoch {epoch+1}")
            break

    pop.prototypes   = [p.clone() for p in best_protos]
    pop.class_counts = [c.clone() for c in best_counts]
    pop.proto_class  = [c.clone() for c in best_classes]
    print(f"\n>>> BEST ACCURACY [{name}]: {best_acc:.4f}")
    return best_acc, pop, trainer, history


#---------------------------------------------