# # ============================================================
# # train.py — Fonctions d'entraînement
# # ============================================================

# import torch
# from model import PopulationBFastExact, TrainerFastExact


# # def init_prototypes_from_data(population, images, device, n_samples=200):
# #     import torch.nn.functional as F
# #     imgs_batch = torch.stack(images[:n_samples]).to(device)
# #     patches    = population.extract_patches_batch(imgs_batch)
# #     patches    = population.preprocess_patches(patches)
# #     patches    = patches.reshape(-1, patches.shape[2])
# #     idx = torch.randperm(patches.shape[0])[:population.B]
# #     population.prototypes = patches[idx].clone().to(device)
# #     print(f"Prototypes initialisés depuis {patches.shape[0]} patches réels")

# # Dans train.py — remplacer init_prototypes_from_data

# # def init_prototypes_from_data(population, images, device, n_samples=200):
# #     """
# #     Initialisation par K-means au lieu d'un échantillonnage aléatoire.
# #     Les prototypes sont initialisés aux centroïdes des clusters de patches.
# #     """
# #     from sklearn.cluster import MiniBatchKMeans
# #     import numpy as np

# #     print("Extraction des patches pour K-means...")
# #     imgs_batch = torch.stack(images[:n_samples]).to(device)
# #     patches    = population.extract_patches_batch(imgs_batch)
# #     patches    = population.preprocess_patches(patches)
# #     patches    = patches.reshape(-1, patches.shape[2])  # (N*P, D)
# #     patches_np = patches.cpu().numpy()

# #     print(f"  {patches_np.shape[0]} patches extraits — lancement K-means (K={population.B})...")

# #     # MiniBatchKMeans : beaucoup plus rapide que KMeans classique
# #     kmeans = MiniBatchKMeans(
# #         n_clusters  = population.B,
# #         batch_size  = 4096,
# #         n_init      = 3,
# #         max_iter    = 100,
# #         random_state= 42,
# #         verbose     = 0
# #     )
# #     kmeans.fit(patches_np)

# #     # Les centroïdes deviennent les prototypes
# #     centroids = torch.tensor(kmeans.cluster_centers_,
# #                              dtype=torch.float32, device=device)
# #     population.prototypes = centroids
# #     print(f"Prototypes initialisés par K-means depuis {patches_np.shape[0]} patches")



# def init_prototypes_from_data(population, images, labels, device, n_samples=200):
#     """
#     Initialisation K-means séparé par classe.
#     Chaque classe obtient B//num_classes prototypes.
#     Les proto_class sont initialisés directement.
#     """
#     from sklearn.cluster import MiniBatchKMeans
#     import numpy as np

#     n_per_class = population.B // population.num_classes
#     all_centroids = []

#     for c in range(population.num_classes):
#         # Images de cette classe uniquement
#         class_idx  = [i for i, l in enumerate(labels) if l == c][:n_samples]
#         class_imgs = torch.stack([images[i] for i in class_idx]).to(device)

#         patches     = population.extract_patches_batch(class_imgs)
#         patches_std = population.preprocess_patches(patches)
#         patches_np  = patches_std.reshape(-1, patches_std.shape[2]).cpu().numpy()

#         print(f"  Classe {c} : {patches_np.shape[0]} patches → K-means (K={n_per_class})")

#         kmeans = MiniBatchKMeans(
#             n_clusters   = n_per_class,
#             batch_size   = 4096,
#             n_init       = 3,
#             max_iter     = 100,
#             random_state = 42
#         )
#         kmeans.fit(patches_np)
#         all_centroids.append(kmeans.cluster_centers_)

#     # Concaténer tous les centroïdes
#     centroids = np.vstack(all_centroids)
#     population.prototypes = torch.tensor(
#         centroids, dtype=torch.float32, device=device)

#     # ✅ Initialiser proto_class directement par classe
#     for c in range(population.num_classes):
#         start = c * n_per_class
#         end   = start + n_per_class
#         population.proto_class[start:end] = c

#     print(f"✅ Prototypes initialisés : {n_per_class} par classe")
#     for c in range(population.num_classes):
#         n = (population.proto_class == c).sum().item()
#         print(f"   Classe {c} : {n} prototypes")


# def run_experiment(train_images, train_labels, val_images, val_labels,
#                    name, num_classes, epochs=40, lr=0.1,
#                    num_cells=1600, patch_size=(5, 5),
#                    theta_init=0.5, device="cuda" , K=1):

#     print(f"\n{'='*50}")
#     print(f"EXPÉRIENCE : {name}")
#     print(f"{'='*50}")

#     pop = PopulationBFastExact(
#         num_cells=num_cells,
#         patch_size=patch_size,
#         theta_init=theta_init,
#         beta=5.0,
#         num_classes=num_classes,
#         device=device,
#         K=K 
#     )
#     trainer = TrainerFastExact(population=pop, num_classes=num_classes, device=device)
#     # init_prototypes_from_data(pop, train_images, device, n_samples=200)
#     init_prototypes_from_data(pop, train_images, train_labels, device, n_samples=200)


#     best_acc     = 0.0
#     best_protos  = pop.prototypes.clone()
#     best_counts  = pop.class_counts.clone()
#     best_classes = pop.proto_class.clone()
#     patience, max_patience = 0, 7
#     history = []

#     for epoch in range(epochs):
#         lr_epoch = lr * (0.95 ** epoch)

#         freeze = (epoch < 5)
#         trainer.train_batch(
#         train_images, train_labels,
#         batch_size=16, lr=lr_epoch,
#         freeze_classes=freeze
#         )
#         pop.reassign_proto_class(train_images, train_labels, device)

        
#         # trainer.train_batch(train_images, train_labels, batch_size=16, lr=lr_epoch)
#         # pop.reassign_proto_class(train_images, train_labels, device)

#         preds   = trainer.predict_batch(val_images, batch_size=32)
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
#             marker           = f"⚠️  restauré (patience {patience}/{max_patience})"
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
# train.py — Fonctions d'entraînement
# ============================================================

import torch
from model import PopulationBFastExact, TrainerFastExact


def init_prototypes_from_data(population, images, device, n_samples=200):
    """Initialisation simple depuis patches réels."""
    imgs_batch = torch.stack(images[:n_samples]).to(device)
    patches    = population.extract_patches_batch(imgs_batch)
    patches    = population.preprocess_patches(patches)
    patches    = patches.reshape(-1, patches.shape[2])
    idx = torch.randperm(patches.shape[0])[:population.B]
    population.prototypes = patches[idx].clone().to(device)
    population.proto_class.fill_(-1)
    print(f"Prototypes initialisés depuis {patches.shape[0]} patches réels")


def run_experiment(train_images, train_labels, val_images, val_labels,
                   name, num_classes, epochs=40, lr=0.1,
                   num_cells=1600, patch_size=(5, 5),
                   theta_init=0.5, K=1, device="cuda"):

    print(f"\n{'='*50}")
    print(f"EXPÉRIENCE : {name}")
    print(f"{'='*50}")

    pop = PopulationBFastExact(
        num_cells  = num_cells,
        patch_size = patch_size,
        theta_init = theta_init,
        beta       = 5.0,
        num_classes= num_classes,
        K          = K,
        device     = device
    )
    trainer = TrainerFastExact(population=pop, num_classes=num_classes, device=device)
    init_prototypes_from_data(pop, train_images, device, n_samples=200)  # ← sans labels

    best_acc     = 0.0
    best_protos  = pop.prototypes.clone()
    best_counts  = pop.class_counts.clone()
    best_classes = pop.proto_class.clone()
    patience, max_patience = 0, 7
    history = []

    for epoch in range(epochs):
        lr_epoch = lr * (0.95 ** epoch)
        trainer.train_batch(train_images, train_labels, batch_size=16, lr=lr_epoch)
        pop.reassign_proto_class(train_images, train_labels, device)

        preds   = trainer.predict_batch(val_images, batch_size=32)
        correct = sum(p == l for p, l in zip(preds, val_labels) if p is not None)
        acc     = correct / len(val_images)
        history.append(acc)

        if acc > best_acc:
            best_acc     = acc
            best_protos  = pop.prototypes.clone()
            best_counts  = pop.class_counts.clone()
            best_classes = pop.proto_class.clone()
            patience     = 0
            marker       = "✅"
        elif acc < best_acc - 0.05:
            pop.prototypes   = best_protos.clone()
            pop.class_counts = best_counts.clone()
            pop.proto_class  = best_classes.clone()
            patience        += 1
            marker           = f"restauré (patience {patience}/{max_patience})"
        else:
            patience += 1
            marker    = f"  (patience {patience}/{max_patience})"

        print(f"  Epoch {epoch+1:2d} | Acc: {acc:.4f} | Best: {best_acc:.4f} | "
              f"lr: {lr_epoch:.4f} {marker}")

        if patience >= max_patience:
            print(f"\n  Early stopping à l'epoch {epoch+1}")
            break

    pop.prototypes   = best_protos.clone()
    pop.class_counts = best_counts.clone()
    pop.proto_class  = best_classes.clone()
    print(f"\n>>> BEST ACCURACY [{name}]: {best_acc:.4f}")
    return best_acc, pop, trainer, history