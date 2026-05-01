# # ============================================================
# # model.py — Version 256×256 avec normalisation
# # ============================================================

# import torch
# import torch.nn.functional as F

# class PopulationBFastExact:
#     def __init__(self, num_cells=1600, patch_size=(5, 5),
#                  theta_init=0.5, beta=5.0, num_classes=2,
#                  K=1, device="cuda"):
#         self.device      = device
#         self.B           = num_cells
#         self.patch_size  = patch_size
#         self.D           = patch_size[0] * patch_size[1]
#         self.beta        = beta
#         self.theta_init  = theta_init
#         self.num_classes = num_classes
#         self.K           = K

#         self.prototypes   = torch.randn(self.B, self.D, device=device) * 0.1
#         self.proto_class  = torch.full((self.B,), -1, dtype=torch.long, device=device)
#         self.class_counts = torch.zeros(self.B, num_classes, device=device)

#     def extract_patches_batch(self, imgs):
#         patches = F.unfold(imgs.unsqueeze(1), kernel_size=self.patch_size, stride=1)
#         return patches.transpose(1, 2)

#     def preprocess_patches(self, patches):
#         """Normalisation z-score — ACTIVE."""
#         mean = patches.mean(dim=-1, keepdim=True)
#         std  = patches.std(dim=-1, keepdim=True).clamp(min=1e-6)
#         return (patches - mean) / std

#     def process_batch(self, imgs):
#         imgs        = imgs.to(self.device)
#         patches     = self.extract_patches_batch(imgs)
#         patches_std = self.preprocess_patches(patches)
#         protos      = self.preprocess_patches(self.prototypes.unsqueeze(0)).squeeze(0)

#         N, P, D = patches_std.shape
#         B       = protos.shape[0]

#         patches_sq = (patches_std ** 2).sum(dim=-1)
#         protos_sq  = (protos ** 2).sum(dim=-1)
#         dot        = torch.einsum("npd,bd->nbp", patches_std, protos)
#         dists_sq   = (patches_sq.unsqueeze(1) + protos_sq.view(1, B, 1) - 2 * dot).clamp(min=0)

#         K = self.K
#         topk_dists, topk_idx = dists_sq.topk(K, dim=2, largest=False)
#         sim       = torch.exp(-topk_dists.mean(dim=2) / self.D ** 0.5)
#         activated = (sim >= self.theta_init).bool()

#         topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)
#         patches_exp  = patches_std.unsqueeze(1).expand(-1, B, -1, -1)
#         z = patches_exp.gather(2, topk_idx_exp).mean(dim=2)

#         return activated, z

#     def update_batch_lvq(self, activated, z, labels, lr=0.05):
#         N, B, D = z.shape
#         for i in range(N):
#             lbl = labels[i].item()
#             act = activated[i]
#             if not act.any():
#                 continue

#             self.class_counts[act] *= 0.99
#             self.class_counts[act, lbl] += 1
#             self.proto_class[act] = self.class_counts[act].argmax(dim=1)

#             correct   = act & (self.proto_class == lbl)
#             incorrect = act & (self.proto_class != lbl)

#             if correct.any():
#                 self.prototypes[correct] += lr * (z[i][correct] - self.prototypes[correct])
#             if incorrect.any():
#                 self.prototypes[incorrect] -= lr * (z[i][incorrect] - self.prototypes[incorrect])

#         self.prototypes.clamp_(-5.0, 5.0)

#     def get_vote_weights(self):
#         total = self.class_counts.sum(dim=1, keepdim=True).clamp(min=1)
#         freq  = self.class_counts / total
#         max_freq  = freq.max(dim=1).values
#         mean_freq = freq.mean(dim=1)
#         weights   = (max_freq - mean_freq) * 2
#         return weights, freq

#     def reassign_proto_class(self, train_images, train_labels, device, batch_size=2):  # ← 2 au lieu de 8
#         self.class_counts.zero_()
#         images_t = torch.stack(train_images).to(device)
#         labels_t = torch.tensor(train_labels, device=device, dtype=torch.long)

#         for start in range(0, len(images_t), batch_size):
#             end = min(start + batch_size, len(images_t))
#             activated, _ = self.process_batch(images_t[start:end])
#             lbls_b = labels_t[start:end]
#             for i in range(end - start):
#                 self.class_counts[activated[i], lbls_b[i].item()] += 1

#         assigned = self.class_counts.sum(dim=1) > 0
#         n_assigned = assigned.sum().item()
#         print(f"    [Reassign] {n_assigned}/{self.B} prototypes assignés")
        
#         class_freq        = self.class_counts.sum(dim=0).clamp(min=1)
#         counts_normalized = self.class_counts / class_freq.unsqueeze(0)
#         self.proto_class[assigned]  = counts_normalized[assigned].argmax(dim=1)
#         self.proto_class[~assigned] = -1


# class TrainerFastExact:
#     def __init__(self, population, num_classes=2, device="cuda"):
#         self.population  = population
#         self.device      = device
#         self.num_classes = num_classes

#     def train_batch(self, images, labels, batch_size=2, lr=0.05):  # ← 2 au lieu de 4
#         images_t = torch.stack(images).to(self.device)
#         labels_t = torch.tensor(labels, device=self.device, dtype=torch.long)
#         for start in range(0, len(images_t), batch_size):
#             end = min(start + batch_size, len(images_t))
#             activated, z = self.population.process_batch(images_t[start:end])
#             if not activated.any():
#                 continue
#             self.population.update_batch_lvq(activated, z, labels_t[start:end], lr)

#     def predict_batch(self, images, batch_size=4):  # ← 4 au lieu de 8
#         images_t  = torch.stack(images).to(self.device)
#         all_preds = []
#         weights, freq = self.population.get_vote_weights()

#         for start in range(0, len(images_t), batch_size):
#             end = min(start + batch_size, len(images_t))
#             activated, _ = self.population.process_batch(images_t[start:end])

#             for i in range(end - start):
#                 act_i = activated[i]
#                 valid = act_i & (self.population.proto_class >= 0)

#                 if not valid.any():
#                     all_preds.append(None)
#                     continue

#                 active_freq    = freq[valid]
#                 active_weights = weights[valid]
#                 votes = (active_freq * active_weights.unsqueeze(1)).sum(dim=0)

#                 if votes.sum() == 0:
#                     active_classes = self.population.proto_class[valid]
#                     all_preds.append(torch.mode(active_classes).values.item())
#                 else:
#                     all_preds.append(votes.argmax().item())

#         return all_preds

# ============================================================
# model.py — Multi-scale patches
# ============================================================

# import torch
# import torch.nn.functional as F

# # ============================================================
# # model.py — Multi-scale AVEC intensité
# # ============================================================

# import torch
# import torch.nn.functional as F


# class PopulationBMultiScale:
#     def __init__(self, num_cells=6400, patch_sizes=[(5,5), (9,9), (13,13)],
#                  theta_init=0.5, beta=5.0, num_classes=2, K=1, 
#                  use_intensity=True, device="cuda"):  # ← Nouveau paramètre
#         self.device       = device
#         self.B            = num_cells
#         self.patch_sizes  = patch_sizes
#         self.n_scales     = len(patch_sizes)
#         self.beta         = beta
#         self.theta_init   = theta_init
#         self.num_classes  = num_classes
#         self.K            = K
#         self.use_intensity = use_intensity  # ← Feature intensité ON/OFF
        
#         # Répartir prototypes
#         self.B_per_scale = [num_cells // self.n_scales] * self.n_scales
#         self.B_per_scale[-1] += num_cells - sum(self.B_per_scale)
        
#         # Créer prototypes par échelle
#         self.prototypes   = []
#         self.proto_class  = []
#         self.class_counts = []
        
#         for i, (ph, pw) in enumerate(patch_sizes):
#             D_base = ph * pw
#             # Si use_intensity, ajouter 1 dimension pour l'intensité moyenne
#             D = D_base + 1 if use_intensity else D_base
#             B_scale = self.B_per_scale[i]
            
#             self.prototypes.append(
#                 torch.randn(B_scale, D, device=device) * 0.1
#             )
#             self.proto_class.append(
#                 torch.full((B_scale,), -1, dtype=torch.long, device=device)
#             )
#             self.class_counts.append(
#                 torch.zeros(B_scale, num_classes, device=device)
#             )
        
#         print(f"[Multi-scale] {self.n_scales} échelles (intensité: {use_intensity}):")
#         for i, ps in enumerate(patch_sizes):
#             D_feat = (ps[0] * ps[1] + 1) if use_intensity else (ps[0] * ps[1])
#             print(f"  Échelle {i}: {ps[0]}×{ps[1]} → {self.B_per_scale[i]} protos, {D_feat} features")

#     def extract_patches_batch(self, imgs, patch_size):
#         patches = F.unfold(imgs.unsqueeze(1), kernel_size=patch_size, stride=1)
#         return patches.transpose(1, 2)

#     def preprocess_patches(self, patches, keep_intensity=False):
#         """
#         Normalise z-score + optionnellement ajoute intensité.
        
#         Args:
#             patches: (N, P, D) tensor
#             keep_intensity: Si True, ajoute l'intensité moyenne avant z-score
        
#         Returns:
#             (N, P, D) ou (N, P, D+1) si keep_intensity
#         """
#         if not self.use_intensity or not keep_intensity:
#             # Version classique : z-score seul
#             mean = patches.mean(dim=-1, keepdim=True)
#             std  = patches.std(dim=-1, keepdim=True).clamp(min=1e-6)
#             return (patches - mean) / std
        
#         # ── Nouvelle version avec intensité ──────────────────
#         # 1. Capturer intensité AVANT normalisation
#         intensity = patches.mean(dim=-1, keepdim=True)  # (N, P, 1)
        
#         # 2. Normaliser texture
#         mean = patches.mean(dim=-1, keepdim=True)
#         std  = patches.std(dim=-1, keepdim=True).clamp(min=1e-6)
#         patches_std = (patches - mean) / std
        
#         # 3. Concaténer texture + intensité
#         patches_augmented = torch.cat([patches_std, intensity], dim=-1)  # (N, P, D+1)
        
#         return patches_augmented

#     def process_batch(self, imgs):
#         imgs = imgs.to(self.device)
#         all_activated = []
#         all_z = []
        
#         for scale_idx, patch_size in enumerate(self.patch_sizes):
#             # Extraire patches
#             patches = self.extract_patches_batch(imgs, patch_size)
            
#             # Préprocesser AVEC intensité si activé
#             patches_std = self.preprocess_patches(patches, keep_intensity=True)
#             protos = self.prototypes[scale_idx]  # Déjà avec intensité si use_intensity
            
#             _, P, D = patches_std.shape
#             B_scale = protos.shape[0]
            
#             # Calcul distances
#             patches_sq = (patches_std ** 2).sum(dim=-1)
#             protos_sq  = (protos ** 2).sum(dim=-1)
#             dot = torch.einsum("npd,bd->nbp", patches_std, protos)
#             dists_sq = (patches_sq.unsqueeze(1) + 
#                        protos_sq.view(1, B_scale, 1) - 2 * dot).clamp(min=0)
            
#             # Top-K
#             topk_dists, topk_idx = dists_sq.topk(self.K, dim=2, largest=False)
#             sim = torch.exp(-topk_dists.mean(dim=2) / D ** 0.5)
#             activated = (sim >= self.theta_init).bool()
            
#             # Agréger
#             topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)
#             patches_exp  = patches_std.unsqueeze(1).expand(-1, B_scale, -1, -1)
#             z = patches_exp.gather(2, topk_idx_exp).mean(dim=2)
            
#             all_activated.append(activated)
#             all_z.append(z)
        
#         return all_activated, all_z

#     def update_batch_lvq(self, all_activated, all_z, labels, lr=0.05):
#         N = len(labels)
        
#         for scale_idx in range(self.n_scales):
#             activated = all_activated[scale_idx]
#             z = all_z[scale_idx]
            
#             for i in range(N):
#                 lbl = labels[i].item()
#                 act = activated[i]
#                 if not act.any():
#                     continue
                
#                 self.class_counts[scale_idx][act] *= 0.99
#                 self.class_counts[scale_idx][act, lbl] += 1
#                 self.proto_class[scale_idx][act] = \
#                     self.class_counts[scale_idx][act].argmax(dim=1)
                
#                 correct = act & (self.proto_class[scale_idx] == lbl)
#                 incorrect = act & (self.proto_class[scale_idx] != lbl)
                
#                 if correct.any():
#                     self.prototypes[scale_idx][correct] += lr * (
#                         z[i][correct] - self.prototypes[scale_idx][correct]
#                     )
#                 if incorrect.any():
#                     self.prototypes[scale_idx][incorrect] -= lr * (
#                         z[i][incorrect] - self.prototypes[scale_idx][incorrect]
#                     )
            
#             self.prototypes[scale_idx].clamp_(-5.0, 5.0)

#     def get_vote_weights(self, scale_idx):
#         total = self.class_counts[scale_idx].sum(dim=1, keepdim=True).clamp(min=1)
#         freq  = self.class_counts[scale_idx] / total
#         max_freq  = freq.max(dim=1).values
#         mean_freq = freq.mean(dim=1)
#         weights = (max_freq - mean_freq) * 2
#         return weights, freq

#     def reassign_proto_class(self, train_images, train_labels, device, batch_size=2):
#         for scale_idx in range(self.n_scales):
#             self.class_counts[scale_idx].zero_()
        
#         images_t = torch.stack(train_images).to(device)
#         labels_t = torch.tensor(train_labels, device=device, dtype=torch.long)
        
#         for start in range(0, len(images_t), batch_size):
#             end = min(start + batch_size, len(images_t))
#             all_activated, _ = self.process_batch(images_t[start:end])
#             lbls_b = labels_t[start:end]
            
#             for scale_idx in range(self.n_scales):
#                 activated = all_activated[scale_idx]
#                 for i in range(end - start):
#                     self.class_counts[scale_idx][activated[i], lbls_b[i].item()] += 1
        
#         for scale_idx in range(self.n_scales):
#             assigned = self.class_counts[scale_idx].sum(dim=1) > 0
#             n_assigned = assigned.sum().item()
            
#             class_freq = self.class_counts[scale_idx].sum(dim=0).clamp(min=1)
#             counts_norm = self.class_counts[scale_idx] / class_freq.unsqueeze(0)
#             self.proto_class[scale_idx][assigned] = counts_norm[assigned].argmax(dim=1)
#             self.proto_class[scale_idx][~assigned] = -1
            
#             ps = self.patch_sizes[scale_idx]
#             print(f"    [Reassign {ps[0]}×{ps[1]}] {n_assigned}/{self.B_per_scale[scale_idx]} protos")


# # class PopulationBMultiScale:
# #     def __init__(self, num_cells=6400, patch_sizes=[(5,5), (9,9), (13,13)],
# #                  theta_init=0.5, beta=5.0, num_classes=2, K=1, device="cuda"):
# #         self.device      = device
# #         self.B           = num_cells
# #         self.patch_sizes = patch_sizes
# #         self.n_scales    = len(patch_sizes)
# #         self.beta        = beta
# #         self.theta_init  = theta_init
# #         self.num_classes = num_classes
# #         self.K           = K
        
# #         # Répartir prototypes entre échelles
# #         self.B_per_scale = [num_cells // self.n_scales] * self.n_scales
# #         self.B_per_scale[-1] += num_cells - sum(self.B_per_scale)
        
# #         # Créer prototypes par échelle
# #         self.prototypes   = []
# #         self.proto_class  = []
# #         self.class_counts = []
        
# #         for i, (ph, pw) in enumerate(patch_sizes):
# #             D = ph * pw
# #             B_scale = self.B_per_scale[i]
            
# #             self.prototypes.append(
# #                 torch.randn(B_scale, D, device=device) * 0.1
# #             )
# #             self.proto_class.append(
# #                 torch.full((B_scale,), -1, dtype=torch.long, device=device)
# #             )
# #             self.class_counts.append(
# #                 torch.zeros(B_scale, num_classes, device=device)
# #             )
        
# #         print(f"[Multi-scale] {self.n_scales} échelles :")
# #         for i, ps in enumerate(patch_sizes):
# #             print(f"  Échelle {i} : patch {ps[0]}×{ps[1]} → {self.B_per_scale[i]} prototypes")

# #     def extract_patches_batch(self, imgs, patch_size):
# #         patches = F.unfold(imgs.unsqueeze(1), kernel_size=patch_size, stride=1)
# #         return patches.transpose(1, 2)

# #     def preprocess_patches(self, patches):
# #         mean = patches.mean(dim=-1, keepdim=True)
# #         std  = patches.std(dim=-1, keepdim=True).clamp(min=1e-6)
# #         return (patches - mean) / std

# #     def process_batch(self, imgs):
# #         imgs = imgs.to(self.device)
# #         all_activated = []
# #         all_z = []
        
# #         for scale_idx, patch_size in enumerate(self.patch_sizes):
# #             patches = self.extract_patches_batch(imgs, patch_size)
# #             patches_std = self.preprocess_patches(patches)
# #             protos = self.preprocess_patches(
# #                 self.prototypes[scale_idx].unsqueeze(0)
# #             ).squeeze(0)
            
# #             _, P, D = patches_std.shape
# #             B_scale = protos.shape[0]
            
# #             patches_sq = (patches_std ** 2).sum(dim=-1)
# #             protos_sq  = (protos ** 2).sum(dim=-1)
# #             dot = torch.einsum("npd,bd->nbp", patches_std, protos)
# #             dists_sq = (patches_sq.unsqueeze(1) + 
# #                        protos_sq.view(1, B_scale, 1) - 2 * dot).clamp(min=0)
            
# #             topk_dists, topk_idx = dists_sq.topk(self.K, dim=2, largest=False)
# #             sim = torch.exp(-topk_dists.mean(dim=2) / D ** 0.5)
# #             activated = (sim >= self.theta_init).bool()
            
# #             topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)
# #             patches_exp  = patches_std.unsqueeze(1).expand(-1, B_scale, -1, -1)
# #             z = patches_exp.gather(2, topk_idx_exp).mean(dim=2)
            
# #             all_activated.append(activated)
# #             all_z.append(z)
        
# #         return all_activated, all_z

# #     def update_batch_lvq(self, all_activated, all_z, labels, lr=0.05):
# #         N = len(labels)
        
# #         for scale_idx in range(self.n_scales):
# #             activated = all_activated[scale_idx]
# #             z = all_z[scale_idx]
            
# #             for i in range(N):
# #                 lbl = labels[i].item()
# #                 act = activated[i]
# #                 if not act.any():
# #                     continue
                
# #                 self.class_counts[scale_idx][act] *= 0.99
# #                 self.class_counts[scale_idx][act, lbl] += 1
# #                 self.proto_class[scale_idx][act] = \
# #                     self.class_counts[scale_idx][act].argmax(dim=1)
                
# #                 correct = act & (self.proto_class[scale_idx] == lbl)
# #                 incorrect = act & (self.proto_class[scale_idx] != lbl)
                
# #                 if correct.any():
# #                     self.prototypes[scale_idx][correct] += lr * (
# #                         z[i][correct] - self.prototypes[scale_idx][correct]
# #                     )
# #                 if incorrect.any():
# #                     self.prototypes[scale_idx][incorrect] -= lr * (
# #                         z[i][incorrect] - self.prototypes[scale_idx][incorrect]
# #                     )
            
# #             self.prototypes[scale_idx].clamp_(-5.0, 5.0)

# #     def get_vote_weights(self, scale_idx):
# #         total = self.class_counts[scale_idx].sum(dim=1, keepdim=True).clamp(min=1)
# #         freq  = self.class_counts[scale_idx] / total
# #         max_freq  = freq.max(dim=1).values
# #         mean_freq = freq.mean(dim=1)
# #         weights = (max_freq - mean_freq) * 2
# #         return weights, freq

# #     def reassign_proto_class(self, train_images, train_labels, device, batch_size=2):
# #         for scale_idx in range(self.n_scales):
# #             self.class_counts[scale_idx].zero_()
        
# #         images_t = torch.stack(train_images).to(device)
# #         labels_t = torch.tensor(train_labels, device=device, dtype=torch.long)
        
# #         for start in range(0, len(images_t), batch_size):
# #             end = min(start + batch_size, len(images_t))
# #             all_activated, _ = self.process_batch(images_t[start:end])
# #             lbls_b = labels_t[start:end]
            
# #             for scale_idx in range(self.n_scales):
# #                 activated = all_activated[scale_idx]
# #                 for i in range(end - start):
# #                     self.class_counts[scale_idx][activated[i], lbls_b[i].item()] += 1
        
# #         for scale_idx in range(self.n_scales):
# #             assigned = self.class_counts[scale_idx].sum(dim=1) > 0
# #             n_assigned = assigned.sum().item()
            
# #             class_freq = self.class_counts[scale_idx].sum(dim=0).clamp(min=1)
# #             counts_norm = self.class_counts[scale_idx] / class_freq.unsqueeze(0)
# #             self.proto_class[scale_idx][assigned] = counts_norm[assigned].argmax(dim=1)
# #             self.proto_class[scale_idx][~assigned] = -1
            
# #             ps = self.patch_sizes[scale_idx]
# #             print(f"    [Reassign {ps[0]}×{ps[1]}] {n_assigned}/{self.B_per_scale[scale_idx]} prototypes")


# class TrainerMultiScale:
#     def __init__(self, population, num_classes=2, device="cuda"):
#         self.population  = population
#         self.device      = device
#         self.num_classes = num_classes

#     def train_batch(self, images, labels, batch_size=2, lr=0.05):
#         images_t = torch.stack(images).to(self.device)
#         labels_t = torch.tensor(labels, device=self.device, dtype=torch.long)
        
#         for start in range(0, len(images_t), batch_size):
#             end = min(start + batch_size, len(images_t))
#             all_activated, all_z = self.population.process_batch(images_t[start:end])
#             if not any(a.any() for a in all_activated):
#                 continue
#             self.population.update_batch_lvq(
#                 all_activated, all_z, labels_t[start:end], lr
#             )

#     def predict_batch(self, images, batch_size=4):
#         images_t = torch.stack(images).to(self.device)
#         all_preds = []
        
#         for start in range(0, len(images_t), batch_size):
#             end = min(start + batch_size, len(images_t))
#             all_activated, _ = self.population.process_batch(images_t[start:end])
            
#             for i in range(end - start):
#                 total_votes = torch.zeros(self.num_classes, device=self.device)
                
#                 for scale_idx in range(self.population.n_scales):
#                     act_i = all_activated[scale_idx][i]
#                     valid = act_i & (self.population.proto_class[scale_idx] >= 0)
                    
#                     if not valid.any():
#                         continue
                    
#                     weights, freq = self.population.get_vote_weights(scale_idx)
#                     active_freq = freq[valid]
#                     active_weights = weights[valid]
#                     votes = (active_freq * active_weights.unsqueeze(1)).sum(dim=0)
#                     total_votes += votes
                
#                 if total_votes.sum() == 0:
#                     all_preds.append(None)
#                 else:
#                     all_preds.append(total_votes.argmax().item())
        
#         return all_preds
    

#---------------------------------------------

# ============================================================
# model.py — Multi-scale avec gradient descent
# ============================================================

# import torch
# import torch.nn.functional as F


# class PopulationBMultiScale:
#     def __init__(self, num_cells=6400, patch_sizes=[(5,5), (9,9), (13,13)],
#                  theta_init=0.5, beta=5.0, num_classes=2, K=1, 
#                  use_intensity=True, device="cuda"):
#         self.device       = device
#         self.B            = num_cells
#         self.patch_sizes  = patch_sizes
#         self.n_scales     = len(patch_sizes)
#         self.beta         = beta
#         self.theta_init   = theta_init
#         self.num_classes  = num_classes
#         self.K            = K
#         self.use_intensity = use_intensity
        
#         # Répartir prototypes
#         self.B_per_scale = [num_cells // self.n_scales] * self.n_scales
#         self.B_per_scale[-1] += num_cells - sum(self.B_per_scale)
        
#         # Créer prototypes par échelle
#         self.prototypes   = []
#         self.proto_class  = []
#         self.class_counts = []
        
#         for i, (ph, pw) in enumerate(patch_sizes):
#             D_base = ph * pw
#             D = D_base + 1 if use_intensity else D_base
#             B_scale = self.B_per_scale[i]
            
#             self.prototypes.append(
#                 torch.randn(B_scale, D, device=device) * 0.1
#             )
#             self.proto_class.append(
#                 torch.full((B_scale,), -1, dtype=torch.long, device=device)
#             )
#             self.class_counts.append(
#                 torch.zeros(B_scale, num_classes, device=device)
#             )
        
#         print(f"[Multi-scale] {self.n_scales} échelles (intensité: {use_intensity}):")
#         for i, ps in enumerate(patch_sizes):
#             D_feat = (ps[0] * ps[1] + 1) if use_intensity else (ps[0] * ps[1])
#             print(f"  Échelle {i}: {ps[0]}×{ps[1]} → {self.B_per_scale[i]} protos, {D_feat} features")

#     def extract_patches_batch(self, imgs, patch_size):
#         patches = F.unfold(imgs.unsqueeze(1), kernel_size=patch_size, stride=1)
#         return patches.transpose(1, 2)

#     def preprocess_patches(self, patches, keep_intensity=False):
#         if not self.use_intensity or not keep_intensity:
#             mean = patches.mean(dim=-1, keepdim=True)
#             std  = patches.std(dim=-1, keepdim=True).clamp(min=1e-6)
#             return (patches - mean) / std
        
#         intensity = patches.mean(dim=-1, keepdim=True)
#         mean = patches.mean(dim=-1, keepdim=True)
#         std  = patches.std(dim=-1, keepdim=True).clamp(min=1e-6)
#         patches_std = (patches - mean) / std
#         patches_augmented = torch.cat([patches_std, intensity], dim=-1)
        
#         return patches_augmented

#     def process_batch(self, imgs):
#         imgs = imgs.to(self.device)
#         all_activated = []
#         all_z = []
        
#         for scale_idx, patch_size in enumerate(self.patch_sizes):
#             patches = self.extract_patches_batch(imgs, patch_size)
#             patches_std = self.preprocess_patches(patches, keep_intensity=True)
#             protos = self.prototypes[scale_idx]
            
#             _, P, D = patches_std.shape
#             B_scale = protos.shape[0]
            
#             patches_sq = (patches_std ** 2).sum(dim=-1)
#             protos_sq  = (protos ** 2).sum(dim=-1)
#             dot = torch.einsum("npd,bd->nbp", patches_std, protos)
#             dists_sq = (patches_sq.unsqueeze(1) + 
#                        protos_sq.view(1, B_scale, 1) - 2 * dot).clamp(min=0)
            
#             topk_dists, topk_idx = dists_sq.topk(self.K, dim=2, largest=False)
#             sim = torch.exp(-topk_dists.mean(dim=2) / D ** 0.5)
#             activated = (sim >= self.theta_init).bool()
            
#             topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)
#             patches_exp  = patches_std.unsqueeze(1).expand(-1, B_scale, -1, -1)
#             z = patches_exp.gather(2, topk_idx_exp).mean(dim=2)
            
#             all_activated.append(activated)
#             all_z.append(z)
        
#         return all_activated, all_z

#     def update_batch_gradient(self, all_activated, all_z, labels, lr=0.05):
#         """
#         Apprentissage par GRADIENT DESCENT au lieu de LVQ.
#         Loss contrastive : rapprocher prototypes de leur classe, éloigner des autres.
#         """
#         N = len(labels)
        
#         for scale_idx in range(self.n_scales):
#             activated = all_activated[scale_idx]
#             z = all_z[scale_idx]
            
#             # Accumuler gradients
#             proto_grads = torch.zeros_like(self.prototypes[scale_idx])
            
#             for i in range(N):
#                 lbl = labels[i].item()
#                 act = activated[i]
#                 if not act.any():
#                     continue
                
#                 # ✅ Mettre à jour compteurs (système de vote intact)
#                 self.class_counts[scale_idx][act] *= 0.99
#                 self.class_counts[scale_idx][act, lbl] += 1
#                 self.proto_class[scale_idx][act] = \
#                     self.class_counts[scale_idx][act].argmax(dim=1)
                
#                 # ✅ GRADIENT DESCENT : calculer gradients
#                 for proto_idx in torch.where(act)[0]:
#                     proto_class = self.proto_class[scale_idx][proto_idx].item()
#                     if proto_class < 0:
#                         continue
                    
#                     proto = self.prototypes[scale_idx][proto_idx]
#                     patch = z[i][proto_idx]
                    
#                     # Distance au carré
#                     diff = proto - patch
#                     distance_sq = (diff ** 2).sum()
                    
#                     if proto_class == lbl:  # Même classe
#                         # Loss = distance² → rapprocher
#                         grad = 2 * diff  # Gradient de ||proto - patch||²
#                     else:  # Classe différente
#                         # Loss = max(0, margin - distance²) → éloigner si trop proche
#                         margin = 1.0
#                         if distance_sq < margin:
#                             grad = -2 * diff  # Éloigner
#                         else:
#                             grad = torch.zeros_like(diff)  # Déjà assez loin
                    
#                     proto_grads[proto_idx] += grad
            
#             # ✅ Appliquer gradients accumulés
#             self.prototypes[scale_idx] -= lr * proto_grads
#             self.prototypes[scale_idx].clamp_(-5.0, 5.0)

#     def get_vote_weights(self, scale_idx):
#         total = self.class_counts[scale_idx].sum(dim=1, keepdim=True).clamp(min=1)
#         freq  = self.class_counts[scale_idx] / total
#         max_freq  = freq.max(dim=1).values
#         mean_freq = freq.mean(dim=1)
#         weights = (max_freq - mean_freq) * 2
#         return weights, freq

#     def reassign_proto_class(self, train_images, train_labels, device, batch_size=2):
#         for scale_idx in range(self.n_scales):
#             self.class_counts[scale_idx].zero_()
        
#         images_t = torch.stack(train_images).to(device)
#         labels_t = torch.tensor(train_labels, device=device, dtype=torch.long)
        
#         for start in range(0, len(images_t), batch_size):
#             end = min(start + batch_size, len(images_t))
#             all_activated, _ = self.process_batch(images_t[start:end])
#             lbls_b = labels_t[start:end]
            
#             for scale_idx in range(self.n_scales):
#                 activated = all_activated[scale_idx]
#                 for i in range(end - start):
#                     self.class_counts[scale_idx][activated[i], lbls_b[i].item()] += 1
        
#         for scale_idx in range(self.n_scales):
#             assigned = self.class_counts[scale_idx].sum(dim=1) > 0
#             n_assigned = assigned.sum().item()
            
#             class_freq = self.class_counts[scale_idx].sum(dim=0).clamp(min=1)
#             counts_norm = self.class_counts[scale_idx] / class_freq.unsqueeze(0)
#             self.proto_class[scale_idx][assigned] = counts_norm[assigned].argmax(dim=1)
#             self.proto_class[scale_idx][~assigned] = -1
            
#             ps = self.patch_sizes[scale_idx]
#             print(f"    [Reassign {ps[0]}×{ps[1]}] {n_assigned}/{self.B_per_scale[scale_idx]} protos")


# class TrainerMultiScale:
#     def __init__(self, population, num_classes=2, device="cuda"):
#         self.population  = population
#         self.device      = device
#         self.num_classes = num_classes

#     def train_batch(self, images, labels, batch_size=2, lr=0.05):
#         images_t = torch.stack(images).to(self.device)
#         labels_t = torch.tensor(labels, device=self.device, dtype=torch.long)
        
#         for start in range(0, len(images_t), batch_size):
#             end = min(start + batch_size, len(images_t))
#             all_activated, all_z = self.population.process_batch(images_t[start:end])
#             if not any(a.any() for a in all_activated):
#                 continue
            
#             # ✅ CHANGEMENT : Gradient descent au lieu de LVQ
#             self.population.update_batch_gradient(
#                 all_activated, all_z, labels_t[start:end], lr
#             )

#     def predict_batch(self, images, batch_size=4):
#         images_t = torch.stack(images).to(self.device)
#         all_preds = []
        
#         for start in range(0, len(images_t), batch_size):
#             end = min(start + batch_size, len(images_t))
#             all_activated, _ = self.population.process_batch(images_t[start:end])
            
#             for i in range(end - start):
#                 total_votes = torch.zeros(self.num_classes, device=self.device)
                
#                 for scale_idx in range(self.population.n_scales):
#                     act_i = all_activated[scale_idx][i]
#                     valid = act_i & (self.population.proto_class[scale_idx] >= 0)
                    
#                     if not valid.any():
#                         continue
                    
#                     weights, freq = self.population.get_vote_weights(scale_idx)
#                     active_freq = freq[valid]
#                     active_weights = weights[valid]
#                     votes = (active_freq * active_weights.unsqueeze(1)).sum(dim=0)
#                     total_votes += votes
                
#                 if total_votes.sum() == 0:
#                     all_preds.append(None)
#                 else:
#                     all_preds.append(total_votes.argmax().item())
        
#         return all_preds



# ============================================================
# model.py — Gradient descent + Entropie + Hard assignment
# ============================================================

import torch
import torch.nn.functional as F


class PopulationBMultiScale:

    def __init__(self, num_cells=6400, patch_sizes=[(5,5), (9,9), (13,13)],
                 theta_init=0.5, beta=5.0, num_classes=2, K=1, 
                 use_intensity=True, device="cuda"):
        self.device       = device
        self.B            = num_cells
        self.patch_sizes  = patch_sizes
        self.n_scales     = len(patch_sizes)
        self.beta         = beta
        self.theta_init   = theta_init
        self.num_classes  = num_classes
        self.K            = K
        self.use_intensity = use_intensity
        
        self.B_per_scale = [num_cells // self.n_scales] * self.n_scales
        self.B_per_scale[-1] += num_cells - sum(self.B_per_scale)
        
        self.prototypes   = []
        self.proto_class  = []
        self.class_counts = []
        
        for i, (ph, pw) in enumerate(patch_sizes):
            D_base = ph * pw
            D = D_base + 1 if use_intensity else D_base
            B_scale = self.B_per_scale[i]
            
            self.prototypes.append(
                torch.randn(B_scale, D, device=device) * 0.1
            )
            self.proto_class.append(
                torch.full((B_scale,), -1, dtype=torch.long, device=device)
            )
            self.class_counts.append(
                torch.zeros(B_scale, num_classes, device=device)
            )
        
        print(f"[Multi-scale GPU] {self.n_scales} échelles (intensité: {use_intensity}):")
        for i, ps in enumerate(patch_sizes):
            D_feat = (ps[0] * ps[1] + 1) if use_intensity else (ps[0] * ps[1])
            print(f"  Échelle {i}: {ps[0]}×{ps[1]} → {self.B_per_scale[i]} protos, {D_feat} features")

    def extract_patches_batch(self, imgs, patch_size):
        patches = F.unfold(imgs.unsqueeze(1), kernel_size=patch_size, stride=1)
        return patches.transpose(1, 2)

    def preprocess_patches(self, patches, keep_intensity=False):
        if not self.use_intensity or not keep_intensity:
            mean = patches.mean(dim=-1, keepdim=True)
            std  = patches.std(dim=-1, keepdim=True).clamp(min=1e-6)
            return (patches - mean) / std
        
        intensity = patches.mean(dim=-1, keepdim=True)
        mean = patches.mean(dim=-1, keepdim=True)
        std  = patches.std(dim=-1, keepdim=True).clamp(min=1e-6)
        patches_std = (patches - mean) / std
        patches_augmented = torch.cat([patches_std, intensity], dim=-1)
        
        return patches_augmented

    def process_batch(self, imgs):
        imgs = imgs.to(self.device)
        all_activated = []
        all_z = []
        
        for scale_idx, patch_size in enumerate(self.patch_sizes):
            patches = self.extract_patches_batch(imgs, patch_size)
            patches_std = self.preprocess_patches(patches, keep_intensity=True)
            protos = self.prototypes[scale_idx]
            
            _, P, D = patches_std.shape
            B_scale = protos.shape[0]
            
            patches_sq = (patches_std ** 2).sum(dim=-1)
            protos_sq  = (protos ** 2).sum(dim=-1)
            dot = torch.einsum("npd,bd->nbp", patches_std, protos)
            dists_sq = (patches_sq.unsqueeze(1) + 
                       protos_sq.view(1, B_scale, 1) - 2 * dot).clamp(min=0)
            
            topk_dists, topk_idx = dists_sq.topk(self.K, dim=2, largest=False)
            sim = torch.exp(-topk_dists.mean(dim=2) / D ** 0.5)
            activated = (sim >= self.theta_init).bool()
            
            topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)
            patches_exp  = patches_std.unsqueeze(1).expand(-1, B_scale, -1, -1)
            z = patches_exp.gather(2, topk_idx_exp).mean(dim=2)
            
            all_activated.append(activated)
            all_z.append(z)
        
        return all_activated, all_z



    # def update_batch_gradient_vectorized(self, all_activated, all_z, labels, lr=0.05, 
    #                                       entropy_weight=0.1):
    #     """
    #     Gradient descent avec PÉNALITÉ ENTROPIE pour forcer spécialisation.
        
    #     Args:
    #         entropy_weight: Coefficient pénalité entropie (0.1 par défaut)
    #     """
    #     N = len(labels)
        
    #     for scale_idx in range(self.n_scales):
    #         activated = all_activated[scale_idx]
    #         z = all_z[scale_idx]
            
    #         # Mettre à jour compteurs
    #         for i in range(N):
    #             lbl = labels[i].item()
    #             act = activated[i]
    #             if not act.any():
    #                 continue
                
    #             self.class_counts[scale_idx][act] *= 0.99
    #             self.class_counts[scale_idx][act, lbl] += 1
            
    #         # Réassigner classes
    #         self.proto_class[scale_idx] = self.class_counts[scale_idx].argmax(dim=1)
    #         self.proto_class[scale_idx][self.class_counts[scale_idx].sum(dim=1) == 0] = -1
            
    #         # ✅ GRADIENT CONTRASTIVE
    #         proto_grads = torch.zeros_like(self.prototypes[scale_idx])
    #         margin = 1.0
            
    #         for i in range(N):
    #             lbl = labels[i].item()
    #             act_i = activated[i]
                
    #             if not act_i.any():
    #                 continue
                
    #             protos_active = self.prototypes[scale_idx][act_i]
    #             z_active = z[i][act_i]
    #             classes_active = self.proto_class[scale_idx][act_i]
                
    #             diff = protos_active - z_active
    #             dist_sq = (diff ** 2).sum(dim=1)
                
    #             same_class = (classes_active == lbl) & (classes_active >= 0)
    #             diff_class = (classes_active != lbl) & (classes_active >= 0)
                
    #             grads_active = torch.zeros_like(diff)
    #             grads_active[same_class] = 2 * diff[same_class]
                
    #             need_repel = diff_class & (dist_sq < margin)
    #             grads_active[need_repel] = -2 * diff[need_repel]
                
    #             active_indices = torch.where(act_i)[0]
    #             proto_grads.index_add_(0, active_indices, grads_active)
            
    #         # ✅ PÉNALITÉ ENTROPIE (forcer spécialisation)
    #         _, freq = self.get_vote_weights(scale_idx)
            
    #         # Calculer entropie : -sum(p * log(p))
    #         # Entropie faible = spécialisé [0.95, 0.05]
    #         # Entropie élevée = ambigu [0.5, 0.5]
    #         freq_safe = freq.clamp(min=1e-8)  # Éviter log(0)
    #         entropy = -(freq_safe * torch.log(freq_safe)).sum(dim=1)  # (B,)
            
    #         # Gradient entropie pousse vers réduction entropie = spécialisation
    #         # Pour chaque prototype, on veut augmenter la freq de sa classe dominante
    #         max_class = freq.argmax(dim=1)  # (B,)
            
    #         # Gradient simplifié : pousser vers classe dominante
    #         entropy_grad = torch.zeros_like(self.prototypes[scale_idx])
    #         for proto_idx in range(self.prototypes[scale_idx].shape[0]):
    #             if self.proto_class[scale_idx][proto_idx] >= 0:
    #                 # Pénalité proportionnelle à l'entropie
    #                 entropy_grad[proto_idx] = entropy[proto_idx] * proto_grads[proto_idx].sign()
            
    #         # ✅ Combiner gradients
    #         total_grads = proto_grads + entropy_weight * entropy_grad
            
    #         # Mise à jour
    #         self.prototypes[scale_idx] -= lr * total_grads
    #         self.prototypes[scale_idx].clamp_(-5.0, 5.0)


    def update_batch_gradient_vectorized(self, all_activated, all_z, labels, lr=0.05, 
                                      entropy_weight=0.1):
        """
        Gradient descent 100% VECTORISÉ GPU avec pénalité entropie.
        """
        N = len(labels)
        
        for scale_idx in range(self.n_scales):
            activated = all_activated[scale_idx]  # (N, B)
            z = all_z[scale_idx]                   # (N, B, D)
            
            # ✅ Mettre à jour compteurs (vectorisé)
            for i in range(N):
                lbl = labels[i].item()
                act = activated[i]
                if not act.any():
                    continue
                
                self.class_counts[scale_idx][act] *= 0.99
                self.class_counts[scale_idx][act, lbl] += 1
            
            # Réassigner classes
            self.proto_class[scale_idx] = self.class_counts[scale_idx].argmax(dim=1)
            self.proto_class[scale_idx][self.class_counts[scale_idx].sum(dim=1) == 0] = -1
            
            # ✅ CALCUL GRADIENTS CONTRASTIFS VECTORISÉ
            proto_grads = torch.zeros_like(self.prototypes[scale_idx])  # (B, D)
            margin = 1.0
            
            for i in range(N):
                lbl = labels[i].item()
                act_i = activated[i]  # (B,) booléen
                
                if not act_i.any():
                    continue
                
                # Prototypes activés
                protos_active = self.prototypes[scale_idx][act_i]  # (n_active, D)
                z_active = z[i][act_i]                              # (n_active, D)
                classes_active = self.proto_class[scale_idx][act_i] # (n_active,)
                
                # ✅ Différence vectorisée
                diff = protos_active - z_active  # (n_active, D)
                dist_sq = (diff ** 2).sum(dim=1)  # (n_active,)
                
                # ✅ Masques vectorisés
                same_class = (classes_active == lbl) & (classes_active >= 0)  # (n_active,)
                diff_class = (classes_active != lbl) & (classes_active >= 0)  # (n_active,)
                
                # ✅ Gradients vectorisés
                grads_active = torch.zeros_like(diff)  # (n_active, D)
                
                # Même classe : grad = 2 * diff (rapprocher)
                grads_active[same_class] = 2 * diff[same_class]
                
                # Classe différente : grad = -2 * diff si distance < margin (éloigner)
                need_repel = diff_class & (dist_sq < margin)
                grads_active[need_repel] = -2 * diff[need_repel]
                
                # ✅ Accumuler dans proto_grads
                active_indices = torch.where(act_i)[0]
                proto_grads.index_add_(0, active_indices, grads_active)
            
            # ✅ PÉNALITÉ ENTROPIE VECTORISÉE (CORRIGÉ - RAPIDE)
            _, freq = self.get_vote_weights(scale_idx)
            
            # Calculer entropie : -sum(p * log(p))
            freq_safe = freq.clamp(min=1e-8)
            entropy = -(freq_safe * torch.log(freq_safe)).sum(dim=1)  # (B,)
            
            # ✅ Gradient entropie vectorisé (pas de boucle Python)
            assigned_mask = self.proto_class[scale_idx] >= 0  # (B,) booléen
            
            entropy_grad = torch.zeros_like(self.prototypes[scale_idx])
            entropy_grad[assigned_mask] = (
                entropy[assigned_mask].unsqueeze(1) * proto_grads[assigned_mask].sign()
            )
            
            # ✅ Combiner gradients
            total_grads = proto_grads + entropy_weight * entropy_grad
            
            # ✅ Mise à jour GPU vectorisée
            self.prototypes[scale_idx] -= lr * total_grads
            self.prototypes[scale_idx].clamp_(-5.0, 5.0)

    def hard_reset_ambiguous(self, threshold=0.70):
        """
        RESET prototypes ambigus (spécialisation < threshold).
        À appeler tous les 2 epochs.
        """
        n_reset_total = 0
        
        for scale_idx in range(self.n_scales):
            _, freq = self.get_vote_weights(scale_idx)
            specialization = freq.max(dim=1).values  # Spécialisation par proto
            
            # Identifier prototypes ambigus
            ambiguous = specialization < threshold
            n_ambiguous = ambiguous.sum().item()
            
            if n_ambiguous > 0:
                # RESET : réinitialiser aléatoirement
                self.prototypes[scale_idx][ambiguous] = torch.randn_like(
                    self.prototypes[scale_idx][ambiguous]
                ) * 0.1
                
                # Réinitialiser compteurs
                self.class_counts[scale_idx][ambiguous] = 0
                self.proto_class[scale_idx][ambiguous] = -1
                
                n_reset_total += n_ambiguous
            
            ps = self.patch_sizes[scale_idx]
            print(f"    [Hard Reset {ps[0]}×{ps[1]}] {n_ambiguous}/{self.B_per_scale[scale_idx]} protos réinitialisés")
        
        if n_reset_total > 0:
            print(f"  ✅ Total reset : {n_reset_total} prototypes ambigus")

    def get_vote_weights(self, scale_idx):
        total = self.class_counts[scale_idx].sum(dim=1, keepdim=True).clamp(min=1)
        freq  = self.class_counts[scale_idx] / total
        max_freq  = freq.max(dim=1).values
        mean_freq = freq.mean(dim=1)
        weights = (max_freq - mean_freq) * 2
        return weights, freq

    def reassign_proto_class(self, train_images, train_labels, device, batch_size=2):
        for scale_idx in range(self.n_scales):
            self.class_counts[scale_idx].zero_()
        
        images_t = torch.stack(train_images).to(device)
        labels_t = torch.tensor(train_labels, device=device, dtype=torch.long)
        
        for start in range(0, len(images_t), batch_size):
            end = min(start + batch_size, len(images_t))
            all_activated, _ = self.process_batch(images_t[start:end])
            lbls_b = labels_t[start:end]
            
            for scale_idx in range(self.n_scales):
                activated = all_activated[scale_idx]
                for i in range(end - start):
                    self.class_counts[scale_idx][activated[i], lbls_b[i].item()] += 1
        
        for scale_idx in range(self.n_scales):
            assigned = self.class_counts[scale_idx].sum(dim=1) > 0
            n_assigned = assigned.sum().item()
            
            class_freq = self.class_counts[scale_idx].sum(dim=0).clamp(min=1)
            counts_norm = self.class_counts[scale_idx] / class_freq.unsqueeze(0)
            self.proto_class[scale_idx][assigned] = counts_norm[assigned].argmax(dim=1)
            self.proto_class[scale_idx][~assigned] = -1
            
            ps = self.patch_sizes[scale_idx]
            print(f"    [Reassign {ps[0]}×{ps[1]}] {n_assigned}/{self.B_per_scale[scale_idx]} protos")


class TrainerMultiScale:
    def __init__(self, population, num_classes=2, device="cuda"):
        self.population  = population
        self.device      = device
        self.num_classes = num_classes

    def train_batch(self, images, labels, batch_size=2, lr=0.05, entropy_weight=0.1):
        images_t = torch.stack(images).to(self.device)
        labels_t = torch.tensor(labels, device=self.device, dtype=torch.long)
        
        for start in range(0, len(images_t), batch_size):
            end = min(start + batch_size, len(images_t))
            all_activated, all_z = self.population.process_batch(images_t[start:end])
            if not any(a.any() for a in all_activated):
                continue
            
            # ✅ Gradient descent avec pénalité entropie
            self.population.update_batch_gradient_vectorized(
                all_activated, all_z, labels_t[start:end], lr, entropy_weight
            )

    def predict_batch(self, images, batch_size=4):
        images_t = torch.stack(images).to(self.device)
        all_preds = []
        
        for start in range(0, len(images_t), batch_size):
            end = min(start + batch_size, len(images_t))
            all_activated, _ = self.population.process_batch(images_t[start:end])
            
            for i in range(end - start):
                total_votes = torch.zeros(self.num_classes, device=self.device)
                
                for scale_idx in range(self.population.n_scales):
                    act_i = all_activated[scale_idx][i]
                    
                    proto_class = self.population.proto_class[scale_idx]
                    _, freq = self.population.get_vote_weights(scale_idx)
                    
                    specialization = torch.zeros(len(proto_class), device=self.device)
                    for c in range(self.num_classes):
                        mask = proto_class == c
                        specialization[mask] = freq[mask, c]
                    
                    highly_specialized = specialization > 0.75
                    valid = act_i & (proto_class >= 0) & highly_specialized
                    
                    if not valid.any():
                        continue
                    
                    active_freq = freq[valid]
                    active_spec = specialization[valid]
                    
                    votes = (active_freq * active_spec.unsqueeze(1)).sum(dim=0)
                    total_votes += votes
                
                if total_votes.sum() == 0:
                    all_preds.append(None)
                else:
                    all_preds.append(total_votes.argmax().item())
        
        return all_preds