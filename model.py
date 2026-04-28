import torch
import torch.nn.functional as F


class PopulationBFastExact:
    def __init__(self, num_cells=1600, patch_size=(5, 5),
                 theta_init=0.5, beta=5.0, num_classes=2,
                 K=1, device="cuda"):
        self.device      = device
        self.B           = num_cells
        self.patch_size  = patch_size
        self.D           = patch_size[0] * patch_size[1]
        self.beta        = beta
        self.theta_init  = theta_init
        self.num_classes = num_classes
        self.K           = K

        self.prototypes   = torch.randn(self.B, self.D, device=device) * 0.1
        self.proto_class  = torch.full((self.B,), -1, dtype=torch.long, device=device)

        # ✅ Compteur d'activations par classe par prototype
        # class_counts[i, c] = combien de fois le prototype i
        # s'est activé pour la classe c
        self.class_counts = torch.zeros(self.B, num_classes, device=device)

    def extract_patches_batch(self, imgs):
        patches = F.unfold(imgs.unsqueeze(1), kernel_size=self.patch_size, stride=1)
        return patches.transpose(1, 2)  # (N, P, D)

    def preprocess_patches(self, patches):
        mean = patches.mean(dim=-1, keepdim=True)
        std  = patches.std(dim=-1, keepdim=True).clamp(min=1e-6)
        return (patches - mean) / std


    def process_batch(self, imgs):
        imgs        = imgs.to(self.device)
        patches     = self.extract_patches_batch(imgs)
        patches_std = self.preprocess_patches(patches)
        protos      = self.preprocess_patches(self.prototypes.unsqueeze(0)).squeeze(0)

        N, P, D = patches_std.shape
        B       = protos.shape[0]

        patches_sq = (patches_std ** 2).sum(dim=-1)
        protos_sq  = (protos ** 2).sum(dim=-1)
        dot        = torch.einsum("npd,bd->nbp", patches_std, protos)
        dists_sq   = (patches_sq.unsqueeze(1) + protos_sq.view(1, B, 1) - 2 * dot).clamp(min=0)

        K = self.K
        topk_dists, topk_idx = dists_sq.topk(K, dim=2, largest=False)
        sim       = torch.exp(-topk_dists.mean(dim=2) / self.D ** 0.5)
        activated = (sim >= self.theta_init).bool()

        topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        patches_exp  = patches_std.unsqueeze(1).expand(-1, B, -1, -1)
        z = patches_exp.gather(2, topk_idx_exp).mean(dim=2)

        return activated, z


    # def process_batch(self, imgs):
    #     imgs    = imgs.to(self.device)
    #     patches = self.extract_patches_batch(imgs)

    #     # ── Filtrer les patches uniformes ──────────────────────
    #     raw_std   = patches.std(dim=-1)                          # (N, P)
    #     threshold = torch.quantile(raw_std, 0.10, dim=1,
    #                             keepdim=True)                 # (N, 1)
    #     valid_mask = (raw_std > threshold)                       # (N, P)
    #     # ────────────────────────────────────────────────────────

    #     patches_std = self.preprocess_patches(patches)
    #     protos      = self.preprocess_patches(
    #                     self.prototypes.unsqueeze(0)).squeeze(0)

    #     N, P, D = patches_std.shape
    #     B       = protos.shape[0]

    #     patches_sq = (patches_std ** 2).sum(dim=-1)
    #     protos_sq  = (protos ** 2).sum(dim=-1)
    #     dot        = torch.einsum("npd,bd->nbp", patches_std, protos)
    #     dists_sq   = (patches_sq.unsqueeze(1) +
    #                 protos_sq.view(1, B, 1) - 2 * dot).clamp(min=0)

    #     # ── Patches uniformes → distance infinie ───────────────
    #     dists_sq = dists_sq.masked_fill(
    #         ~valid_mask.unsqueeze(1),
    #         float('inf')
    #     )
    #     # ────────────────────────────────────────────────────────

    #     K = self.K
    #     topk_dists, topk_idx = dists_sq.topk(K, dim=2, largest=False)
    #     sim       = torch.exp(-topk_dists.mean(dim=2) / self.D ** 0.5)
    #     activated = (sim >= self.theta_init).bool()

    #     topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)
    #     patches_exp  = patches_std.unsqueeze(1).expand(-1, B, -1, -1)
    #     z = patches_exp.gather(2, topk_idx_exp).mean(dim=2)

    #     return activated, z



    def update_batch_lvq(self, activated, z, labels, lr=0.05):
        """
        LVQ + mise à jour des compteurs d'exclusivité.
        """
        N, B, D = z.shape

        for i in range(N):
            lbl = labels[i].item()
            act = activated[i]
            if not act.any():
                continue

            # ✅ Incrémenter compteur avec decay
            self.class_counts[act] *= 0.99
            self.class_counts[act, lbl] += 1

            # ✅ Assigner classe majoritaire
            self.proto_class[act] = self.class_counts[act].argmax(dim=1)

            # LVQ update
            correct   = act & (self.proto_class == lbl)
            incorrect = act & (self.proto_class != lbl)

            if correct.any():
                self.prototypes[correct] += lr * (
                    z[i][correct] - self.prototypes[correct])
            if incorrect.any():
                self.prototypes[incorrect] -= lr * (
                    z[i][incorrect] - self.prototypes[incorrect])

        self.prototypes.clamp_(-5.0, 5.0)

    def get_vote_weights(self):
        """
        Calcule le poids de vote de chaque prototype.
        Poids = exclusivité = à quel point il est exclusif à sa classe.

        Prototype vu 950x Cancer, 50x Normal → exclusivité = 0.95
        Prototype vu 500x Cancer, 500x Normal → exclusivité = 0.0 (commun)
        """
        total = self.class_counts.sum(dim=1, keepdim=True).clamp(min=1)
        freq  = self.class_counts / total  # (B, C) — fréquence normalisée

        # Exclusivité = écart entre la classe dominante et les autres
        # Si freq = [0.95, 0.05] → exclusivité = 0.95 - 0.05 = 0.90
        # Si freq = [0.50, 0.50] → exclusivité = 0.50 - 0.50 = 0.00
        max_freq  = freq.max(dim=1).values          # (B,)
        mean_freq = freq.mean(dim=1)                # (B,)
        weights   = (max_freq - mean_freq) * 2      # (B,) dans [0, 1]

        return weights, freq  # weights = poids, freq = distribution

    def reassign_proto_class(self, train_images, train_labels, device, batch_size=32):
        self.class_counts.zero_()
        images_t = torch.stack(train_images).to(device)
        labels_t = torch.tensor(train_labels, device=device, dtype=torch.long)

        for start in range(0, len(images_t), batch_size):
            end = min(start + batch_size, len(images_t))
            activated, _ = self.process_batch(images_t[start:end])
            lbls_b = labels_t[start:end]
            for i in range(end - start):
                self.class_counts[activated[i], lbls_b[i].item()] += 1

        assigned = self.class_counts.sum(dim=1) > 0
        class_freq        = self.class_counts.sum(dim=0).clamp(min=1)
        counts_normalized = self.class_counts / class_freq.unsqueeze(0)
        self.proto_class[assigned]  = counts_normalized[assigned].argmax(dim=1)
        self.proto_class[~assigned] = -1


class TrainerFastExact:
    def __init__(self, population, num_classes=2, device="cuda"):
        self.population  = population
        self.device      = device
        self.num_classes = num_classes

    def train_batch(self, images, labels, batch_size=16, lr=0.05):
        images_t = torch.stack(images).to(self.device)
        labels_t = torch.tensor(labels, device=self.device, dtype=torch.long)
        for start in range(0, len(images_t), batch_size):
            end = min(start + batch_size, len(images_t))
            activated, z = self.population.process_batch(images_t[start:end])
            if not activated.any():
                continue
            self.population.update_batch_lvq(
                activated, z, labels_t[start:end], lr)

    def predict_batch(self, images, batch_size=32):
        """
        Prédiction par vote pondéré.
        Chaque prototype actif vote pour sa classe
        avec un poids proportionnel à son exclusivité.
        """
        images_t  = torch.stack(images).to(self.device)
        all_preds = []

        # Calculer les poids de vote une fois pour tout le batch
        weights, freq = self.population.get_vote_weights()  # (B,), (B, C)

        for start in range(0, len(images_t), batch_size):
            end = min(start + batch_size, len(images_t))
            activated, _ = self.population.process_batch(images_t[start:end])

            for i in range(end - start):
                act_i = activated[i]  # (B,) booléen

                # Filtrer les prototypes assignés
                valid = act_i & (self.population.proto_class >= 0)

                if not valid.any():
                    all_preds.append(None)
                    continue

                # ✅ Vote pondéré par exclusivité
                # freq[valid] = distribution de classe de chaque proto actif
                # weights[valid] = exclusivité de chaque proto actif
                active_freq    = freq[valid]        # (n_active, C)
                active_weights = weights[valid]     # (n_active,)

                # Vote = somme pondérée des distributions
                # Proto exclusif Cancer (freq=[0.95,0.05], weight=0.9)
                #   → contribue fortement au vote Cancer
                # Proto commun (freq=[0.5,0.5], weight=0.0)
                #   → ne contribue pas
                votes = (active_freq * active_weights.unsqueeze(1)).sum(dim=0)  # (C,)

                if votes.sum() == 0:
                    # Fallback : vote majoritaire simple
                    active_classes = self.population.proto_class[valid]
                    all_preds.append(torch.mode(active_classes).values.item())
                else:
                    all_preds.append(votes.argmax().item())

        return all_preds