# ============================================================
# model.py — Version 256×256 avec normalisation
# ============================================================

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
        self.class_counts = torch.zeros(self.B, num_classes, device=device)

    def extract_patches_batch(self, imgs):
        patches = F.unfold(imgs.unsqueeze(1), kernel_size=self.patch_size, stride=1)
        return patches.transpose(1, 2)

    def preprocess_patches(self, patches):
        """Normalisation z-score — ACTIVE."""
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

    def update_batch_lvq(self, activated, z, labels, lr=0.05):
        N, B, D = z.shape
        for i in range(N):
            lbl = labels[i].item()
            act = activated[i]
            if not act.any():
                continue

            self.class_counts[act] *= 0.99
            self.class_counts[act, lbl] += 1
            self.proto_class[act] = self.class_counts[act].argmax(dim=1)

            correct   = act & (self.proto_class == lbl)
            incorrect = act & (self.proto_class != lbl)

            if correct.any():
                self.prototypes[correct] += lr * (z[i][correct] - self.prototypes[correct])
            if incorrect.any():
                self.prototypes[incorrect] -= lr * (z[i][incorrect] - self.prototypes[incorrect])

        self.prototypes.clamp_(-5.0, 5.0)

    def get_vote_weights(self):
        total = self.class_counts.sum(dim=1, keepdim=True).clamp(min=1)
        freq  = self.class_counts / total
        max_freq  = freq.max(dim=1).values
        mean_freq = freq.mean(dim=1)
        weights   = (max_freq - mean_freq) * 2
        return weights, freq

    def reassign_proto_class(self, train_images, train_labels, device, batch_size=8):
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
        n_assigned = assigned.sum().item()
        print(f"    [Reassign] {n_assigned}/{self.B} prototypes assignés")
        
        class_freq        = self.class_counts.sum(dim=0).clamp(min=1)
        counts_normalized = self.class_counts / class_freq.unsqueeze(0)
        self.proto_class[assigned]  = counts_normalized[assigned].argmax(dim=1)
        self.proto_class[~assigned] = -1


class TrainerFastExact:
    def __init__(self, population, num_classes=2, device="cuda"):
        self.population  = population
        self.device      = device
        self.num_classes = num_classes

    def train_batch(self, images, labels, batch_size=4, lr=0.05):
        images_t = torch.stack(images).to(self.device)
        labels_t = torch.tensor(labels, device=self.device, dtype=torch.long)
        for start in range(0, len(images_t), batch_size):
            end = min(start + batch_size, len(images_t))
            activated, z = self.population.process_batch(images_t[start:end])
            if not activated.any():
                continue
            self.population.update_batch_lvq(activated, z, labels_t[start:end], lr)

    def predict_batch(self, images, batch_size=8):
        images_t  = torch.stack(images).to(self.device)
        all_preds = []
        weights, freq = self.population.get_vote_weights()

        for start in range(0, len(images_t), batch_size):
            end = min(start + batch_size, len(images_t))
            activated, _ = self.population.process_batch(images_t[start:end])

            for i in range(end - start):
                act_i = activated[i]
                valid = act_i & (self.population.proto_class >= 0)

                if not valid.any():
                    all_preds.append(None)
                    continue

                active_freq    = freq[valid]
                active_weights = weights[valid]
                votes = (active_freq * active_weights.unsqueeze(1)).sum(dim=0)

                if votes.sum() == 0:
                    active_classes = self.population.proto_class[valid]
                    all_preds.append(torch.mode(active_classes).values.item())
                else:
                    all_preds.append(votes.argmax().item())

        return all_preds