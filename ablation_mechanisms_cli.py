# ablation_mechanisms_cli.py
import sys, argparse, importlib, torch, gc, os, json, shutil, time
import torch.nn.functional as F

sys.path.insert(0, '/content/population-CBT-learning')

for mod_name in list(sys.modules.keys()):
    if mod_name in ['data', 'run', 'model', 'train', 'save_load']:
        del sys.modules[mod_name]
importlib.invalidate_caches()

from data      import load_ddsm
from save_load import save_model
from run       import set_seed, TRAIN_DIR, VAL_DIR, DEVICE, NUM_CLASSES
from model     import PopulationBMultiScale, TrainerMultiScale

CACHE_CLEAN = "/content/drive/MyDrive/MiniDDSM/miniddsm_val_clean_256.pt"
DRIVE_PATH  = "/content/drive/MyDrive/ablation_results/mechanisms/"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--patch",        type=str,   default="18")
    p.add_argument("--theta",        type=float, default=0.2)
    p.add_argument("--lr",           type=float, default=0.001)
    p.add_argument("--num_cells",    type=int,   default=2133)
    p.add_argument("--K",            type=int,   default=1)
    # ── Flags ablation ──────────────────────────────────────
    p.add_argument("--no_decay",     action="store_true", 
                   help="Désactiver decay 0.99 → γ=1.0")
    p.add_argument("--uniform_vote", action="store_true",
                   help="Vote uniforme au lieu d'exclusivity-weighted")
    p.add_argument("--no_norm",      action="store_true",
                   help="Sans normalisation fréquence dans reassign")
    p.add_argument("--name",         type=str,   default=None)
    if "ipykernel" in sys.modules:
        return p.parse_args(args=[])
    return p.parse_args()


def run(args):
    patch_list  = [int(x) for x in args.patch.split(",")]
    patch_sizes = [(p, p) for p in patch_list]

    # Nom automatique selon les flags
    if args.name:
        run_name = args.name
    else:
        flags = []
        if args.no_decay:     flags.append("nodecay")
        if args.uniform_vote: flags.append("unifvote")
        if args.no_norm:      flags.append("nonorm")
        flag_str = "_".join(flags) if flags else "baseline"
        run_name = f"ablation_{flag_str}_seed{args.seed}"

    print(f"\n{'='*70}\nRUN : {run_name}\n{'='*70}\n")
    os.makedirs("figs", exist_ok=True)
    os.makedirs(DRIVE_PATH, exist_ok=True)
    torch.cuda.empty_cache(); gc.collect()

    # ── Données ──────────────────────────────────────────────
    set_seed(args.seed)
    train_images, train_labels, _, _ = load_ddsm(
        TRAIN_DIR, VAL_DIR, img_size=256, 
        use_mask=True, crop_roi=False
    )
    data_clean = torch.load(CACHE_CLEAN)
    val_images = data_clean["val_images"]
    val_labels = data_clean["val_labels"]

    print(f"Train : {len(train_images)} images")
    print(f"Val   : {len(val_images)} images\n")

    # ── Modèle ───────────────────────────────────────────────
    pop = PopulationBMultiScale(
        num_cells=args.num_cells, patch_sizes=patch_sizes,
        theta_init=args.theta, beta=5.0,
        num_classes=NUM_CLASSES, K=args.K,
        use_intensity=False, device=DEVICE
    )

    # ── Initialisation ───────────────────────────────────────
    print("Initialisation depuis 50 premières images...")
    images_init = torch.stack(train_images[:50]).to(DEVICE)
    for scale_idx, ps in enumerate(pop.patch_sizes):
        patches = pop.extract_patches_batch(images_init, ps)
        patches_std = pop.preprocess_patches(patches)
        flat = patches_std.reshape(-1, patches_std.shape[-1])
        idx  = torch.randperm(flat.shape[0])[:pop.B_per_scale[scale_idx]]
        pop.prototypes[scale_idx] = flat[idx]
        print(f"  Échelle {scale_idx} : {pop.B_per_scale[scale_idx]} protos initialisés")

    # ── Entraînement ─────────────────────────────────────────
    start_time   = time.time()
    best_acc     = 0.0
    best_protos  = [p.clone() for p in pop.prototypes]
    best_counts  = [c.clone() for c in pop.class_counts]
    best_class   = [c.clone() for c in pop.proto_class]
    patience     = 0
    max_patience = 7
    history      = []

    images_t = torch.stack(train_images).to(DEVICE)

    for epoch in range(args.epochs):
        lr_epoch = args.lr * (0.95 ** epoch)

        # ── Train batch ──────────────────────────────────────
        for start in range(0, len(images_t), 2):
            end = min(start + 2, len(images_t))
            all_activated, all_z = pop.process_batch(images_t[start:end])

            if not any(a.any() for a in all_activated):
                continue

            labels_b = train_labels[start:end]
            labels_t = torch.tensor(labels_b, device=DEVICE, dtype=torch.long)
            N = end - start

            for scale_idx in range(pop.n_scales):
                activated = all_activated[scale_idx]
                z         = all_z[scale_idx]

                for i in range(N):
                    lbl = labels_t[i].item()
                    act = activated[i]
                    if not act.any():
                        continue

                    # ── Decay (ablation) ─────────────────────
                    if not args.no_decay:
                        pop.class_counts[scale_idx][act] *= 0.99
                    pop.class_counts[scale_idx][act, lbl] += 1

                    # Classe temporaire
                    pop.proto_class[scale_idx] = \
                        pop.class_counts[scale_idx].argmax(dim=1)
                    pop.proto_class[scale_idx][
                        pop.class_counts[scale_idx].sum(dim=1) == 0] = -1

                    # LVQ update
                    act_i         = activated[i]
                    z_active      = z[i][act_i]
                    protos_active = pop.prototypes[scale_idx][act_i]
                    classes_active= pop.proto_class[scale_idx][act_i]
                    diff          = z_active - protos_active

                    same = (classes_active == lbl) & (classes_active >= 0)
                    diff_c = (classes_active != lbl) & (classes_active >= 0)

                    grads = torch.zeros_like(diff)
                    grads[same]   =  diff[same]
                    grads[diff_c] = -diff[diff_c]

                    idx_act = torch.where(act_i)[0]
                    updates = torch.zeros_like(pop.prototypes[scale_idx])
                    updates.index_add_(0, idx_act, grads)
                    pop.prototypes[scale_idx] += lr_epoch * updates
                    pop.prototypes[scale_idx].clamp_(-5.0, 5.0)

        # ── Reassign ─────────────────────────────────────────
        for scale_idx in range(pop.n_scales):
            pop.class_counts[scale_idx].zero_()

        for start in range(0, len(images_t), 2):
            end = min(start + 2, len(images_t))
            all_activated, _ = pop.process_batch(images_t[start:end])
            lbls_b = train_labels[start:end]

            for scale_idx in range(pop.n_scales):
                activated = all_activated[scale_idx]
                for i in range(end - start):
                    lbl = lbls_b[i] if isinstance(lbls_b[i], int) \
                          else lbls_b[i].item()
                    pop.class_counts[scale_idx][activated[i], lbl] += 1

        for scale_idx in range(pop.n_scales):
            assigned = pop.class_counts[scale_idx].sum(dim=1) > 0

            if args.no_norm:
                # Sans normalisation fréquence
                counts = pop.class_counts[scale_idx]
            else:
                # Avec normalisation fréquence
                class_freq  = pop.class_counts[scale_idx].sum(dim=0).clamp(min=1)
                counts = pop.class_counts[scale_idx] / class_freq.unsqueeze(0)

            pop.proto_class[scale_idx][assigned]  = \
                counts[assigned].argmax(dim=1)
            pop.proto_class[scale_idx][~assigned] = -1

        # ── Évaluation ───────────────────────────────────────
        val_t  = torch.stack(val_images).to(DEVICE)
        preds  = []

        for start in range(0, len(val_t), 4):
            end = min(start + 4, len(val_t))
            all_activated, _ = pop.process_batch(val_t[start:end])

            for i in range(end - start):
                total_votes = torch.zeros(NUM_CLASSES, device=DEVICE)

                for scale_idx in range(pop.n_scales):
                    act_i = all_activated[scale_idx][i]
                    valid = act_i & (pop.proto_class[scale_idx] >= 0)
                    if not valid.any():
                        continue

                    total  = pop.class_counts[scale_idx].sum(
                        dim=1, keepdim=True).clamp(min=1)
                    freq   = pop.class_counts[scale_idx] / total

                    if args.uniform_vote:
                        # Vote uniforme
                        weights = torch.ones(
                            pop.B_per_scale[scale_idx], device=DEVICE)
                    else:
                        max_freq  = freq.max(dim=1).values
                        mean_freq = freq.mean(dim=1)
                        weights   = (max_freq - mean_freq) * 2

                    active_freq    = freq[valid]
                    active_weights = weights[valid]
                    votes = (active_freq * 
                             active_weights.unsqueeze(1)).sum(dim=0)
                    total_votes += votes

                if total_votes.sum() == 0:
                    preds.append(None)
                else:
                    preds.append(total_votes.argmax().item())

        correct = sum(p == l for p, l in zip(preds, val_labels) 
                      if p is not None)
        acc     = correct / len(val_labels)
        history.append(acc)

        if acc > best_acc:
            best_acc    = acc
            best_protos = [p.clone() for p in pop.prototypes]
            best_counts = [c.clone() for c in pop.class_counts]
            best_class  = [c.clone() for c in pop.proto_class]
            patience    = 0
            marker      = "✅"
        else:
            patience += 1
            marker    = f"  (patience {patience}/{max_patience})"

        print(f"  Epoch {epoch+1:2d} | Acc: {acc:.4f} | "
              f"Best: {best_acc:.4f} | lr: {lr_epoch:.4f} {marker}")

        if patience >= max_patience:
            print(f"\n  Early stopping à l'epoch {epoch+1}")
            break

    pop.prototypes   = best_protos
    pop.class_counts = best_counts
    pop.proto_class  = best_class

    elapsed_min = (time.time() - start_time) / 60
    best_epoch  = int(history.index(max(history))) + 1

    print(f"\n✅ {run_name} → {best_acc:.4f} | "
          f"Best epoch: {best_epoch}/{len(history)} | "
          f"Temps: {elapsed_min:.1f} min")

    # ── Sauvegarde ───────────────────────────────────────────
    result = {
        "run_name"    : run_name,
        "no_decay"    : args.no_decay,
        "uniform_vote": args.uniform_vote,
        "no_norm"     : args.no_norm,
        "seed"        : args.seed,
        "acc"         : best_acc,
        "best_epoch"  : best_epoch,
        "n_epochs"    : len(history),
        "time_min"    : elapsed_min,
        "history"     : history,
        "val_set"     : "clean_773",
    }

    with open(f"figs/{run_name}.json", "w") as f:
        json.dump(result, f, indent=2)
    shutil.copy(f"figs/{run_name}.json",
                f"{DRIVE_PATH}{run_name}.json")

    save_model(pop, path=f"figs/model_{run_name}.pt")
    shutil.copy(f"figs/model_{run_name}.pt",
                f"{DRIVE_PATH}model_{run_name}.pt")

    print(f"✅ Sauvegardé : {DRIVE_PATH}{run_name}.json")
    return best_acc, history


if __name__ == "__main__":
    run(parse_args())