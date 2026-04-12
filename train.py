# ============================================================
# train.py — Fonctions d'entraînement
# ============================================================

import torch
from model import PopulationBFastExact, TrainerFastExact


def init_prototypes_from_data(population, images, device, n_samples=200):
    import torch.nn.functional as F
    imgs_batch = torch.stack(images[:n_samples]).to(device)
    patches    = population.extract_patches_batch(imgs_batch)
    patches    = population.preprocess_patches(patches)
    patches    = patches.reshape(-1, patches.shape[2])
    idx = torch.randperm(patches.shape[0])[:population.B]
    population.prototypes = patches[idx].clone().to(device)
    print(f"Prototypes initialisés depuis {patches.shape[0]} patches réels")


def run_experiment(train_images, train_labels, val_images, val_labels,
                   name, num_classes, epochs=40, lr=0.1,
                   num_cells=1600, patch_size=(5, 5),
                   theta_init=0.5, device="cuda"):

    print(f"\n{'='*50}")
    print(f"EXPÉRIENCE : {name}")
    print(f"{'='*50}")

    pop = PopulationBFastExact(
        num_cells=num_cells,
        patch_size=patch_size,
        theta_init=theta_init,
        beta=5.0,
        num_classes=num_classes,
        device=device
    )
    trainer = TrainerFastExact(population=pop, num_classes=num_classes, device=device)
    init_prototypes_from_data(pop, train_images, device, n_samples=200)

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
            marker           = f"⚠️  restauré (patience {patience}/{max_patience})"
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