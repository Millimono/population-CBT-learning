# baselines_svm_patch_cli.py
import sys, argparse, json, shutil, os, time
import numpy as np
import torch
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, '/content/population-CBT-learning')

CACHE_CLEAN = "/content/drive/MyDrive/MiniDDSM/miniddsm_val_clean_256.pt"
CACHE_TRAIN = "/content/drive/MyDrive/MiniDDSM/miniddsm_cache_256.pt"
DRIVE_PATH  = "/content/drive/MyDrive/ablation_results/baselines/"


def extract_patch_features(images, patch_size=18, agg="mean"):
    """
    Extraire features patch-based pour SVM.
    Aggrège les patches par moyenne ou max.
    → vecteur 324-dim par image
    """
    import torch.nn.functional as F
    
    features = []
    images_t = torch.stack(images)
    
    for i in range(0, len(images_t), 32):
        batch = images_t[i:i+32]
        
        # Extraire patches 18×18
        patches = F.unfold(
            batch.unsqueeze(1),
            kernel_size=(patch_size, patch_size),
            stride=1
        ).transpose(1, 2)
        # shape : (batch, P, 324)
        
        # Z-score par patch
        mean = patches.mean(dim=-1, keepdim=True)
        std  = patches.std(dim=-1, keepdim=True).clamp(min=1e-8)
        patches = (patches - mean) / std
        
        # Agréger
        if agg == "mean":
            feat = patches.mean(dim=1)  # (batch, 324)
        elif agg == "max":
            feat = patches.max(dim=1).values  # (batch, 324)
        
        features.append(feat.cpu().numpy())
    
    return np.vstack(features)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agg",  type=str, default="mean",
                   choices=["mean", "max"])
    p.add_argument("--C",    type=float, default=1.0)
    p.add_argument("--name", type=str,   default=None)
    if "ipykernel" in sys.modules:
        return p.parse_args(args=[])
    return p.parse_args()


def run(args):
    run_name = args.name or f"svm_patch_{args.agg}_C{args.C}"
    
    print(f"\n{'='*70}\nRUN : {run_name}\n{'='*70}\n")
    os.makedirs(DRIVE_PATH, exist_ok=True)

    # ── Charger données ──────────────────────────────────────
    data_train = torch.load(CACHE_TRAIN)
    data_clean = torch.load(CACHE_CLEAN)

    train_images = data_train["train_images"]
    train_labels = np.array(data_train["train_labels"])
    val_images   = data_clean["val_images"]
    val_labels   = np.array(data_clean["val_labels"])

    # ── Extraire features patch ──────────────────────────────
    print(f"Extraction features patch (agg={args.agg})...")
    start = time.time()
    
    X_train = extract_patch_features(train_images, agg=args.agg)
    X_val   = extract_patch_features(val_images,   agg=args.agg)
    
    print(f"Train : {X_train.shape}")
    print(f"Val   : {X_val.shape}")

    # Normalisation
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    # ── SVM ──────────────────────────────────────────────────
    print(f"\nEntraînement SVM (C={args.C})...")
    svm = SVC(C=args.C, kernel='rbf', probability=True)
    svm.fit(X_train, train_labels)

    elapsed_min = (time.time() - start) / 60

    # ── Métriques ────────────────────────────────────────────
    preds  = svm.predict(X_val)
    scores = svm.predict_proba(X_val)[:, 1]

    acc    = accuracy_score(val_labels, preds)
    f1     = f1_score(val_labels, preds, average='macro')
    f1s    = f1_score(val_labels, preds, average=None)
    auc    = roc_auc_score(val_labels, scores)

    print(f"\n✅ {run_name} :")
    print(f"   Accuracy  : {acc:.4f}")
    print(f"   F1 macro  : {f1:.4f}")
    print(f"   F1 Cancer : {f1s[0]:.4f}")
    print(f"   F1 Normal : {f1s[1]:.4f}")
    print(f"   AUC       : {auc:.4f}")
    print(f"   Temps     : {elapsed_min:.1f} min")

    # ── Sauvegarder ─────────────────────────────────────────
    result = {
        "run_name" : run_name,
        "method"   : "SVM-patch",
        "agg"      : args.agg,
        "C"        : args.C,
        "acc"      : float(acc),
        "f1_macro" : float(f1),
        "f1_cancer": float(f1s[0]),
        "f1_normal": float(f1s[1]),
        "auc"      : float(auc),
        "time_min" : elapsed_min,
        "val_set"  : "clean_773",
    }

    out = f"{DRIVE_PATH}{run_name}.json"
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"✅ Sauvegardé : {out}")
    return result


if __name__ == "__main__":
    run(parse_args())