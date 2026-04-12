# baselines.py — À ajouter au repo
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch

def run_baselines(train_images, train_labels, val_images, val_labels):

    print("\n=== BASELINES ===\n")

    X_tr = torch.stack(train_images).numpy().reshape(len(train_images), -1)
    X_vl = torch.stack(val_images).numpy().reshape(len(val_images), -1)
    y_tr = np.array(train_labels)
    y_vl = np.array(val_labels)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_vl_s = scaler.transform(X_vl)

    baselines = {
        "kNN (k=5)"  : KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "SVM (RBF)"  : SVC(kernel="rbf", C=1.0),
        "MLP"        : MLPClassifier(hidden_layer_sizes=(256, 128),
                                     max_iter=50, random_state=42),
    }

    results = {}
    for name, clf in baselines.items():
        clf.fit(X_tr_s, y_tr)
        preds = clf.predict(X_vl_s)
        acc   = accuracy_score(y_vl, preds)
        f1    = f1_score(y_vl, preds, average="macro")
        results[name] = {"acc": acc, "f1": f1}
        print(f"  {name:15s} → Acc: {acc:.4f} | F1: {f1:.4f}")

    return results