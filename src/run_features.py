"""Approach 3: frozen MobileNetV3-Small features + traditional scikit-learn classifiers.

    python src/run_features.py

The pre-trained network is frozen. For each image we take the output of
`model.features` followed by the network's global average pooling and flatten it
(576-dimensional vector), i.e. the input of the original classifier. Features are
extracted once (no augmentation) for train / val / test.

Feature scaling (StandardScaler) is fitted on the training features only.
For each classifier the regularisation strength C (or the forest size) is chosen
on the validation set; the selected classifier is then evaluated once on test.
"""
import argparse
import json

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from common import (NUM_CLASSES, Timer, build_model, common_args, compute_metrics,
                    environment_info, get_data, get_device, make_loader, results_dir,
                    save_confusion, save_predictions, set_seed)


@torch.no_grad()
def extract(model, ds, device, batch_size, seed, num_workers):
    """Features after the pooling layer, before the classifier."""
    model.eval()
    loader = make_loader(ds, batch_size, False, seed, num_workers)
    feats, ys = [], []
    for x, y in loader:
        f = model.avgpool(model.features(x.to(device))).flatten(1)
        feats.append(f.cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(feats), np.concatenate(ys)


def candidates(seed):
    return {
        "logistic_regression": [(f"C={c}", c, lambda c=c: make_pipeline(
            StandardScaler(), LogisticRegression(C=c, max_iter=2000)))
            for c in (0.01, 0.1, 1.0)],
        "linear_svm": [(f"C={c}", c, lambda c=c: make_pipeline(
            StandardScaler(), LinearSVC(C=c, max_iter=5000, random_state=seed)))
            for c in (0.001, 0.01, 0.1)],
        "random_forest": [(f"n_estimators={n}", n, lambda n=n: RandomForestClassifier(
            n_estimators=n, n_jobs=-1, random_state=seed))
            for n in (100, 300)],
    }


def main():
    ap = common_args(argparse.ArgumentParser())
    args = ap.parse_args()
    set_seed(args.seed, not args.no_deterministic)
    device = get_device()
    out = results_dir(args)
    print("Environment:", json.dumps(environment_info()))
    data = get_data(args)

    model = build_model(args.smoke).to(device)
    for p in model.parameters():
        p.requires_grad = False  # frozen network

    cache = out / "features.npz"
    with Timer() as t_extract:
        Xtr, ytr = extract(model, data["train_eval"], device, args.batch_size, args.seed, args.num_workers)
        Xva, yva = extract(model, data["val"], device, args.batch_size, args.seed, args.num_workers)
        Xte, yte = extract(model, data["test"], device, args.batch_size, args.seed, args.num_workers)
    np.savez_compressed(cache, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte)
    print(f"features: train {Xtr.shape}, val {Xva.shape}, test {Xte.shape} "
          f"in {t_extract.elapsed:.1f}s", flush=True)

    summary = []
    for clf_name, grid in candidates(args.seed).items():
        best = None
        for label, value, factory in grid:
            with Timer() as t_fit:
                clf = factory().fit(Xtr, ytr)  # scaler (if any) fitted on train only
            val_acc = compute_metrics(yva, clf.predict(Xva))["accuracy"]
            print(f"[{clf_name}] {label}: val_acc={val_acc:.4f} (fit {t_fit.elapsed:.1f}s)", flush=True)
            if best is None or val_acc > best["val_acc"]:
                best = {"label": label, "value": value, "val_acc": val_acc,
                        "fit_seconds": t_fit.elapsed, "clf": clf}
        clf = best["clf"]
        pred_te = clf.predict(Xte)  # test set evaluated once per selected classifier
        res = {
            "approach": f"features_{clf_name}",
            "selected": {"setting": best["label"], "rule": "best validation accuracy"},
            "trainable_parameters": "0 (frozen CNN) + sklearn classifier",
            "feature_dim": int(Xtr.shape[1]),
            "feature_extraction_seconds": t_extract.elapsed,
            "classifier_fit_seconds": best["fit_seconds"],
            "total_seconds": t_extract.elapsed + best["fit_seconds"],
            "train": compute_metrics(ytr, clf.predict(Xtr)),
            "val": compute_metrics(yva, clf.predict(Xva)),
            "test": compute_metrics(yte, pred_te),
            "preprocessing": "StandardScaler fitted on training features"
                             if clf_name != "random_forest" else "none (trees are scale-invariant)",
            "environment": environment_info(),
            "smoke_test": bool(args.smoke),
        }
        name = res["approach"]
        (out / f"{name}_metrics.json").write_text(json.dumps(res, indent=2))
        save_predictions(out / f"{name}_test_predictions.csv", yte, pred_te)
        save_confusion(out / f"{name}_confusion.npy", yte, pred_te)
        summary.append((name, res["selected"]["setting"], res["train"]["accuracy"],
                        res["test"]["accuracy"], res["test"]["macro_f1"]))
    print("\nSummary (approach, setting, train acc, test acc, test macro-F1):")
    for s in summary:
        print("  %-28s %-18s %.4f  %.4f  %.4f" % s)


if __name__ == "__main__":
    main()
