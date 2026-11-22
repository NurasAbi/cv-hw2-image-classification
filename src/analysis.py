"""Extra analysis of the saved test predictions (no model is run, no test-set tuning).

    python src/analysis.py

Produces in results/tables/:
  accuracy_ci.csv        test accuracy with 95% Wilson confidence interval
  mcnemar.csv            exact McNemar test between approaches on the same test images
  per_class_recall.csv   recall of every breed for every approach
  error_types.csv        share of errors that stay within a species (cat/cat, dog/dog)
These numbers are only descriptive statistics of the already selected models.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
ORDER = ["full_finetuning", "head_training", "features_logistic_regression",
         "features_linear_svm", "features_random_forest"]
CATS = {"Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair", "Egyptian Mau",
        "Maine Coon", "Persian", "Ragdoll", "Russian Blue", "Siamese", "Sphynx"}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    args = ap.parse_args()
    out = Path(args.results)
    tab = out / "tables"
    tab.mkdir(exist_ok=True)
    names = pd.read_csv(tab / "split_counts.csv")["breed"].tolist()
    preds = {n: pd.read_csv(out / f"{n}_test_predictions.csv")
             for n in ORDER if (out / f"{n}_test_predictions.csv").exists()}

    rows = []
    for n, d in preds.items():
        k, m = int((d.true == d.pred).sum()), len(d)
        lo, hi = wilson(k, m)
        rows.append({"approach": n, "n_test": m, "correct": k, "accuracy": k / m,
                     "ci95_low": lo, "ci95_high": hi})
    pd.DataFrame(rows).to_csv(tab / "accuracy_ci.csv", index=False)

    rows = []
    keys = list(preds)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            ca = (preds[a].true == preds[a].pred).to_numpy()
            cb = (preds[b].true == preds[b].pred).to_numpy()
            n01 = int((ca & ~cb).sum())   # a right, b wrong
            n10 = int((~ca & cb).sum())   # a wrong, b right
            p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
            rows.append({"a": a, "b": b, "a_only_correct": n01, "b_only_correct": n10,
                         "acc_diff_a_minus_b": (ca.mean() - cb.mean()), "p_value": p})
    pd.DataFrame(rows).to_csv(tab / "mcnemar.csv", index=False)

    rec = {}
    for n in preds:
        cm = np.load(out / f"{n}_confusion.npy")
        rec[n] = np.diag(cm) / np.maximum(cm.sum(1), 1)
    r = pd.DataFrame(rec, index=names)
    r.to_csv(tab / "per_class_recall.csv")

    rows = []
    for n in preds:
        cm = np.load(out / f"{n}_confusion.npy").copy()
        np.fill_diagonal(cm, 0)
        total = cm.sum()
        within = sum(cm[i, j] for i in range(len(names)) for j in range(len(names))
                     if (names[i] in CATS) == (names[j] in CATS))
        rows.append({"approach": n, "errors": int(total), "within_species": int(within),
                     "cross_species": int(total - within),
                     "share_within_species": within / total})
    pd.DataFrame(rows).to_csv(tab / "error_types.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nLowest-recall breeds (head_training):")
    if "head_training" in r:
        print(r["head_training"].sort_values().head(6).round(3).to_string())


if __name__ == "__main__":
    main()
