"""Build the LaTeX tables and the combined figure used in report/report.tex.

    python src/make_report_assets.py

Needs the outputs of run_finetune.py, run_features.py, make_figures.py and
analysis.py. Writes results/report_assets/{table_results.tex, table_confusions.tex,
confusion_pair.pdf}.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PRETTY = {"full_finetuning": "1. Full fine-tuning",
          "head_training": "2. Classifier-head training",
          "features_logistic_regression": "3a. Features + logistic regression",
          "features_linear_svm": "3b. Features + linear SVM",
          "features_random_forest": "3c. Features + random forest"}


def pct(x):
    return f"{100 * x:.1f}"


def results_table(out, assets):
    ci = pd.read_csv(out / "tables" / "accuracy_ci.csv").set_index("approach")
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Approach & Train. params & Train & Val & Test acc. [95\% CI] & Test macro-F1 & Gap & Time (s) \\",
             r"\midrule"]
    for name, label in PRETTY.items():
        r = json.loads((out / f"{name}_metrics.json").read_text())
        if "trainable_parameters" in r and isinstance(r["trainable_parameters"], int):
            params = f"{r['trainable_parameters']:,}".replace(",", r"\,")
            t = r["train_seconds_selected_run_until_best"]
        else:
            params = "0 (CNN)"
            t = r["total_seconds"]
        lo, hi = ci.loc[name, "ci95_low"], ci.loc[name, "ci95_high"]
        gap = r["train"]["accuracy"] - r["test"]["accuracy"]
        lines.append(f"{label} & {params} & {pct(r['train']['accuracy'])} & {pct(r['val']['accuracy'])} & "
                     f"\\textbf{{{pct(r['test']['accuracy'])}}} [{pct(lo)}, {pct(hi)}] & "
                     f"{pct(r['test']['macro_f1'])} & {100 * gap:.1f} & {t:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (assets / "table_results.tex").write_text("\n".join(lines))


def confusion_table(out, assets, names):
    cms = {n: np.load(out / f"{n}_confusion.npy") for n in
           ["head_training", "full_finetuning", "features_logistic_regression"]}
    pairs = [("American Pit Bull Terrier", "Staffordshire Bull Terrier"),
             ("American Pit Bull Terrier", "American Bulldog"),
             ("Ragdoll", "Birman"), ("Birman", "Ragdoll"),
             ("British Shorthair", "Russian Blue"), ("Russian Blue", "British Shorthair"),
             ("Chihuahua", "Shiba Inu"), ("Maine Coon", "Bengal")]
    idx = {n: i for i, n in enumerate(names)}
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"True breed & Predicted as & Head & Full FT & Log. reg. \\", r"\midrule"]
    for a, b in pairs:
        vals = [int(cms[k][idx[a], idx[b]]) for k in
                ["head_training", "full_finetuning", "features_logistic_regression"]]
        lines.append(f"{a} & {b} & {vals[0]} & {vals[1]} & {vals[2]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (assets / "table_confusions.tex").write_text("\n".join(lines))


def confusion_pair(out, assets, names):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8))
    for ax, key, title in zip(axes, ["full_finetuning", "head_training"],
                              ["1. Full fine-tuning", "2. Classifier-head training"]):
        cm = np.load(out / f"{key}_confusion.npy")
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=6.5)
        ax.set_yticklabels(names, fontsize=6.5)
        acc = np.trace(cm) / cm.sum()
        ax.set_title(f"{title} (test acc. {100 * acc:.1f}%)", fontsize=10)
        ax.set_xlabel("predicted", fontsize=9)
        ax.set_ylabel("true", fontsize=9)
    fig.tight_layout()
    fig.savefig(assets / "confusion_pair.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    args = ap.parse_args()
    out = Path(args.results)
    assets = out / "report_assets"
    assets.mkdir(exist_ok=True)
    names = pd.read_csv(out / "tables" / "split_counts.csv")["breed"].tolist()
    results_table(out, assets)
    confusion_table(out, assets, names)
    confusion_pair(out, assets, names)
    print("written to", assets)


if __name__ == "__main__":
    main()
