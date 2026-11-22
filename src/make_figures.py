"""Generate all tables and figures of the report from the saved results.

    python src/make_figures.py

Reads results/*_metrics.json, *_confusion.npy, *_test_predictions.csv and the
split files; writes into results/figures/ and results/tables/.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (NUM_CLASSES, ROOT, base_labels, common_args, get_data, results_dir)

ORDER = ["full_finetuning", "head_training", "features_logistic_regression",
         "features_linear_svm", "features_random_forest"]
PRETTY = {"full_finetuning": "1. Full fine-tuning",
          "head_training": "2. Classifier-head training",
          "features_logistic_regression": "3a. Features + logistic regression",
          "features_linear_svm": "3b. Features + linear SVM",
          "features_random_forest": "3c. Features + random forest"}


def load_results(out: Path):
    res = {}
    for name in ORDER:
        f = out / f"{name}_metrics.json"
        if f.exists():
            res[name] = json.loads(f.read_text())
    return res


def results_table(res, tab_dir):
    rows = []
    for name, r in res.items():
        if "train_seconds_selected_run_until_best" in r:
            cost = r["train_seconds_selected_run_until_best"]
            params = r["trainable_parameters"]
            cost_note = "training time up to the selected epoch"
        else:
            cost = r["total_seconds"]
            params = 0
            cost_note = "feature extraction + classifier fit"
        rows.append({
            "Approach": PRETTY[name],
            "Trainable params (NN)": params,
            "Train acc": round(r["train"]["accuracy"], 4),
            "Val acc": round(r["val"]["accuracy"], 4),
            "Test acc": round(r["test"]["accuracy"], 4),
            "Test macro-F1": round(r["test"]["macro_f1"], 4),
            "Train-test gap": round(r["train"]["accuracy"] - r["test"]["accuracy"], 4),
            "Time (s)": round(cost, 1),
            "Time covers": cost_note,
            "Selected": json.dumps(r["selected"].get("lr", r["selected"].get("setting"))),
        })
    df = pd.DataFrame(rows)
    df.to_csv(tab_dir / "results_table.csv", index=False)
    (tab_dir / "results_table.md").write_text(df.to_markdown(index=False))
    # compact LaTeX version for the report
    tex = df[["Approach", "Trainable params (NN)", "Train acc", "Test acc",
              "Test macro-F1", "Time (s)"]].to_latex(index=False, escape=True, float_format="%.3f")
    (tab_dir / "results_table.tex").write_text(tex)
    return df


def split_counts(data, names, fig_dir, tab_dir):
    tr = np.bincount(base_labels(data["trainval_base"])[data["train_idx"]], minlength=NUM_CLASSES)
    va = np.bincount(base_labels(data["trainval_base"])[data["val_idx"]], minlength=NUM_CLASSES)
    te = np.bincount(base_labels(data["test_base"]), minlength=NUM_CLASSES)
    df = pd.DataFrame({"breed": names, "train": tr, "val": va, "test": te})
    df["total"] = df[["train", "val", "test"]].sum(axis=1)
    df.to_csv(tab_dir / "split_counts.csv", index=False)
    tot = pd.DataFrame([{"breed": "TOTAL", **{c: int(df[c].sum()) for c in ["train", "val", "test", "total"]}}])
    pd.concat([df, tot]).to_csv(tab_dir / "split_counts_with_total.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    x = np.arange(NUM_CLASSES)
    ax.bar(x, tr, label="train")
    ax.bar(x, va, bottom=tr, label="val")
    ax.bar(x, te, bottom=tr + va, label="test")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("images")
    ax.set_title("Images per breed in each split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "split_counts.png", dpi=130)
    plt.close(fig)
    return df


def confusion_figures(res, out, names, fig_dir, tab_dir):
    for name in res:
        f = out / f"{name}_confusion.npy"
        if not f.exists():
            continue
        cm = np.load(f)
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(names, rotation=90, fontsize=6)
        ax.set_yticklabels(names, fontsize=6)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{PRETTY[name]} - test confusion matrix (row-normalised)\n"
                     f"accuracy {res[name]['test']['accuracy']:.3f}")
        fig.colorbar(im, fraction=0.04)
        fig.tight_layout()
        fig.savefig(fig_dir / f"confusion_{name}.png", dpi=130)
        plt.close(fig)
        # most confused pairs
        off = cm.copy()
        np.fill_diagonal(off, 0)
        pairs = [(names[i], names[j], int(off[i, j])) for i in range(NUM_CLASSES)
                 for j in range(NUM_CLASSES) if off[i, j] > 0]
        pairs.sort(key=lambda t: -t[2])
        pd.DataFrame(pairs[:15], columns=["true_breed", "predicted_as", "count"]).to_csv(
            tab_dir / f"top_confusions_{name}.csv", index=False)


def curves(out, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    any_plot = False
    for ax, mode, title in zip(axes, ["full", "head"], ["Full fine-tuning", "Classifier-head training"]):
        f = out / f"{mode}_history.csv"
        if not f.exists():
            continue
        h = pd.read_csv(f)
        for lr, g in h.groupby("lr"):
            ax.plot(g["epoch"], g["val_acc"], marker="o", label=f"lr={lr:g}")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
        any_plot = True
    axes[0].set_ylabel("validation accuracy")
    fig.tight_layout()
    if any_plot:
        fig.savefig(fig_dir / "validation_curves.png", dpi=130)
    plt.close(fig)


def prediction_examples(res, out, data, names, fig_dir, approach, k=6):
    f = out / f"{approach}_test_predictions.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    rng = np.random.default_rng(0)
    ok = df[df.true == df.pred]
    bad = df[df.true != df.pred]
    pick_ok = ok.iloc[rng.permutation(len(ok))[:k]] if len(ok) else ok
    pick_bad = bad.iloc[rng.permutation(len(bad))[:k]] if len(bad) else bad
    fig, axes = plt.subplots(2, k, figsize=(2.2 * k, 6.4))
    for row, (title, pick) in enumerate([("correct", pick_ok), ("incorrect", pick_bad)]):
        for col in range(k):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(pick):
                r = pick.iloc[col]
                img, _ = data["test_base"][int(r.test_index)]
                ax.imshow(img)
                color = "green" if row == 0 else "red"
                ax.set_title(f"true: {names[int(r.true)]}\npred: {names[int(r.pred)]}",
                             fontsize=7, color=color)
    fig.suptitle(f"{PRETTY[approach]} - random correct (top) and incorrect (bottom) test predictions",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.0)
    fig.savefig(fig_dir / f"examples_{approach}.png", dpi=130)
    plt.close(fig)


def main():
    ap = common_args(argparse.ArgumentParser())
    ap.add_argument("--examples-for", nargs="*", default=None,
                    help="approaches to draw prediction examples for (default: best and worst)")
    args = ap.parse_args()
    out = results_dir(args)
    fig_dir = out / "figures"
    tab_dir = out / "tables"
    fig_dir.mkdir(exist_ok=True)
    tab_dir.mkdir(exist_ok=True)
    res = load_results(out)
    if not res:
        raise SystemExit("No *_metrics.json found. Run the experiments first.")
    data = get_data(args)
    names = data["class_names"]
    split_counts(data, names, fig_dir, tab_dir)
    df = results_table(res, tab_dir)
    confusion_figures(res, out, names, fig_dir, tab_dir)
    curves(out, fig_dir)
    chosen = args.examples_for
    if chosen is None:
        by_acc = sorted(res, key=lambda n: res[n]["test"]["accuracy"])
        chosen = [by_acc[-1], by_acc[0]] if len(by_acc) > 1 else by_acc
    for a in chosen:
        prediction_examples(res, out, data, names, fig_dir, a)
    print(df.to_string(index=False))
    print(f"\nFigures: {fig_dir}\nTables:  {tab_dir}")


if __name__ == "__main__":
    main()
