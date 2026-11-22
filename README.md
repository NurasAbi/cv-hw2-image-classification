# Homework 2 - Image Classification with Transfer Learning (Computer Vision, Fall 2026)

Author: Nuras Assylbekov (matricola VR565535)

Three ways of adapting an ImageNet-pre-trained **MobileNetV3-Small** (torchvision,
`MobileNet_V3_Small_Weights.DEFAULT` = ImageNet-1K) to the **37 breeds** of the
Oxford-IIIT Pet dataset, compared with an identical data protocol:

| # | Approach | What is trained |
|---|----------|-----------------|
| 1 | Full fine-tuning | all parameters (1,555,781 incl. the new 37-way layer) |
| 2 | Classifier-head training | `classifier[0]` Linear(576,1024) and the new `classifier[3]` Linear(1024,37) = 628,773 parameters; `model.features` frozen (BatchNorm statistics frozen too) |
| 3 | Feature extraction + scikit-learn | nothing in the CNN; 576-d pooled features -> logistic regression / linear SVM / random forest |

## Dataset

[Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) (Parkhi, Vedaldi, Zisserman,
Jawahar, *Cats and Dogs*, CVPR 2012). Distributed by the authors under
**Creative Commons Attribution-ShareAlike 4.0** (check the dataset page for the current terms).
The dataset is **not** stored in this repository; it is downloaded automatically by
`torchvision.datasets.OxfordIIITPet(download=True)` into `./data` the first time a script runs.

Splits: official `trainval` (3,680 images) is split 80/20 **stratified by breed** into
train / validation (`random_state=42`); official `test` (3,669 images) is used only for the final
evaluation. The exact indices are in `splits/split_indices.json`; per-breed counts are in
`results/tables/split_counts.csv`.

Preprocessing: the transform shipped with the weights (resize 256, center crop 224, ImageNet
mean/std) for validation, test and feature extraction. Training augmentation (training set only):
`RandomResizedCrop(224, scale=(0.6,1))`, horizontal flip, light colour jitter; same normalisation.
No class re-balancing (the dataset is nearly balanced, ~100 images per breed).

## Environment

The reported numbers were produced with Python 3.11.1, PyTorch 2.15.0 (CPU build), TorchVision 0.29.2,
scikit-learn 1.9.1 and NumPy 2.2.4 on a Windows laptop CPU (no GPU); the exact versions are also stored in
`results/*_metrics.json -> environment`.

```
pip install -r requirements.txt
```

(On Windows with an NVIDIA GPU, install the CUDA build of PyTorch from https://pytorch.org first.)

## Reproduce the experiments

Run from the repository root:

```
python src/run_finetune.py --mode full      # approach 1
python src/run_finetune.py --mode head      # approach 2
python src/run_features.py                  # approach 3 (3 classifiers)
python src/make_figures.py                  # tables + figures
python src/analysis.py                      # confidence intervals, McNemar tests, per-class recall
python src/make_report_assets.py            # LaTeX tables and the confusion-matrix figure of the report
```

The report is `report/Report_HW2_Assylbekov.pdf` (source: `report/report.tex`, compiled with `pdflatex`).

Defaults: AdamW, cross-entropy, batch size 64, cosine learning-rate schedule, weight decay 1e-4;
full fine-tuning 8 epochs with lr in {1e-4, 3e-4}; head training 10 epochs with lr in {1e-3, 3e-3}.
Options: `--epochs`, `--lrs`, `--batch-size`, `--num-workers` (use 0 if you get DataLoader errors on Windows),
`--seed`, `--no-deterministic`.

**Model selection** uses only the validation set: for each candidate learning rate the epoch with
the highest validation accuracy is kept; the best (lr, epoch) pair is selected. For the
scikit-learn classifiers, `C` (or the number of trees) is chosen on the validation set;
`StandardScaler` is fitted on training features only. The test set is evaluated once per selected
model, after selection. It is never used for early stopping, feature selection or tuning.

Reproducibility: seeds for Python, NumPy and PyTorch are set (`--seed`, default 0) and
`torch.use_deterministic_algorithms(True, warn_only=True)` + deterministic cuDNN are enabled.
Remaining sources of variation: GPU kernels without deterministic implementations (PyTorch warns
about them), different hardware / library versions, multi-process data loading order.
Differences of a fraction of a percent between runs should not be interpreted as meaningful.

## Main results (test set, 3,669 images)

| Approach | Trainable params | Test acc. | Macro-F1 | Time (CPU) |
|---|---:|---:|---:|---:|
| 1. Full fine-tuning | 1,555,781 | 82.0 % | 81.6 % | 623 s |
| 2. Classifier-head training | 628,773 | 86.9 % | 86.8 % | 406 s |
| 3a. Frozen features + logistic regression | 0 (CNN) | 85.7 % | 85.5 % | 230 s |
| 3b. Frozen features + linear SVM | 0 (CNN) | 83.3 % | 83.1 % | 246 s |
| 3c. Frozen features + random forest | 0 (CNN) | 82.7 % | 82.2 % | 237 s |

Single run per approach (seed 0); see the report for confidence intervals, significance tests and limitations.

## Outputs (`results/`)

- `*_metrics.json`: selected hyper-parameters, trainable parameters, train/val/test accuracy and
  macro-F1, timings, environment.
- `*_confusion.npy`, `*_test_predictions.csv`: confusion matrices and per-image predictions.
- `full_history.csv`, `head_history.csv`: per-epoch training loss and validation accuracy.
- `checkpoints/{full,head}_best.pt`: selected checkpoints (~6 MB each).
- `tables/`: `results_table.{csv,md,tex}`, `split_counts.csv`, `top_confusions_*.csv`,
  `accuracy_ci.csv`, `mcnemar.csv`, `per_class_recall.csv`, `error_types.csv`.
- `report_assets/`: tables and combined confusion-matrix figure included in the report.
- `figures/`: confusion matrices, validation curves, prediction examples, split counts.

## Smoke test

`--smoke` runs the same code on tiny synthetic images with random weights (no download). It is
only used to check that the pipeline runs; its output (`results_smoke/`) is not a result.

## Layout

```
src/common.py          seeding, data, splits, transforms, model, metrics
src/run_finetune.py    approaches 1 and 2
src/run_features.py    approach 3
src/make_figures.py    tables and figures
src/analysis.py        confidence intervals, McNemar tests, per-class recall, error types
src/make_report_assets.py   LaTeX tables / figure for the report
splits/                split_indices.json (created on first run)
results/               metrics, predictions, checkpoints, tables, figures
report/                PDF report
```
