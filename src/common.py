"""Shared utilities for Homework 2 (transfer learning on Oxford-IIIT Pet).

Contains: seeding, data loading, stratified train/val split, transforms,
model construction and metric computation. Every script in this repository
imports from here so that the three approaches use *exactly* the same data
splits, preprocessing and evaluation protocol.
"""
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parent.parent
NUM_CLASSES = 37
SPLIT_SEED = 42          # seed of the stratified train/val split (kept fixed)
VAL_FRACTION = 0.2       # fraction of the official trainval used as validation


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch. Optionally request deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: a few ops have no deterministic version; they are reported
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def environment_info() -> dict:
    import sklearn
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
class TransformedSubset(Dataset):
    """A subset (by index) of a base dataset with its own transform.

    The base dataset returns PIL images; the transform is applied here so that
    the training subset can use augmentation while val/test do not.
    """

    def __init__(self, base, indices, transform):
        self.base = base
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base[self.indices[i]]
        return self.transform(img), label


class SyntheticPets(Dataset):
    """Tiny synthetic stand-in for the dataset, used ONLY by --smoke tests.

    Each class has a characteristic colour plus noise, so a model can learn it.
    """

    def __init__(self, n, seed):
        rng = np.random.default_rng(seed)
        self._labels = np.array([i % NUM_CLASSES for i in range(n)])
        self.colors = np.random.default_rng(123).integers(0, 255, (NUM_CLASSES, 3))
        self.noise_seed = seed * 100003
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        from PIL import Image
        rng = np.random.default_rng(self.noise_seed + i)
        base = np.ones((280, 280, 3)) * self.colors[self._labels[i]]
        arr = np.clip(base + rng.normal(0, 40, base.shape), 0, 255).astype(np.uint8)
        return Image.fromarray(arr), int(self._labels[i])


def get_weights(smoke: bool = False):
    if smoke:  # no download; random weights, only for the pipeline smoke test
        return None
    return models.MobileNet_V3_Small_Weights.DEFAULT


def get_transforms(smoke: bool = False):
    """Return (train_transform, eval_transform).

    The evaluation transform is the one shipped with the pre-trained weights
    (resize 256 -> center crop 224 -> rescale to [0,1] -> ImageNet normalisation).
    The training transform keeps the same normalisation but replaces the
    deterministic resize/crop with RandomResizedCrop + horizontal flip +
    light colour jitter (augmentation is applied to the training set only).
    """
    weights = get_weights(smoke)
    if weights is not None:
        eval_tf = weights.transforms()
        mean, std = eval_tf.mean, eval_tf.std
    else:
        eval_tf = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


def load_base_datasets(data_root: str, smoke: bool = False):
    """Return (trainval_base, test_base, class_names).

    Official partitions of Oxford-IIIT Pet: 'trainval' (3680 images) and
    'test' (3669 images). Labels are the 37 breed categories.
    """
    if smoke:
        n_trainval, n_test = NUM_CLASSES * 12, NUM_CLASSES * 4
        names = [f"class_{i:02d}" for i in range(NUM_CLASSES)]
        return SyntheticPets(n_trainval, 1), SyntheticPets(n_test, 2), names
    trainval = datasets.OxfordIIITPet(data_root, split="trainval",
                                      target_types="category", download=True)
    test = datasets.OxfordIIITPet(data_root, split="test",
                                  target_types="category", download=True)
    return trainval, test, list(trainval.classes)


def base_labels(base) -> np.ndarray:
    for attr in ("_labels", "targets"):
        if hasattr(base, attr):
            return np.asarray(getattr(base, attr))
    return np.asarray([base[i][1] for i in range(len(base))])


def make_splits(trainval_base, split_dir: Path):
    """Stratified (by breed) train/val split of the official trainval partition.

    The indices are saved to split_dir/split_indices.json the first time and
    re-used afterwards, so every approach sees identical partitions.
    """
    split_dir.mkdir(parents=True, exist_ok=True)
    f = split_dir / "split_indices.json"
    labels = base_labels(trainval_base)
    if f.exists():
        d = json.loads(f.read_text())
        if len(labels) == d["n_trainval"]:
            return d["train"], d["val"]
    idx = np.arange(len(labels))
    tr, va = train_test_split(idx, test_size=VAL_FRACTION, stratify=labels,
                              random_state=SPLIT_SEED)
    f.write_text(json.dumps({"seed": SPLIT_SEED, "val_fraction": VAL_FRACTION,
                             "n_trainval": int(len(labels)),
                             "train": sorted(int(i) for i in tr),
                             "val": sorted(int(i) for i in va)}))
    return sorted(int(i) for i in tr), sorted(int(i) for i in va)


def get_data(args):
    """Build all datasets/loaders. Returns a dict."""
    trainval, test, names = load_base_datasets(args.data_root, args.smoke)
    split_dir = ROOT / ("splits_smoke" if args.smoke else "splits")
    tr_idx, va_idx = make_splits(trainval, split_dir)
    train_tf, eval_tf = get_transforms(args.smoke)
    out = {
        "class_names": names,
        "trainval_base": trainval,
        "test_base": test,
        "train_idx": tr_idx,
        "val_idx": va_idx,
        "train_aug": TransformedSubset(trainval, tr_idx, train_tf),
        "train_eval": TransformedSubset(trainval, tr_idx, eval_tf),
        "val": TransformedSubset(trainval, va_idx, eval_tf),
        "test": TransformedSubset(test, list(range(len(test))), eval_tf),
    }
    return out


def make_loader(ds, batch_size, shuffle, seed, num_workers):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, generator=g,
                      worker_init_fn=worker_init_fn, pin_memory=torch.cuda.is_available())


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_model(smoke: bool = False) -> torch.nn.Module:
    """MobileNetV3-Small pre-trained on ImageNet-1K with a new 37-way last layer.

    torchvision's classifier is:
        [0] Linear(576, 1024)  [1] Hardswish  [2] Dropout(0.2)  [3] Linear(1024, 1000)
    Only classifier[3] is replaced (-> Linear(1024, 37)).
    """
    weights = get_weights(smoke)
    model = models.mobilenet_v3_small(weights=weights)
    in_f = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(in_f, NUM_CLASSES)
    return model


def count_params(model, trainable_only=True) -> int:
    return sum(p.numel() for p in model.parameters()
               if (p.requires_grad or not trainable_only))


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps, probs = [], [], []
    for x, y in loader:
        logits = model(x.to(device))
        pr = torch.softmax(logits, 1).cpu()
        ps.append(pr.argmax(1).numpy())
        probs.append(pr.max(1).values.numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(probs)


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   labels=list(range(NUM_CLASSES)), zero_division=0)),
    }


def save_predictions(path: Path, y_true, y_pred, conf=None):
    import pandas as pd
    df = pd.DataFrame({"test_index": np.arange(len(y_true)), "true": y_true, "pred": y_pred})
    if conf is not None:
        df["confidence"] = conf
    df.to_csv(path, index=False)


def save_confusion(path: Path, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    np.save(path, cm)


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.elapsed = time.perf_counter() - self.t


def common_args(p):
    p.add_argument("--data-root", default=str(ROOT / "data"),
                   help="where the Oxford-IIIT Pet dataset is stored / downloaded")
    p.add_argument("--seed", type=int, default=0, help="training seed")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--no-deterministic", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="pipeline smoke test on synthetic data (random weights, NOT a result)")
    p.add_argument("--out", default=None, help="results directory")
    return p


def results_dir(args) -> Path:
    d = Path(args.out) if args.out else ROOT / ("results_smoke" if args.smoke else "results")
    d.mkdir(parents=True, exist_ok=True)
    return d
