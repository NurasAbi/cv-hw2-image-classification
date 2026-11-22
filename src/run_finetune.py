"""Approaches 1 and 2: full fine-tuning and classifier-head training.

    python src/run_finetune.py --mode full
    python src/run_finetune.py --mode head

--mode full : every parameter of MobileNetV3-Small is trained.
--mode head : the convolutional feature extractor (model.features) is frozen,
              including its BatchNorm statistics; the two fully connected layers
              of the classifier (classifier[0] and the new classifier[3]) are trained.

Model selection uses ONLY the validation set: for each candidate learning rate
the epoch with the best validation accuracy is kept as checkpoint, then the best
(learning rate, epoch) pair is chosen. The test set is evaluated exactly once,
after the selection, with the selected checkpoint.
"""
import argparse
import copy
import json

import pandas as pd
import torch
from torch import nn

from common import (ROOT, Timer, build_model, common_args, compute_metrics,
                    count_params, environment_info, get_data, get_device,
                    make_loader, predict, results_dir, save_confusion,
                    save_predictions, set_seed)


def configure_trainable(model, mode):
    if mode == "full":
        for p in model.parameters():
            p.requires_grad = True
    else:  # head
        for p in model.features.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True


def set_train_mode(model, mode):
    model.train()
    if mode == "head":
        model.features.eval()  # keep BatchNorm running statistics frozen


def trainable_names(model):
    return [n for n, p in model.named_parameters() if p.requires_grad]


def train_one_config(mode, lr, args, data, device):
    """Train with a given learning rate; return best-val checkpoint and history."""
    set_seed(args.seed, not args.no_deterministic)
    model = build_model(args.smoke).to(device)
    configure_trainable(model, mode)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = make_loader(data["train_aug"], args.batch_size, True, args.seed, args.num_workers)
    val_loader = make_loader(data["val"], args.batch_size, False, args.seed, args.num_workers)

    history, best = [], {"val_acc": -1.0}
    with Timer() as t_total:
        for epoch in range(1, args.epochs + 1):
            set_train_mode(model, mode)
            run_loss, n = 0.0, 0
            with Timer() as t_ep:
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad(set_to_none=True)
                    loss = loss_fn(model(x), y)
                    loss.backward()
                    opt.step()
                    run_loss += loss.item() * len(y)
                    n += len(y)
            sched.step()
            yv, pv, _ = predict(model, val_loader, device)
            val_acc = compute_metrics(yv, pv)["accuracy"]
            history.append({"mode": mode, "lr": lr, "epoch": epoch,
                            "train_loss": run_loss / n, "val_acc": val_acc,
                            "epoch_seconds": t_ep.elapsed})
            print(f"[{mode}] lr={lr:g} epoch {epoch}/{args.epochs} "
                  f"loss={run_loss / n:.4f} val_acc={val_acc:.4f} ({t_ep.elapsed:.0f}s)", flush=True)
            if val_acc > best["val_acc"]:
                best = {"val_acc": val_acc, "epoch": epoch, "lr": lr,
                        "state": copy.deepcopy(model.state_dict()),
                        "train_seconds_until_best": sum(h["epoch_seconds"] for h in history)}
    return best, history, t_total.elapsed, model


def main():
    ap = common_args(argparse.ArgumentParser())
    ap.add_argument("--mode", choices=["full", "head"], required=True)
    ap.add_argument("--lrs", type=float, nargs="+", default=None,
                    help="candidate learning rates (selected on the validation set)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    args = ap.parse_args()
    if args.lrs is None:
        args.lrs = [1e-4, 3e-4] if args.mode == "full" else [1e-3, 3e-3]
    if args.epochs is None:
        args.epochs = 8 if args.mode == "full" else 10
    if args.smoke:
        args.epochs = min(args.epochs, 2)

    device = get_device()
    out = results_dir(args)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    print("Environment:", json.dumps(environment_info()))
    data = get_data(args)

    all_hist, runs, best_overall = [], [], None
    total_time = 0.0
    for lr in args.lrs:
        best, hist, secs, model = train_one_config(args.mode, lr, args, data, device)
        all_hist += hist
        total_time += secs
        runs.append({"lr": lr, "best_epoch": best["epoch"], "best_val_acc": best["val_acc"],
                     "train_seconds_total": secs,
                     "train_seconds_until_best": best["train_seconds_until_best"]})
        if best_overall is None or best["val_acc"] > best_overall["val_acc"]:
            best_overall = best
    pd.DataFrame(all_hist).to_csv(out / f"{args.mode}_history.csv", index=False)

    # ---- final evaluation of the selected checkpoint (test set used once) ----
    model = build_model(args.smoke).to(device)
    configure_trainable(model, args.mode)
    model.load_state_dict(best_overall["state"])
    n_train_params = count_params(model, True)
    n_all_params = count_params(model, False)
    torch.save({"state_dict": best_overall["state"], "mode": args.mode,
                "lr": best_overall["lr"], "epoch": best_overall["epoch"]},
               ckpt_dir / f"{args.mode}_best.pt")

    ld = lambda ds: make_loader(ds, args.batch_size, False, args.seed, args.num_workers)
    y_tr, p_tr, _ = predict(model, ld(data["train_eval"]), device)
    y_va, p_va, _ = predict(model, ld(data["val"]), device)
    y_te, p_te, conf = predict(model, ld(data["test"]), device)
    m_tr, m_va, m_te = (compute_metrics(y_tr, p_tr), compute_metrics(y_va, p_va),
                        compute_metrics(y_te, p_te))

    name = "full_finetuning" if args.mode == "full" else "head_training"
    result = {
        "approach": name,
        "selected": {"lr": best_overall["lr"], "epoch": best_overall["epoch"],
                     "rule": "best validation accuracy over epochs and learning rates"},
        "hyperparameters": {"loss": "CrossEntropyLoss", "optimizer": "AdamW",
                            "weight_decay": args.weight_decay, "lr_schedule": "cosine annealing",
                            "batch_size": args.batch_size, "epochs": args.epochs,
                            "candidate_lrs": args.lrs, "seed": args.seed,
                            "deterministic": not args.no_deterministic},
        "trainable_parameters": n_train_params,
        "total_parameters": n_all_params,
        "trainable_layers": sorted({n.rsplit(".", 1)[0] for n in trainable_names(model)}),
        "runs": runs,
        "train_seconds_all_runs": total_time,
        "train_seconds_selected_run_until_best": best_overall["train_seconds_until_best"],
        "train": m_tr, "val": m_va, "test": m_te,
        "environment": environment_info(),
        "smoke_test": bool(args.smoke),
    }
    (out / f"{name}_metrics.json").write_text(json.dumps(result, indent=2))
    save_predictions(out / f"{name}_test_predictions.csv", y_te, p_te, conf)
    save_confusion(out / f"{name}_confusion.npy", y_te, p_te)
    print(json.dumps({k: result[k] for k in ("approach", "selected", "trainable_parameters",
                                             "train", "val", "test")}, indent=2))


if __name__ == "__main__":
    main()
