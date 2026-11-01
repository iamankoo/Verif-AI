"""Training / fine-tuning of the two-stream EfficientNet-B0 + SRM model (architecture unchanged).

Key properties
  * source-/identity-disjoint splits come from the validated manifest (see prepare_faces.py);
  * per-epoch validation, ReduceLROnPlateau on validation loss, early stopping and best-checkpoint saving,
    both selected on VALIDATION data only (the best epoch maximises validation ROC-AUC; the 0.5 threshold is untouched);
  * BCEWithLogits (numerically stable) with pos_weight for class imbalance; AdamW; gradient clipping;
  * documented augmentation; seeds recorded; history.csv/json (+TensorBoard when installed);
  * checkpoint = {"state_dict", "meta"}; the original raw-state_dict checkpoints are still loadable.

CPU:  python -m training.train --manifest manifests/dff_faces_manifest.csv --init models/original.pth --name v2_cpu
GPU:  same command; CUDA is selected automatically (use --batch-size 32 --workers 4 on a GPU machine).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn

from verifai.model import TwoStreamNet, load_checkpoint_state, resolve_device

from .dataset import AugConfig, FaceCropDataset
from .metrics import compute_metrics


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ps, ys, loss_sum, n = [], [], 0.0, 0
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        z = model.logits(x).squeeze(1)
        loss_sum += bce(z, y).item(); n += len(y)
        ps.append(torch.sigmoid(z).cpu().numpy()); ys.append(y.cpu().numpy())
    return np.concatenate(ys).astype(int), np.concatenate(ps), loss_sum / max(n, 1)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="manifests/dff_faces_manifest.csv")
    ap.add_argument("--train-splits", nargs="+", default=["train"])
    ap.add_argument("--val-splits", nargs="+", default=["val"])
    ap.add_argument("--init", default="scripts/two_stream_model_epoch_10.pth", help="checkpoint to fine-tune from, or 'imagenet'")
    ap.add_argument("--name", default="v2")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=3, help="early-stopping patience (epochs without val ROC-AUC improvement)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=0, help="stop after this wall time (0 = no limit)")
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--limit-train", type=int, default=0, help="debug: use only N training rows")
    ap.add_argument("--limit-val", type=int, default=0, help="debug: use only N validation rows")
    ap.add_argument("--model-version", default="2.0")
    a = ap.parse_args()

    seed_everything(a.seed)
    device = resolve_device("auto")
    run = os.path.join(a.out, a.name); os.makedirs(run, exist_ok=True)
    aug = AugConfig(flip=0.0, jitter=False, blur_p=0.0, jpeg_p=0.0) if a.no_aug else AugConfig()
    train_ds = FaceCropDataset(a.manifest, a.train_splits, train=True, aug=aug)
    val_ds = FaceCropDataset(a.manifest, a.val_splits, train=False)
    if a.limit_train:
        train_ds.rows = train_ds.rows[: a.limit_train]
    if a.limit_val:                       # keep both classes represented
        val_ds.rows = val_ds.rows[: a.limit_val // 2] + val_ds.rows[-(a.limit_val // 2):]
    g = torch.Generator().manual_seed(a.seed)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, num_workers=a.workers, generator=g, drop_last=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=a.batch_size * 2, shuffle=False, num_workers=a.workers)

    model = TwoStreamNet(pretrained=(a.init == "imagenet"))
    if a.init != "imagenet":
        state, _ = load_checkpoint_state(a.init, torch.device("cpu"))
        model.load_state_dict(state, strict=True)
    model.to(device)

    ytr = np.array(train_ds.labels())
    n_real, n_fake = int((ytr == 0).sum()), int((ytr == 1).sum())
    pos_weight = torch.tensor([n_real / max(n_fake, 1)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=1)
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(os.path.join(run, "tb"))
    except Exception:
        tb = None

    cfg = dict(vars(a), device=str(device), n_train=len(train_ds), n_val=len(val_ds), train_real=n_real, train_fake=n_fake,
               pos_weight=float(pos_weight.item()), torch=str(torch.__version__), python=str(platform.python_version()),
               manifest_sha256=hashlib.sha256(open(a.manifest, "rb").read()).hexdigest(), git_commit=git_commit(),
               augmentation=None if a.no_aug else aug.__dict__, started=datetime.now(timezone.utc).isoformat())
    json.dump(cfg, open(os.path.join(run, "config.json"), "w"), indent=2, default=str)
    print(json.dumps(cfg, indent=1, default=str), flush=True)

    hist, best_auc, best_epoch, bad = [], -1.0, 0, 0
    t_start = time.time()
    fields = ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "val_auc", "val_f1", "val_fpr", "val_fnr", "epoch_seconds"]
    with open(os.path.join(run, "history.csv"), "w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=fields).writeheader()
    for epoch in range(1, a.epochs + 1):
        t0 = time.time(); model.train()
        tl, tc, tn = 0.0, 0, 0
        for step, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            z = model.logits(x).squeeze(1)
            loss = crit(z, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tl += loss.item() * len(y); tc += ((z > 0).float() == y).sum().item(); tn += len(y)
            if step % 20 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_dl)} loss {loss.item():.4f} ({time.time() - t_start:.0f}s)", flush=True)
            if a.max_minutes and (time.time() - t_start) / 60 > a.max_minutes:
                break
        yv, pv, vloss = predict(model, val_dl, device)
        m = compute_metrics(yv, pv)
        sched.step(vloss)
        row = dict(epoch=epoch, lr=opt.param_groups[0]["lr"], train_loss=tl / max(tn, 1), train_acc=tc / max(tn, 1), val_loss=vloss,
                   val_acc=m["accuracy"], val_auc=m["roc_auc"], val_f1=m["f1_fake"], val_fpr=m["false_positive_rate"],
                   val_fnr=m["false_negative_rate"], epoch_seconds=time.time() - t0)
        hist.append(row)
        with open(os.path.join(run, "history.csv"), "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=fields).writerow(row)
        json.dump(hist, open(os.path.join(run, "history.json"), "w"), indent=2)
        if tb:
            for k, v in row.items():
                if k != "epoch" and v is not None:
                    tb.add_scalar(k, v, epoch)
        print(f"== epoch {epoch}: " + " ".join(f"{k}={v:.4f}" for k, v in row.items() if isinstance(v, float)), flush=True)

        meta = dict(name=f"verifai-two-stream-{a.name}", version=a.model_version, epoch=epoch, seed=a.seed, created=datetime.now(timezone.utc).isoformat(),
                    trained_on=f"DeepFakeFace sample ({a.manifest}); splits {a.train_splits}", init=a.init,
                    training_notes=f"fine-tuned {a.epochs} max epochs, AdamW lr {a.lr}, BCEWithLogits pos_weight {float(pos_weight.item()):.2f}, augmentation {'off' if a.no_aug else 'on'}",
                    val_metrics={k: m[k] for k in ("accuracy", "roc_auc", "pr_auc", "f1_fake", "false_positive_rate", "false_negative_rate")},
                    metrics_note="Validation metrics only (identity-disjoint). See docs/MODEL_EVALUATION_REPORT.md for test results.",
                    config={k: v for k, v in cfg.items() if k not in ("started",)})
        torch.save({"state_dict": model.state_dict(), "meta": meta}, os.path.join(run, "last.pth"))
        if m["roc_auc"] is not None and m["roc_auc"] > best_auc:
            best_auc, best_epoch, bad = m["roc_auc"], epoch, 0
            torch.save({"state_dict": model.state_dict(), "meta": meta}, os.path.join(run, "best.pth"))
        else:
            bad += 1
            if bad >= a.patience:
                print(f"early stopping at epoch {epoch} (best epoch {best_epoch}, val AUC {best_auc:.4f})"); break
        if a.max_minutes and (time.time() - t_start) / 60 > a.max_minutes:
            print("time budget reached"); break
    json.dump(dict(best_epoch=best_epoch, best_val_auc=best_auc, epochs_run=len(hist), minutes=(time.time() - t_start) / 60),
              open(os.path.join(run, "summary.json"), "w"), indent=2)
    print("done. best epoch", best_epoch, "val AUC", best_auc)


if __name__ == "__main__":
    main()
