import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models.baseline import Baseline
from models.physnet import PhysNet
from models.loss import CNNLoss, ShiftLoss
from src import config
from src.dataset import (
    RPPGDataset,
    build_dataloaders,
    describe_dataset,
    discover_window_files,
    split_by_patient,
)


class NegPearson(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred - pred.mean(dim=1, keepdim=True)
        target = target - target.mean(dim=1, keepdim=True)
        num = (pred * target).sum(dim=1)
        denom = pred.norm(dim=1) * target.norm(dim=1) + 1e-8
        return (1.0 - num / denom).mean()


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fft_hr(signal: np.ndarray, fps: float, lo: float = 0.75, hi: float = 2.5) -> float:
    n = 2 ** int(np.ceil(np.log2(len(signal))))
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    power = np.abs(np.fft.rfft(signal, n=n)) ** 2
    mask = (freqs >= lo) & (freqs <= hi)
    if not mask.any():
        return float("nan")
    return float(freqs[mask][np.argmax(power[mask])] * 60.0)


def hr_metrics(preds: np.ndarray, targets: np.ndarray, fps: float) -> dict:
    pairs = []
    for p, t in zip(preds, targets):
        hp, ht = fft_hr(p, fps), fft_hr(t, fps)
        if np.isfinite(hp) and np.isfinite(ht):
            pairs.append((hp, ht))
    if not pairs:
        return {"mae": float("nan"), "rmse": float("nan"), "n": 0}
    pred_hr, true_hr = np.array(pairs).T
    err = pred_hr - true_hr
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "n": int(len(pairs)),
    }


def train_one_epoch(model, loader, optimizer, criterion, device, fps, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    total_samples = 0
    preds_all, targets_all = [], []
    pbar = tqdm(loader, desc=f"epoch {epoch}/{total_epochs} train", leave=False,
                bar_format="{l_bar}{bar:20}{r_bar}", dynamic_ncols=True)
    for patches, ppg in pbar:
        patches = patches.to(device, non_blocking=True)
        ppg = ppg.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(patches)
        loss = criterion(pred, ppg)
        loss.backward()
        optimizer.step()

        bs = patches.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        preds_all.append(pred.detach().cpu().numpy())
        targets_all.append(ppg.detach().cpu().numpy())
        pbar.set_postfix(loss=f"{total_loss / total_samples:.4f}")

    metrics = hr_metrics(np.concatenate(preds_all), np.concatenate(targets_all), fps)
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device, fps, epoch, total_epochs):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    preds_all, targets_all = [], []
    pbar = tqdm(loader, desc=f"epoch {epoch}/{total_epochs} val  ", leave=False,
                bar_format="{l_bar}{bar:20}{r_bar}", dynamic_ncols=True)
    for patches, ppg in pbar:
        patches = patches.to(device, non_blocking=True)
        ppg = ppg.to(device, non_blocking=True)

        pred = model(patches)
        loss = criterion(pred, ppg)

        bs = patches.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        preds_all.append(pred.cpu().numpy())
        targets_all.append(ppg.cpu().numpy())
        pbar.set_postfix(loss=f"{total_loss / total_samples:.4f}")

    preds = np.concatenate(preds_all)
    targets = np.concatenate(targets_all)
    metrics = hr_metrics(preds, targets, fps)
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics, preds, targets


def save_plots(history: list[dict], best_pred: np.ndarray, best_target: np.ndarray,
               fps: float, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train", marker="o")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val", marker="o")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("Neg Pearson loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [h["train_hr_mae"] for h in history], label="train", marker="o")
    axes[1].plot(epochs, [h["val_hr_mae"] for h in history], label="val", marker="o")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("HR MAE, BPM")
    axes[1].set_title("HR MAE"); axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, [h["val_hr_rmse"] for h in history], color="C2", marker="o")
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("HR RMSE, BPM")
    axes[2].set_title("Validation HR RMSE"); axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=120)
    plt.close(fig)

    n_show = min(4, len(best_pred))
    fig, axes = plt.subplots(n_show, 1, figsize=(12, 2.5 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]
    t = np.arange(best_pred.shape[1]) / fps
    for i in range(n_show):
        axes[i].plot(t, best_target[i], label="target PPG", color="C0", lw=1.4)
        axes[i].plot(t, best_pred[i], label="predicted BVP", color="C3", lw=1.4, alpha=0.85)
        hp = fft_hr(best_pred[i], fps); ht = fft_hr(best_target[i], fps)
        axes[i].set_title(f"sample {i}  |  pred HR={hp:.1f}  target HR={ht:.1f} BPM")
        axes[i].grid(alpha=0.3)
        if i == 0:
            axes[i].legend(loc="upper right")
    axes[-1].set_xlabel("time, s")
    fig.tight_layout()
    fig.savefig(out_dir / "best_predictions.png", dpi=120)
    plt.close(fig)

    pred_hr = np.array([fft_hr(p, fps) for p in best_pred])
    true_hr = np.array([fft_hr(t, fps) for t in best_target])
    valid = np.isfinite(pred_hr) & np.isfinite(true_hr)
    pred_hr, true_hr = pred_hr[valid], true_hr[valid]
    if len(pred_hr) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    lo = float(min(pred_hr.min(), true_hr.min())) - 5
    hi = float(max(pred_hr.max(), true_hr.max())) + 5
    axes[0].scatter(true_hr, pred_hr, alpha=0.5, s=14)
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[0].set_xlabel("target HR, BPM"); axes[0].set_ylabel("predicted HR, BPM")
    axes[0].set_title(f"Validation HR scatter (n={len(pred_hr)})")
    axes[0].grid(alpha=0.3); axes[0].set_xlim(lo, hi); axes[0].set_ylim(lo, hi)

    mean_hr = (pred_hr + true_hr) / 2
    diff_hr = pred_hr - true_hr
    bias = float(diff_hr.mean()); sd = float(diff_hr.std())
    axes[1].scatter(mean_hr, diff_hr, alpha=0.5, s=14)
    axes[1].axhline(bias, color="C3", lw=1.2, label=f"bias={bias:+.2f}")
    axes[1].axhline(bias + 1.96 * sd, color="C3", ls="--", lw=1, label=f"±1.96σ ({1.96 * sd:.2f})")
    axes[1].axhline(bias - 1.96 * sd, color="C3", ls="--", lw=1)
    axes[1].set_xlabel("mean HR, BPM"); axes[1].set_ylabel("pred − target, BPM")
    axes[1].set_title("Bland-Altman"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "hr_scatter.png", dpi=120)
    plt.close(fig)


def run(args=None) -> None:
    args = parse_args(args)
    fix_seed(args.seed)

    out_dir = Path(args.output) / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    fps = float(config.FPS_TARGET)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"[1/5] Discover windows in {args.data_dir}")
    files = discover_window_files(args.data_dir)
    if not files:
        return
    describe_dataset(files)
    dataset = RPPGDataset(files, use_frame_diff=args.use_frame_diff)

    print(f"\n[2/5] Subject-stratified split")
    split = split_by_patient(files, val_split=args.val_split, seed=args.seed)
    if args.max_train_patients or args.max_val_patients:
        rng = random.Random(args.seed)
        train_p = sorted(split.train_patients)
        val_p = sorted(split.val_patients)
        rng.shuffle(train_p); rng.shuffle(val_p)
        if args.max_train_patients:
            train_p = train_p[:args.max_train_patients]
        if args.max_val_patients:
            val_p = val_p[:args.max_val_patients]
        keep_train = set(train_p); keep_val = set(val_p)
        from src.dataset import DatasetSplit, get_patient_id
        new_train = [i for i in split.train_indices if get_patient_id(files[i]) in keep_train]
        new_val = [i for i in split.val_indices if get_patient_id(files[i]) in keep_val]
        split = DatasetSplit(new_train, new_val, keep_train, keep_val)
    print(f"train: {len(split.train_indices):>6} windows | {len(split.train_patients)} patients")
    print(f"val:   {len(split.val_indices):>6} windows | {len(split.val_patients)} patients")

    print(f"\n[3/5] Build dataloaders (batch_size={args.batch_size}, workers={args.num_workers})")
    train_loader, val_loader = build_dataloaders(
        dataset, split,
        train_config={
            "BATCH_SIZE": args.batch_size,
            "NUM_WORKERS": args.num_workers,
        },
    )
    print(f"train batches: {len(train_loader)} | val batches: {len(val_loader)}")

    print(f"\n[4/5] Build model")
    if args.model == "baseline":
        model = Baseline().to(device)
    elif args.model == "physnet":
        model = PhysNet().to(device)
    else:
        raise ValueError(f"unknown --model {args.model!r}")

    if args.pretrained:
        checkpoint_path = Path(args.pretrained)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"loaded pretrained weights: {checkpoint_path}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.loss == "negpearson":
        criterion = NegPearson()
    elif args.loss == "cnn":
        criterion = CNNLoss(spectral_alpha=args.spectral_alpha)
    elif args.loss == "shiftloss":
        criterion = ShiftLoss(max_shift_sec=args.max_shift_sec, fps=fps)
    else:
        raise ValueError(f"unknown --loss {args.loss!r}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device: {device} | model: {args.model} | loss: {args.loss} | params: {n_params:,}")
    print(f"output: {out_dir}")

    print(f"\n[5/5] Train ({args.epochs} epochs)")
    print("-" * 100)
    print(f"{'epoch':>5} {'train_loss':>11} {'val_loss':>10} "
          f"{'train_mae':>10} {'val_mae':>9} {'val_rmse':>10} {'lr':>10} {'time':>7}")
    print("-" * 100)

    history: list[dict] = []
    best_mae = float("inf")
    best_pred = best_target = None
    epochs_without_improvement = 0
    stopped_early = False
    stop_reason = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, optimizer, criterion, device, fps,
                                  epoch, args.epochs)
        val_m, val_pred, val_target = eval_one_epoch(model, val_loader, criterion, device, fps,
                                                     epoch, args.epochs)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        record = {
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "val_loss": val_m["loss"],
            "train_hr_mae": train_m["mae"],
            "val_hr_mae": val_m["mae"],
            "val_hr_rmse": val_m["rmse"],
            "lr": lr_now,
            "time": elapsed,
        }
        history.append(record)

        marker = " "
        improved = val_m["mae"] < best_mae - args.early_stopping_min_delta
        if improved:
            best_mae = val_m["mae"]
            best_pred, best_target = val_pred, val_target
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_dir / config.CNN_MODEL_PATH)
            marker = "*"
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(f"{marker}{epoch:4d} {train_m['loss']:11.4f} {val_m['loss']:10.4f} "
              f"{train_m['mae']:10.2f} {val_m['mae']:9.2f} {val_m['rmse']:10.2f} "
              f"{lr_now:10.2e} {elapsed:6.1f}s")

        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "history.json").open("w") as f:
            json.dump(history, f, indent=2)

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            stopped_early = True
            stop_reason = (
                f"val_hr_mae plateau: no improvement >= {args.early_stopping_min_delta:.4f} "
                f"for {args.early_stopping_patience} epochs"
            )
            print(f"early stopping: {stop_reason}")
            break

    print("-" * 100)
    print(f"best val HR MAE: {best_mae:.2f} BPM")

    summary = {
        "best_val_hr_mae": best_mae,
        "n_epochs": args.epochs,
        "n_epochs_completed": len(history),
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "fps": fps,
        "window": config.CNN_WINDOW,
        "window_sec": config.CNN_WINDOW / fps,
        "n_train_windows": len(split.train_indices),
        "n_val_windows": len(split.val_indices),
        "n_train_patients": len(split.train_patients),
        "n_val_patients": len(split.val_patients),
        "model_params": n_params,
        "args": vars(args),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    if best_pred is not None:
        save_plots(history, best_pred, best_target, fps, out_dir)
        print(f"plots and summary saved: {out_dir}")


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Baseline rPPG model on patch windows.")
    parser.add_argument("--data-dir", default="data/mcd_rppg_windows")
    parser.add_argument("--output", default="results")
    parser.add_argument("--epochs", type=int, default=config.CNN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.CNN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.CNN_LR)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-train-patients", type=int, default=None,
                        help="cap train set to N patients (for fast CPU sanity runs)")
    parser.add_argument("--max-val-patients", type=int, default=None)
    parser.add_argument("--model", choices=["baseline", "physnet"], default="baseline")
    parser.add_argument("--pretrained", default=None,
                        help="path to model weights used as the training start point")
    parser.add_argument("--loss", choices=["negpearson", "cnn", "shiftloss"], default="negpearson")
    parser.add_argument("--spectral-alpha", type=float, default=0.1,
                        help="weight of spectral term in CNNLoss (only used when --loss cnn)")
    parser.add_argument("--max-shift-sec", type=float, default=0.33,
                        help="max temporal shift in seconds for ShiftLoss (only used when --loss shiftloss)")
    parser.add_argument("--use-frame-diff", action="store_true",
                        help="apply normalized frame difference preprocessing (Ho et al.)")
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="stop when val HR MAE has not improved for N epochs; 0 disables")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0,
                        help="minimum val HR MAE improvement in BPM to reset early stopping")
    return parser.parse_args(args)


if __name__ == "__main__":
    run()
