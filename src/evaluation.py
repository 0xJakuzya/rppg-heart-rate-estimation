import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.baseline import Baseline
from models.physnet import PhysNet
from src import config
from src.dataset import RPPGDataset, discover_window_files, get_patient_id


def parse_window_stem(file: str) -> tuple[str, int]:
    stem = Path(file).stem
    video_id, start_text = stem.rsplit("_", 1)
    try:
        start_frame = int(start_text)
    except ValueError as exc:
        raise ValueError(f"expected window filename ending with _00000 style frame index: {stem}") from exc
    return video_id, start_frame


def reconstruct_video_signal(segments: list[tuple[int, np.ndarray]]) -> np.ndarray:
    if not segments:
        return np.array([], dtype=np.float32)

    max_frame = max(start + len(signal) for start, signal in segments)
    summed = np.zeros(max_frame, dtype=np.float64)
    counts = np.zeros(max_frame, dtype=np.float64)

    for start, signal in segments:
        end = start + len(signal)
        summed[start:end] += signal
        counts[start:end] += 1.0

    covered = counts > 0
    reconstructed = np.zeros_like(summed)
    reconstructed[covered] = summed[covered] / counts[covered]
    return reconstructed[covered].astype(np.float32)


def spectral_confidence(signal: np.ndarray, fps: float, eps: float = 1e-12) -> dict[str, float]:
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) < 2:
        return {
            "hr_bpm": float("nan"),
            "peak_ratio": float("nan"),
            "peak_power_fraction": float("nan"),
            "spectral_entropy": float("nan"),
            "confidence": 0.0,
        }

    signal = signal - signal.mean()
    n = 2 ** int(np.ceil(np.log2(len(signal))))
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    power = np.abs(np.fft.rfft(signal, n=n)) ** 2
    mask = (freqs >= config.HR_LO_HZ) & (freqs <= config.HR_HI_HZ)
    if not mask.any():
        return {
            "hr_bpm": float("nan"),
            "peak_ratio": float("nan"),
            "peak_power_fraction": float("nan"),
            "spectral_entropy": float("nan"),
            "confidence": 0.0,
        }

    band_freqs = freqs[mask]
    band_power = power[mask]
    order = np.argsort(band_power)[::-1]
    top_power = float(band_power[order[0]])
    second_power = float(band_power[order[1]]) if len(order) > 1 else 0.0
    total_power = float(band_power.sum())

    probabilities = band_power / (total_power + eps)
    entropy = -float(np.sum(probabilities * np.log(probabilities + eps)))
    if len(probabilities) > 1:
        entropy /= float(np.log(len(probabilities)))

    peak_ratio = top_power / (second_power + eps)
    peak_power_fraction = top_power / (total_power + eps)
    entropy_confidence = 1.0 - entropy
    ratio_score = float(np.clip((peak_ratio - 1.0) / 1.0, 0.0, 1.0))
    fraction_score = float(np.clip(peak_power_fraction / 0.35, 0.0, 1.0))
    confidence = 0.5 * ratio_score + 0.3 * fraction_score + 0.2 * entropy_confidence

    return {
        "hr_bpm": float(band_freqs[order[0]] * 60.0),
        "peak_ratio": peak_ratio,
        "peak_power_fraction": peak_power_fraction,
        "spectral_entropy": entropy,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
    }


def summarize_confidence_thresholds(
    video_rows: list[dict[str, float | int | str]],
    thresholds: list[float],
) -> dict[str, dict[str, float | int]]:
    summary = {}
    total = len(video_rows)
    for threshold in thresholds:
        kept = [
            row
            for row in video_rows
            if float(row["pred_confidence"]) >= threshold
        ]
        key = f"{threshold:.2f}"
        if not kept:
            summary[key] = {
                "n_videos": 0,
                "coverage": 0.0,
                "video_hr_mae": float("nan"),
                "video_hr_rmse": float("nan"),
                "video_hr_bias": float("nan"),
                "n_over_5_bpm": 0,
                "over_5_bpm_rate": float("nan"),
            }
            continue

        errors = np.array([float(row["error_bpm"]) for row in kept], dtype=float)
        abs_errors = np.abs(errors)
        n_over_5 = int(np.sum(abs_errors > 5.0))
        summary[key] = {
            "n_videos": len(kept),
            "coverage": float(len(kept) / total),
            "video_hr_mae": float(np.mean(abs_errors)),
            "video_hr_rmse": float(np.sqrt(np.mean(errors ** 2))),
            "video_hr_bias": float(np.mean(errors)),
            "n_over_5_bpm": n_over_5,
            "over_5_bpm_rate": float(n_over_5 / len(kept)),
        }
    return summary


def patient_ids_from_dirs(data_dirs: list[str]) -> set[str]:
    patient_ids: set[str] = set()
    for data_dir in data_dirs:
        for file in discover_window_files(data_dir):
            patient_ids.add(get_patient_id(file))
    return patient_ids


def select_patient_files(
    files: list[str],
    max_patients: int | None,
    exclude_patient_ids: set[str],
    start_after_patient: str | None,
) -> tuple[list[str], list[str]]:
    patient_to_files: dict[str, list[str]] = {}
    for file in files:
        patient_id = get_patient_id(file)
        if patient_id in exclude_patient_ids:
            continue
        patient_to_files.setdefault(patient_id, []).append(file)

    selected_patients = sorted(patient_to_files)
    if start_after_patient is not None:
        selected_patients = [
            patient_id
            for patient_id in selected_patients
            if patient_id > start_after_patient
        ]
    if max_patients is not None:
        selected_patients = selected_patients[:max_patients]

    selected_files = [
        file
        for patient_id in selected_patients
        for file in sorted(patient_to_files[patient_id])
    ]
    return selected_files, selected_patients


def load_model(model_name: str, model_path: str, device: torch.device) -> torch.nn.Module:
    if model_name == "baseline":
        model = Baseline()
    elif model_name == "physnet":
        model = PhysNet()
    else:
        raise ValueError(f"unknown model: {model_name}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run(args=None) -> None:
    parsed = parse_args(args)
    device = torch.device(parsed.device if (parsed.device == "cpu" or torch.cuda.is_available()) else "cpu")
    fps = float(config.FPS_TARGET)

    files = discover_window_files(parsed.data_dir)
    exclude_patient_ids = patient_ids_from_dirs(parsed.exclude_data_dir)
    selected_files, selected_patients = select_patient_files(
        files,
        parsed.max_patients,
        exclude_patient_ids,
        parsed.start_after_patient,
    )
    if not selected_files:
        raise ValueError("no files selected for evaluation")

    out_dir = Path(parsed.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(parsed.model, parsed.model_path, device)
    dataset = RPPGDataset(selected_files, use_frame_diff=parsed.use_frame_diff)
    loader = DataLoader(
        Subset(dataset, range(len(selected_files))),
        batch_size=parsed.batch_size,
        shuffle=False,
        num_workers=parsed.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=parsed.num_workers > 0,
    )

    rows = []
    abs_errors = []
    errors = []
    video_segments = defaultdict(lambda: {"pred": [], "target": [], "window_rows": []})
    pbar = tqdm(loader, desc="eval", dynamic_ncols=True)
    offset = 0
    for patches, ppg in pbar:
        patches = patches.to(device, non_blocking=True)
        pred = model(patches).cpu().numpy()
        target = ppg.numpy()

        for i in range(pred.shape[0]):
            file = selected_files[offset + i]
            video_id, start_frame = parse_window_stem(file)
            pred_metrics = spectral_confidence(pred[i], fps)
            true_metrics = spectral_confidence(target[i], fps)
            pred_hr = pred_metrics["hr_bpm"]
            true_hr = true_metrics["hr_bpm"]
            error = pred_hr - true_hr
            abs_error = abs(error)
            row = {
                "patient_id": get_patient_id(file),
                "file": Path(file).stem,
                "video": video_id,
                "start_frame": start_frame,
                "pred_hr_bpm": pred_hr,
                "true_hr_bpm": true_hr,
                "error_bpm": error,
                "abs_error_bpm": abs_error,
                "pred_peak_ratio": pred_metrics["peak_ratio"],
                "pred_peak_power_fraction": pred_metrics["peak_power_fraction"],
                "pred_spectral_entropy": pred_metrics["spectral_entropy"],
                "pred_confidence": pred_metrics["confidence"],
                "true_peak_ratio": true_metrics["peak_ratio"],
                "true_peak_power_fraction": true_metrics["peak_power_fraction"],
                "true_spectral_entropy": true_metrics["spectral_entropy"],
            }
            rows.append(row)
            video_segments[video_id]["pred"].append((start_frame, pred[i].copy()))
            video_segments[video_id]["target"].append((start_frame, target[i].copy()))
            video_segments[video_id]["window_rows"].append(row)
            errors.append(error)
            abs_errors.append(abs_error)
        offset += pred.shape[0]
        pbar.set_postfix(mae=f"{float(np.mean(abs_errors)):.2f}")

    csv_path = out_dir / "predictions.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    video_rows = []
    for video_id in sorted(video_segments):
        segments = video_segments[video_id]
        pred_signal = reconstruct_video_signal(segments["pred"])
        target_signal = reconstruct_video_signal(segments["target"])
        pred_metrics = spectral_confidence(pred_signal, fps)
        true_metrics = spectral_confidence(target_signal, fps)
        pred_hr = pred_metrics["hr_bpm"]
        true_hr = true_metrics["hr_bpm"]
        error = pred_hr - true_hr
        abs_error = abs(error)
        window_rows = segments["window_rows"]
        video_rows.append({
            "patient_id": window_rows[0]["patient_id"],
            "video": video_id,
            "n_windows": len(window_rows),
            "covered_frames": len(pred_signal),
            "pred_hr_bpm": pred_hr,
            "true_hr_bpm": true_hr,
            "error_bpm": error,
            "abs_error_bpm": abs_error,
            "pred_peak_ratio": pred_metrics["peak_ratio"],
            "pred_peak_power_fraction": pred_metrics["peak_power_fraction"],
            "pred_spectral_entropy": pred_metrics["spectral_entropy"],
            "pred_confidence": pred_metrics["confidence"],
            "true_peak_ratio": true_metrics["peak_ratio"],
            "true_peak_power_fraction": true_metrics["peak_power_fraction"],
            "true_spectral_entropy": true_metrics["spectral_entropy"],
            "window_mean_pred_confidence": float(np.mean([row["pred_confidence"] for row in window_rows])),
            "window_mean_pred_hr_bpm": float(np.mean([row["pred_hr_bpm"] for row in window_rows])),
            "window_mean_true_hr_bpm": float(np.mean([row["true_hr_bpm"] for row in window_rows])),
            "window_mean_abs_error_bpm": float(np.mean([row["abs_error_bpm"] for row in window_rows])),
            "window_median_pred_hr_bpm": float(np.median([row["pred_hr_bpm"] for row in window_rows])),
            "window_median_true_hr_bpm": float(np.median([row["true_hr_bpm"] for row in window_rows])),
        })

    video_csv_path = out_dir / "video_predictions.csv"
    with video_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(video_rows[0]))
        writer.writeheader()
        writer.writerows(video_rows)

    errors_np = np.array(errors, dtype=float)
    abs_errors_np = np.array(abs_errors, dtype=float)
    video_errors_np = np.array([row["error_bpm"] for row in video_rows], dtype=float)
    video_abs_errors_np = np.array([row["abs_error_bpm"] for row in video_rows], dtype=float)
    patient_mae = {}
    patient_video_mae = {}
    for patient_id in selected_patients:
        patient_abs = [row["abs_error_bpm"] for row in rows if row["patient_id"] == patient_id]
        patient_mae[patient_id] = float(np.mean(patient_abs))
        patient_video_abs = [
            row["abs_error_bpm"]
            for row in video_rows
            if row["patient_id"] == patient_id
        ]
        patient_video_mae[patient_id] = float(np.mean(patient_video_abs))

    summary = {
        "model_path": parsed.model_path,
        "data_dir": parsed.data_dir,
        "exclude_data_dir": parsed.exclude_data_dir,
        "n_excluded_patients": len(exclude_patient_ids),
        "n_patients": len(selected_patients),
        "n_windows": len(rows),
        "n_videos": len(video_rows),
        "n_windows_over_5_bpm": int(np.sum(abs_errors_np > 5.0)),
        "windows_over_5_bpm_rate": float(np.mean(abs_errors_np > 5.0)),
        "n_videos_over_5_bpm": int(np.sum(video_abs_errors_np > 5.0)),
        "videos_over_5_bpm_rate": float(np.mean(video_abs_errors_np > 5.0)),
        "hr_mae": float(np.mean(abs_errors_np)),
        "hr_rmse": float(np.sqrt(np.mean(errors_np ** 2))),
        "hr_bias": float(np.mean(errors_np)),
        "median_abs_error": float(np.median(abs_errors_np)),
        "p90_abs_error": float(np.percentile(abs_errors_np, 90)),
        "video_hr_mae": float(np.mean(video_abs_errors_np)),
        "video_hr_rmse": float(np.sqrt(np.mean(video_errors_np ** 2))),
        "video_hr_bias": float(np.mean(video_errors_np)),
        "video_median_abs_error": float(np.median(video_abs_errors_np)),
        "video_p90_abs_error": float(np.percentile(video_abs_errors_np, 90)),
        "video_mean_pred_confidence": float(np.mean([row["pred_confidence"] for row in video_rows])),
        "video_confidence_thresholds": summarize_confidence_thresholds(
            video_rows,
            parsed.confidence_thresholds,
        ),
        "selected_patients": selected_patients,
        "patient_mae": patient_mae,
        "patient_video_mae": patient_video_mae,
        "args": vars(parsed),
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"saved: {csv_path}")
    print(f"saved: {video_csv_path}")
    print(f"summary: {out_dir / 'summary.json'}")
    print(f"HR MAE={summary['hr_mae']:.2f} BPM | RMSE={summary['hr_rmse']:.2f} BPM")
    print(f"Video HR MAE={summary['video_hr_mae']:.2f} BPM | RMSE={summary['video_hr_rmse']:.2f} BPM")
    print(
        f"Video >5 BPM={summary['n_videos_over_5_bpm']}/{summary['n_videos']} "
        f"({summary['videos_over_5_bpm_rate'] * 100:.1f}%)"
    )


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained rPPG model on .npz windows.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", choices=["baseline", "physnet"], default="physnet")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--start-after-patient", default=None)
    parser.add_argument("--exclude-data-dir", action="append", default=[])
    parser.add_argument("--use-frame-diff", action="store_true")
    parser.add_argument(
        "--confidence-thresholds",
        type=float,
        nargs="*",
        default=[0.45, 0.55, 0.65, 0.75],
        help="video confidence thresholds to summarize in summary.json",
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    run()
