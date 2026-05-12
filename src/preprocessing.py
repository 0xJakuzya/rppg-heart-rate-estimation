import argparse
import csv
import math
from pathlib import Path
import cv2
import numpy as np
from src import config
from src.face_detector import FaceDetector
from src.utils import (
    bandpass_filter,
    detrend,
    extract_multi_rois_patches,
    normalize_patch_window,
    normalize_signal,
)


REPORT_FIELDS = [
    "patient_id",
    "stem",
    "status",
    "reason",
    "reported_frames",
    "decoded_frames",
    "sampled_frames",
    "ppg_frames",
    "coverage",
    "missing_ratio",
    "windows",
]


def iter_rows(dataset_root: Path, camera: str, step: str):
    with (dataset_root / "db.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["camera"] != camera:
                continue
            if step != "all" and row["step"] != step:
                continue
            yield row

def fill_missing_patches(patches: np.ndarray) -> np.ndarray | None:
    flat = patches.reshape(patches.shape[0], -1)
    valid = np.isfinite(flat).all(axis=1)
    if not valid.any():
        return None
    idx = np.arange(len(flat))
    filled = flat.copy()
    for channel in range(flat.shape[1]):
        filled[:, channel] = np.interp(idx, idx[valid], flat[valid, channel])
    return filled.reshape(patches.shape).astype(np.float32)

def extract_patch_sequence(
    video_path: Path,
    detector: FaceDetector,
    frame_step: int = 1,
) -> tuple[np.ndarray, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    patch_values: list[np.ndarray] = []
    missing = 0
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % frame_step != 0:
            frame_index += 1
            continue
        landmarks = detector.get_landmarks(frame)
        if landmarks is None:
            patch_values.append(
                np.full(
                    (config.MULTI_ROI_COUNT, config.ROI_PATCH_SIZE, config.ROI_PATCH_SIZE, 3),
                    np.nan,
                    dtype=np.float32,
                )
            )
            missing += 1
            frame_index += 1
            continue
        patches = extract_multi_rois_patches(detector, frame, landmarks, patch_size=config.ROI_PATCH_SIZE)
        if len(patches) != config.MULTI_ROI_COUNT:
            patch_values.append(
                np.full(
                    (config.MULTI_ROI_COUNT, config.ROI_PATCH_SIZE, config.ROI_PATCH_SIZE, 3),
                    np.nan,
                    dtype=np.float32,
                )
            )
            missing += 1
            frame_index += 1
            continue
        else:
            patch_values.append(patches.astype(np.float32))
        frame_index += 1
    cap.release()
    return np.asarray(patch_values, dtype=np.float32), missing, frame_index, reported_frames


def expected_sampled_frames(total_frames: int, frame_step: int) -> int:
    if total_frames <= 0:
        return 0
    return math.ceil(total_frames / frame_step)


def append_report_row(report_path: Path, row: dict[str, object]) -> None:
    needs_header = not report_path.exists()
    with report_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def make_report_row(
    patient_id: str,
    stem: str,
    status: str,
    reason: str,
    reported_frames: int,
    decoded_frames: int,
    sampled_frames: int,
    ppg_frames: int,
    coverage: float,
    missing_ratio: float,
    windows: int,
) -> dict[str, object]:
    return {
        "patient_id": patient_id,
        "stem": stem,
        "status": status,
        "reason": reason,
        "reported_frames": reported_frames,
        "decoded_frames": decoded_frames,
        "sampled_frames": sampled_frames,
        "ppg_frames": ppg_frames,
        "coverage": f"{coverage:.4f}",
        "missing_ratio": f"{missing_ratio:.4f}",
        "windows": windows,
    }


def save_windows(patches: np.ndarray, ppg: np.ndarray, output_dir:
                    Path, stem: str, window: int, stride: int) -> int:
    count = 0
    n = min(len(patches), len(ppg))
    patches = patches[:n]
    ppg = ppg[:n]
    output_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, n - window + 1, stride):
        end = start + window
        np.savez_compressed(
            output_dir / f"{stem}_{start:05d}.npz",
            patches=normalize_patch_window(patches[start:end]),
            ppg=normalize_signal(ppg[start:end]),
        )
        count += 1
    return count


def count_windows(n_frames: int, window: int, stride: int) -> int:
    if n_frames < window:
        return 0
    return ((n_frames - window) // stride) + 1


def filter_target_ppg(ppg: np.ndarray, fps: float) -> np.ndarray:
    filtered = detrend(ppg)
    if len(filtered) >= max(int(fps * 2), 3):
        filtered = bandpass_filter(filtered, fps, config.CHEBY_LO, config.CHEBY_HI)
    return normalize_signal(filtered)


def patient_id_allowed(patient_id: str, min_patient_id: int | None) -> bool:
    if min_patient_id is None:
        return True
    try:
        return int(patient_id) >= min_patient_id
    except ValueError:
        return False

def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_path) if args.report_path else output_dir / "preprocessing_report.csv"
    if report_path.exists():
        report_path.unlink()
    effective_fps = args.video_fps / args.frame_step

    detector = FaceDetector()
    processed = 0
    total_windows = 0
    skipped = 0
    warned = 0
    filtered = 0

    try:
        for row in iter_rows(dataset_root, args.camera, args.step):
            if args.max_videos is not None and processed >= args.max_videos:
                break

            stem = Path(row["video"]).stem
            patient_id = stem.split("_", 1)[0]
            if not patient_id_allowed(patient_id, args.min_patient_id):
                filtered += 1
                continue

            patient_dir = output_dir / patient_id
            patches, missing, decoded_frames, reported_frames = extract_patch_sequence(
                dataset_root / row["video"],
                detector,
                frame_step=args.frame_step,
            )
            if len(patches) == 0:
                print(f"skip {stem}: no frames")
                skipped += 1
                append_report_row(
                    report_path,
                    make_report_row(
                        patient_id=patient_id,
                        stem=stem,
                        status="skip",
                        reason="no_frames",
                        reported_frames=reported_frames,
                        decoded_frames=decoded_frames,
                        sampled_frames=0,
                        ppg_frames=0,
                        coverage=0.0,
                        missing_ratio=1.0,
                        windows=0,
                    ),
                )
                continue

            missing_ratio = missing / len(patches)
            if missing_ratio > args.max_missing:
                print(f"skip {stem}: missing face ratio {missing_ratio:.2f}")
                skipped += 1
                append_report_row(
                    report_path,
                    make_report_row(
                        patient_id=patient_id,
                        stem=stem,
                        status="skip",
                        reason="missing_face_ratio",
                        reported_frames=reported_frames,
                        decoded_frames=decoded_frames,
                        sampled_frames=len(patches),
                        ppg_frames=0,
                        coverage=0.0,
                        missing_ratio=missing_ratio,
                        windows=0,
                    ),
                )
                continue

            patches = fill_missing_patches(patches)
            if patches is None:
                print(f"skip {stem}: no valid face frames")
                skipped += 1
                append_report_row(
                    report_path,
                    make_report_row(
                        patient_id=patient_id,
                        stem=stem,
                        status="skip",
                        reason="no_valid_face_frames",
                        reported_frames=reported_frames,
                        decoded_frames=decoded_frames,
                        sampled_frames=len(patches),
                        ppg_frames=0,
                        coverage=0.0,
                        missing_ratio=missing_ratio,
                        windows=0,
                    ),
                )
                continue

            ppg = np.loadtxt(dataset_root / row["ppg_sync"], usecols=0).astype(np.float32)
            if args.frame_step > 1:
                ppg = ppg[::args.frame_step]
            ppg = filter_target_ppg(ppg, effective_fps)

            coverage = len(patches) / len(ppg) if len(ppg) > 0 else 0.0
            if coverage < args.min_coverage:
                print(
                    f"skip {stem}: video coverage {coverage:.2f} "
                    f"({len(patches)}/{len(ppg)} sampled frames)"
                )
                skipped += 1
                append_report_row(
                    report_path,
                    make_report_row(
                        patient_id=patient_id,
                        stem=stem,
                        status="skip",
                        reason="low_video_coverage",
                        reported_frames=reported_frames,
                        decoded_frames=decoded_frames,
                        sampled_frames=len(patches),
                        ppg_frames=len(ppg),
                        coverage=coverage,
                        missing_ratio=missing_ratio,
                        windows=0,
                    ),
                )
                continue

            sampled_reported_frames = expected_sampled_frames(reported_frames, args.frame_step)
            status = "ok"
            reason = ""
            if sampled_reported_frames and len(patches) < sampled_reported_frames:
                status = "warn"
                reason = "decoder_stopped_early"
                warned += 1
                print(
                    f"warn {stem}: decoder stopped early "
                    f"({len(patches)}/{sampled_reported_frames} sampled frames, raw {decoded_frames}/{reported_frames})"
                )

            n_frames = min(len(patches), len(ppg))
            expected_windows = count_windows(n_frames, args.window, args.stride)
            if expected_windows < args.min_windows:
                print(f"skip {stem}: only {expected_windows} windows")
                skipped += 1
                append_report_row(
                    report_path,
                    make_report_row(
                        patient_id=patient_id,
                        stem=stem,
                        status="skip",
                        reason="too_few_windows",
                        reported_frames=reported_frames,
                        decoded_frames=decoded_frames,
                        sampled_frames=len(patches),
                        ppg_frames=len(ppg),
                        coverage=coverage,
                        missing_ratio=missing_ratio,
                        windows=expected_windows,
                    ),
                )
                continue

            written = save_windows(patches, ppg, patient_dir, stem, args.window, args.stride)
            processed += 1
            total_windows += written
            print(f"{stem}: {written} windows")
            append_report_row(
                report_path,
                make_report_row(
                    patient_id=patient_id,
                    stem=stem,
                    status=status,
                    reason=reason,
                    reported_frames=reported_frames,
                    decoded_frames=decoded_frames,
                    sampled_frames=len(patches),
                    ppg_frames=len(ppg),
                    coverage=coverage,
                    missing_ratio=missing_ratio,
                    windows=written,
                ),
            )
    finally:
        detector.close()

    print(f"processed videos: {processed}")
    print(f"skipped videos: {skipped}")
    print(f"filtered videos: {filtered}")
    print(f"warned videos: {warned}")
    print(f"saved windows: {total_windows}")
    print(f"output dir: {output_dir}")
    print(f"report: {report_path}")

def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare multi-ROI MCD-rPPG windows.")
    parser.add_argument("--dataset-root", default="D:/mcd_rppg")
    parser.add_argument("--output-dir", default="data/mcd_rppg_windows")
    parser.add_argument("--camera", default="FullHDwebcam")
    parser.add_argument("--step", choices=["before", "after", "all"], default="all")
    parser.add_argument("--window", type=int, default=config.CNN_WINDOW)
    parser.add_argument("--stride", type=int, default=150)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--min-patient-id", type=int, default=None,
                        help="Process only videos with numeric patient ID >= this value.")
    parser.add_argument("--max-missing", type=float, default=0.2)
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--min-windows", type=int, default=1)
    parser.add_argument("--report-path", default=None)
    return parser.parse_args(args)

if __name__ == "__main__":
    main()
