import argparse
from collections import deque

import cv2
import numpy as np
import torch

from models.physnet import PhysNet
from src import config
from src.face_detector import FaceDetector
from src.utils import (
    estimate_hr,
    extract_multi_rois_patches,
    make_patch_preview,
    normalize_signal,
)
from src.video import VideoCapture
from src.visualization import bvp_plot, draw_status


def apply_frame_diff(patches: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    diff = np.zeros_like(patches, dtype=np.float32)
    curr = patches[1:]
    prev = patches[:-1]
    diff[1:] = (curr - prev) / (np.abs(curr) + np.abs(prev) + eps)
    return diff


def prepare_window(patch_window: np.ndarray, use_frame_diff: bool) -> torch.Tensor:
    patches = patch_window.astype(np.float32)
    if patches.max() > 2.0:
        patches /= 255.0
    if use_frame_diff:
        patches = apply_frame_diff(patches)

    patches = np.transpose(patches, (0, 1, 4, 2, 3)).copy()
    return torch.from_numpy(patches).unsqueeze(0)


def load_model(model_path: str, device: torch.device) -> PhysNet:
    model = PhysNet().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def predict_bvp(
    model: PhysNet,
    patch_buffer: deque[np.ndarray],
    device: torch.device,
    use_frame_diff: bool,
) -> np.ndarray:
    patch_window = np.asarray(patch_buffer, dtype=np.float32)
    x = prepare_window(patch_window, use_frame_diff).to(device)
    bvp = model(x)[0].cpu().numpy().astype(np.float32)
    return normalize_signal(bvp)


def run_tester(args=None) -> None:
    parsed = parse_args(args)
    device = torch.device(parsed.device if (parsed.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = load_model(parsed.model_path, device)

    camera = VideoCapture()
    camera.start()

    detector = FaceDetector()
    fps = float(config.FPS_TARGET)
    patch_buffer: deque[np.ndarray] = deque(maxlen=config.CNN_WINDOW)
    bvp = np.array([], dtype=np.float32)

    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue

            display = frame.copy()
            landmarks = detector.get_landmarks(frame)
            heart_rate = None
            patch_preview = None

            if landmarks is None:
                patch_buffer.clear()
                bvp = np.array([], dtype=np.float32)
                status, color = "NO FACE", (0, 0, 255)
            else:
                detector.draw_landmarks(display, landmarks)
                patches = extract_multi_rois_patches(detector, frame, landmarks)
                patch_preview = make_patch_preview(patches)
                patch_buffer.append(patches)

                if len(patch_buffer) == config.CNN_WINDOW:
                    bvp = predict_bvp(model, patch_buffer, device, parsed.use_frame_diff)
                    heart_rate = estimate_hr(bvp, fps)

                status, color = (
                    f"DETECTED {len(patch_buffer)}/{config.CNN_WINDOW}",
                    (0, 255, 0),
                )

            draw_status(display, heart_rate, status, color)
            if bvp.size:
                plot = bvp_plot(bvp, display.shape[1], config.PLOT_H, heart_rate)
                display[-config.PLOT_H :, :] = plot

            cv2.imshow(config.WINDOW_NAME, display)
            if patch_preview is not None:
                cv2.imshow("ROI patches", patch_preview)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.stop()
        detector.close()
        cv2.destroyAllWindows()


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time PhysNet rPPG inference from webcam.")
    parser.add_argument("--model-path", default="results/best2/cnn.pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-frame-diff", action="store_true")
    return parser.parse_args(args)


if __name__ == "__main__":
    run_tester()
