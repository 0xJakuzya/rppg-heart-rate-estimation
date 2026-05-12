import cv2
import numpy as np
from src import config


def draw_landmarks(frame: np.ndarray, landmarks: list[tuple[int, int]] | None) -> np.ndarray:
    """отрисовка точек лица на кадре."""

    if landmarks is None:
        return frame

    for x, y in landmarks:
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    return frame


def draw_roi(frame: np.ndarray, roi_mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.3,) -> np.ndarray:
    overlay = frame.copy()
    colored = np.zeros_like(frame)
    colored[:] = color
    cv2.bitwise_and(colored, colored, mask=roi_mask, dst=colored)
    gray_mask = roi_mask > 0
    overlay[gray_mask] = cv2.addWeighted(frame, 1 - alpha, colored, alpha, 0)[gray_mask]
    return overlay

def bvp_plot(bvp: np.ndarray, w: int, h: int, hr: float) -> np.ndarray:
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    if len(bvp) < 2:
        return panel
    sig = bvp[-w:]
    mn, mx = sig.min(), sig.max()
    if mx - mn < 1e-6:
        return panel
    norm = (sig - mn) / (mx - mn)
    ys = ((1 - norm) * (h - 4) + 2).astype(int)
    xs = np.linspace(0, w - 1, len(ys)).astype(int)
    for i in range(len(xs) - 1):
        cv2.line(panel, (xs[i], ys[i]), (xs[i + 1], ys[i + 1]), (0, 255, 100), 1)
    label = f"BVP  |  HR: {hr:.0f} BPM" if hr is not None else "BVP  |  HR: --"
    cv2.putText(
        panel,
        label,
        (4, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        config.FONT_COLOR,
        config.FONT_THICKNESS,
    )
    return panel

def roi_to_mask(roi: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)


def draw_status(
    frame: np.ndarray,
    hr: float | None,
    status: str,
    color: tuple[int, int, int],
) -> np.ndarray:
    label = f"{status}  |  HR: {hr:.0f} BPM" if hr is not None else f"{status}  |  HR: --"
    cv2.putText(
        frame,
        label,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        config.FONT_SCALE,
        color,
        config.FONT_THICKNESS,
    )
    return frame
