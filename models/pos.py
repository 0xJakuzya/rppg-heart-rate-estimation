import math
import numpy as np
from src.utils import detrend, bandpass_filter
from src import config

def pos(rgb, fs):
    """
    Метод POS-Wang для измерения сердечного ритма из видео.
    Вход: rgb: np.ndarray, fs: float
    Выход: bvp: np.ndarray
    """
    win_sec = 1.6
    N = rgb.shape[0]
    H = np.zeros(N)
    l = math.ceil(win_sec * fs)
    for n in range(N):
        m = n - l
        if m >= 0:
            segment = rgb[m:n, :]
            Cn = segment / (np.mean(segment, axis=0) + 1e-9)
            Cn = Cn.T
            S = np.array([[0, 1, -1], [-2, 1, 1]], dtype=np.float64) @ Cn
            std_ratio = np.std(S[0]) / (np.std(S[1]) + 1e-9)
            h = S[0] + std_ratio * S[1]
            h -= h.mean()
            H[m:n] += h
    H = detrend(H)
    bvp = bandpass_filter(H, fs, config.CHEBY_LO, config.CHEBY_HI)
    bvp -= bvp.mean()
    bvp /= bvp.std() + 1e-9
    return bvp.astype(np.float32)

class POS:
    def __init__(self, fps: float):
        self.fps = fps

    def run(self, rgb: np.ndarray) -> np.ndarray:
        return pos(rgb, self.fps)
