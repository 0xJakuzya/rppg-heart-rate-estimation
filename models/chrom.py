import numpy as np
from src.utils import detrend, bandpass_filter
from src import config

def chrom(rgb: np.ndarray, fs: float) -> np.ndarray:
    """
    Метод CHROM для измерения сердечного ритма из видео.
    Вход: rgb: np.ndarray, fs: float
    Выход: bvp: np.ndarray
    """
    win_sec = 1.6
    N = rgb.shape[0]
    l = max(int(win_sec * fs), 2)
    H = np.zeros(N, dtype=np.float64)
    for n in range(l, N + 1):
        segment = rgb[n - l:n, :].astype(np.float64)
        mu = np.mean(segment, axis=0) + 1e-9
        Cn = segment / mu
        Rn, Gn, Bn = Cn[:, 0], Cn[:, 1], Cn[:, 2]
        Xs = 3 * Rn - 2 * Gn
        Ys = 1.5 * Rn + Gn - 1.5 * Bn
        if l >= 9:
            Xs = bandpass_filter(Xs, fs, config.CHEBY_LO, config.CHEBY_HI)
            Ys = bandpass_filter(Ys, fs, config.CHEBY_LO, config.CHEBY_HI)
        alpha = np.std(Xs) / (np.std(Ys) + 1e-9)
        h = Xs - alpha * Ys
        h -= h.mean()
        H[n - l:n] += h
    H = detrend(H)
    bvp = bandpass_filter(H, fs, config.CHEBY_LO, config.CHEBY_HI)
    bvp -= bvp.mean()
    std = bvp.std()
    if std < 1e-6:
        return np.zeros(N, dtype=np.float32)
    bvp /= std
    return bvp.astype(np.float32)

class CHROM:
    def __init__(self, fps: float):
        self.fps = fps

    def run(self, rgb: np.ndarray) -> np.ndarray:
        return chrom(rgb, self.fps)
