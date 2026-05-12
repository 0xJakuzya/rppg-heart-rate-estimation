import torch
import torch.nn as nn
from src import config

class ShiftLoss(nn.Module):
    """
    Сдвинутая по времени функция потерь Пирсона для учета рассинхронизации PPG-сигнала и видео.
    Реализует: Loss = 1 - max_τ∈(-δt,δt) Pearson(srPPG(t+τ), sPPG(t))
    """
    def __init__(
        self,
        max_shift_sec: float = config.SHIFT_LOSS_MAX_SHIFT_SEC,
        fps: float = config.SHIFT_LOSS_FPS,
        eps: float = config.SHIFT_LOSS_EPS,
    ):
        super().__init__()
        self.max_shift_sec = max_shift_sec
        self.fps = fps
        self.eps = eps
        self.max_shift_frames = max(1, int(round(max_shift_sec * fps)))

    def forward(self, predicted_ppg: torch.Tensor, target_ppg: torch.Tensor) -> torch.Tensor:
        batch_size = predicted_ppg.shape[0]
        max_corr = torch.zeros(batch_size, device=predicted_ppg.device)
        for shift in range(-self.max_shift_frames, self.max_shift_frames + 1):
            if shift >= 0:
                pred = predicted_ppg[:, shift:]
                targ = target_ppg[:, :-shift] if shift > 0 else target_ppg
            else:
                pred = predicted_ppg[:, :shift]
                targ = target_ppg[:, -shift:]

            if pred.shape[1] == 0:
                continue

            pred_norm = pred - pred.mean(dim=1, keepdim=True)
            targ_norm = targ - targ.mean(dim=1, keepdim=True)
            numerator = (pred_norm * targ_norm).sum(dim=1)
            denominator = (
                pred_norm.pow(2).sum(dim=1).sqrt() *
                targ_norm.pow(2).sum(dim=1).sqrt() +
                self.eps
            )
            corr = numerator / denominator
            max_corr = torch.maximum(max_corr, corr)
        return (1.0 - max_corr).mean()
