import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysNet(nn.Module):
    """
    Модель PhysNet для измерения сердечного ритма из видео.
    Вход: [batch, time, roi, 3, h, w]
    Выход: [batch, time]
    """
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 5, 5), padding=(1, 2, 2), bias=False),
            nn.BatchNorm3d(16),
            nn.ELU(inplace=True),
        )

        self.enc1 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ELU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ELU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
        )

        self.enc2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
        )

        self.enc3 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ELU(inplace=True),
        )

        self.head = nn.Conv3d(32, 1, kernel_size=1)

    @staticmethod
    def make_mosaic(x: torch.Tensor) -> torch.Tensor:
        b, t, r, c, h, w = x.shape
        x = x.reshape(b, t, 2, 4, c, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.reshape(b, t, c, 2 * h, 4 * w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_t = x.shape[1]
        v = self.make_mosaic(x)
        v = self.stem(v)
        v = self.enc1(v)
        v = self.enc2(v)
        v = self.enc3(v)
        v = self.bottleneck(v)
        v = F.interpolate(v,
            size=(target_t, v.shape[-2], v.shape[-1]),
            mode="trilinear",
            align_corners=False,
        )
        v = self.decoder(v)
        v = F.adaptive_avg_pool3d(v, (target_t, 1, 1))
        v = self.head(v)
        return v.view(v.shape[0], target_t)
