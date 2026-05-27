import torch
from torch import nn
from utils import log


class MinesweeperAgent(nn.Module):
    def __init__(self, feat_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, feat_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_dim), nn.ReLU(),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_dim), nn.ReLU(),
        )
        # Dueling streams: V(s) scalar + A(s,a) spatial map
        self.advantage = nn.Conv2d(feat_dim, 2, kernel_size=1)
        self.value = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, feat_dim, 1, 1)
            nn.Flatten(),             # (B, feat_dim)
            nn.Linear(feat_dim, 1),   # (B, 1)
        )

    @log
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x).to('cuda')
        features = self.net(x)
        advantage = self.advantage(features)         # (B, 2, H, W)
        value = self.value(features)                 # (B, 1)
        value = value[:, :, None, None]              # (B, 1, 1, 1) — broadcasts over H, W, channels
        return value + advantage - advantage.mean(dim=[1, 2, 3], keepdim=True)  # (B, 2, H, W)

    def __call__(self, *args, **kwds) -> torch.Tensor:
        return super().__call__(*args, **kwds)
