import torch
from torch import nn
from utils import log


class MinesweeperAgent(nn.Module):
    def __init__(self, feat_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, feat_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
        )
        self.out = nn.Conv2d(feat_dim, 2, kernel_size=1)

    @log
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x).to('cuda')
        return self.out(self.net(x))  # (B, 2, H, W)

    def __call__(self, *args, **kwds) -> torch.Tensor:
        return super().__call__(*args, **kwds)
