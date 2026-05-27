import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import typing as tp
from utils import log

class MinesweeperAgent(nn.Module):
    def __init__(self, window_size=5, feat_dim=32):
        super().__init__()
        self.window_size = window_size
        self.in_channels = 3
        self.padding = (window_size - 1) // 2

        self.conv = nn.Conv2d(self.in_channels,
                              feat_dim,
                              kernel_size=window_size,
                              padding=0)
        
        self.norm = nn.LayerNorm(feat_dim)
        self.fc = nn.Linear(feat_dim, 2)
    @log
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x).to('cuda')
        b,c,h,w = x.shape
        unfolded = F.unfold(x,
                            kernel_size=self.window_size,
                            stride=1,
                            padding=self.padding)
        windows = unfolded.view(b*h*w, c, self.window_size, self.window_size)
        features = self.conv(windows)

        features = features.squeeze(-1).squeeze(-1)
        features = self.norm(features)
        features = F.relu(features)

        actions = self.fc(features)

        output = actions.view(b, h, w, 2)
        output = output.permute(0, 3, 1, 2)
        return output

    
    def __call__(self, *args, **kwds) -> torch.Tensor:
        return super().__call__(*args, **kwds)