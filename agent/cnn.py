from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
import typing as tp
class CNN(nn.Module):
    def __init__(self, action_space, *args, **kwargs) -> None:
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 45, 3, 1, 1)
        self.flat1 = nn.Flatten()
        self.linear = nn.Linear(400, action_space)

    def __call__(self, *args: tp.Any, **kwds: tp.Any) -> torch.Tensor:
        return super().__call__(*args, **kwds)
    
    def forward(self, x) -> torch.Tensor:
        # print(x.shape)
        print(x.shape)
        x = F.relu(self.conv1(x))
        # x = F.relu(self.flat1(x))
        x = x.view((-1, 45 * 10 * 10))
        x = F.relu(self.linear(x))
        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    x = torch.Tensor(np.zeros((1, 20, 20))).to(device)
    cnn = CNN(20 * 20, 64)
    cnn.to(device)
    q = cnn(x)
    print(q)