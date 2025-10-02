import torch
from torch import nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def swish(x):
    return x * torch.sigmoid(x)


class SelfAttnPooling(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x):
        # x: [n_nodes, hidden_size]
        scores = torch.softmax(self.attn(x), dim=0)   # (n_nodes, 1)
        pooled = torch.sum(scores * x, dim=0)         # (hidden_dim,)
        return pooled
