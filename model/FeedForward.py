import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = 1
        self.linear_h = nn.Linear(in_features=self.in_dim, out_features=self.hidden_dim, bias=True)
        self.linear_z = nn.Linear(in_features=self.hidden_dim, out_features=self.out_dim, bias=True)

    def forward(self, x):
        h = self.linear_h(x)
        z = self.linear_z(h)
        del h
        return z
