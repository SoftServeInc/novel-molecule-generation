import torch
from torch import nn, optim
from torch.nn import functional as F


class C2F(nn.Module):
    def __init__(self, config):
        super(C2F, self).__init__()

        self.drop = nn.Dropout(p=0.0)

        self.fc0 = nn.Linear(512, 32)
        self.fc1 = nn.Linear(32, 2)

    def forward(self, x):

        h0 = x.view(-1, 512)

        h0 = torch.sigmoid(self.drop(self.fc0(h0))) - 0.5
        h0 = torch.sigmoid(self.drop(self.fc1(h0))) - 0.5

        h0 = F.softmax(h0, dim=1)

        return h0

