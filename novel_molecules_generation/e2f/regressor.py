import torch
from torch import nn, optim
from torch.nn import functional as F


class E2F(nn.Module):
    def __init__(self, config):
        super(E2F, self).__init__()

        self.fc00 = nn.Linear(512, 512)
        self.fc01 = nn.Linear(512, 256)

        self.fc10 = nn.Linear(256, 256)
        self.fc11 = nn.Linear(256, 128)

        self.fc20 = nn.Linear(128, 128)
        self.fc21 = nn.Linear(128, 64)

        self.fc30 = nn.Linear(64, 64)
        self.fc31 = nn.Linear(64, 32)

        self.fc4 = nn.Linear(32, 8)

        self.fc5 = nn.Linear(8, 4)

        self.fc6 = nn.Linear(4, 2)

        self.fc7 = nn.Linear(2, 1)

        self.prelu = nn.PReLU()

    def forward(self, x):

        h0 = x.view(-1, 512)

        h1 = F.relu(self.fc00(h0) + h0)
        h2 = F.relu(self.fc00(h1) + h1 + h0)
        h3 = F.relu(self.fc00(h2) + h2 + h1 + h0)
        h0 = F.relu(self.fc01(h3))

        h1 = F.relu(self.fc10(h0) + h0)
        h2 = F.relu(self.fc10(h1) + h1 + h0)
        h3 = F.relu(self.fc10(h2) + h2 + h1 + h0)
        h4 = F.relu(self.fc10(h3) + h3 + h2 + h1 + h0)
        h0 = F.relu(self.fc11(h4))

        h1 = F.relu(self.fc20(h0) + h0)
        h2 = F.relu(self.fc20(h1) + h1 + h0)
        h3 = F.relu(self.fc20(h2) + h2 + h1 + h0)
        h4 = F.relu(self.fc20(h3) + h3 + h2 + h1 + h0)
        h0 = F.relu(self.fc21(h4))

        h1 = F.relu(self.fc30(h0) + h0)
        h2 = F.relu(self.fc30(h1) + h1 + h0)
        h3 = F.relu(self.fc30(h2) + h2 + h1 + h0)
        h4 = F.relu(self.fc30(h3) + h3 + h2 + h1 + h0)
        h0 = F.relu(self.fc31(h4))

        h0 = F.relu(self.fc4(h0))
        h0 = F.relu(self.fc5(h0))
        h2 = self.prelu(self.fc6(h0))
        h1 = self.fc7(h2)

        return h1  #h2

