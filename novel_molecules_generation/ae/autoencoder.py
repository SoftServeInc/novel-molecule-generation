import torch
# from torch import nn, optim
from torch import nn
from torch.nn import functional as F


class AE(nn.Module):
    def __init__(self, config):
        super(AE, self).__init__()

        self.drop = nn.Dropout(p=0.0)

        # input Tensor - (N_batch, Channels = 1, Height = 58, Width = 60)
        self.conv0 = nn.Conv2d(1, 60, kernel_size=(58, 3), stride=(1, 1))
        # output Tensor - (N_batch, Channels = 60, Height = 1, Width = 58)

        # input Tensor - (N_batch, Channels = 60, Height = 1, Width = 58)
        self.conv1 = nn.Conv2d(60, 87, kernel_size=(1, 19), stride=(1, 1))
        # output Tensor - (N_batch, Channels = 87, Height = 1, Width = 40)

        # input Tensor - (N_batch, Channels = 87, Height = 1, Width = 40)
        self.conv2 = nn.Conv2d(87, 116, kernel_size=(1, 11), stride=(1, 1))
        # output Tensor - (N_batch, Channels = 116, Height = 1, Width = 30)

        # input Tensor - (N_batch, Channels = 116, Height = 1, Width = 30)
        self.conv3 = nn.Conv2d(116, 120, kernel_size=(1, 2), stride=(1, 1))
        # output Tensor - (N_batch, Channels = 120, Height = 1, Width = 29)

        self.efc0 = nn.Linear(3480, 512)

        self.dfc0 = nn.Linear(512, 3480)

        # input Tensor - (N_batch, Channels = 120, Height = 1, Width = 29)
        self.deconv0 = nn.ConvTranspose2d(120, 116, kernel_size=(1, 2), stride=(1, 1))
        # output Tensor - (N_batch, Channels = 116, Height = 1, Width = 30)

        # input Tensor - (N_batch, Channels = 116, Height = 1, Width = 30)
        self.deconv1 = nn.ConvTranspose2d(116, 87, kernel_size=(1, 11), stride=(1, 1))
        # output Tensor - (N_batch, Channels = 87, Height = 1, Width = 40)

        # input Tensor - (N_batch, Channels = 87, Height = 1, Width = 40)
        self.deconv2 = nn.ConvTranspose2d(87, 60, kernel_size=(1, 19), stride=(1, 1))
        # output Tensor - (N_batch, Channels = 60, Height = 1, Width = 58)

        # input Tensor - (N_batch, Channels = 60, Height = 1, Width = 58)
        self.deconv3 = nn.ConvTranspose2d(60, 1, kernel_size=(58, 3), stride=(1, 1))
        # output Tensor - (N_batch, Channels = 1, Height = 58, Width = 60)

    def encode(self, x):
        h0 = F.relu(self.drop(self.conv0(x.view(-1, 1, 58, 60))))
        h1 = F.relu(self.drop(self.conv1(h0) + h0.view(-1, 87, 1, 40)))
        h2 = F.relu(self.drop(self.conv2(h1) + h0.view(-1, 116, 1, 30) + h1.view(-1, 116, 1, 30)))
        h3 = F.relu(self.drop(self.conv3(h2) + h0.view(-1, 120, 1, 29) + h1.view(-1, 120, 1, 29) +
                              h2.view(-1, 120, 1, 29)))
        h4 = F.relu(self.drop(self.efc0(h3.view(-1, 3480))))
        return h4

    def decode(self, z):
        h0 = F.relu(self.drop(self.drop(self.dfc0(z))))
        h1 = F.relu(self.drop(self.deconv0(h0.view(-1, 120, 1, 29))))
        h2 = F.relu(self.drop(self.deconv1(h1) + h1.view(-1, 87, 1, 40)))
        h3 = F.relu(self.drop(self.deconv2(h2) + h1.view(-1, 60, 1, 58) + h2.view(-1, 60, 1, 58)))
        h4 = self.deconv3(h3)
        h = torch.sigmoid(h4.view(-1, 3480))
        return h

    def forward(self, x):
        h = self.encode(x)
        x = self.decode(h)
        return x
