import torch
from torch import nn, optim
from torch.nn import functional as F


def loss_function(recon_x, x):
    return F.mse_loss(recon_x, x, reduction='sum')


def train(epoch, model, optimizer, device, train_loader):
    model.train()
    train_loss = 0
    for batch_idx, (label, data, feature) in enumerate(train_loader):
        data = data.to(device)
        feature = feature.to(device)
        optimizer.zero_grad()
        recon_feature = model(data)
        loss = loss_function(recon_feature, feature.view(-1, 1))
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print('====> Epoch: {} Average loss on train set: {:.6f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch, model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (label, data, feature) in enumerate(test_loader):
            data = data.to(device)
            feature = feature.to(device)
            recon_feature = model(data)
            loss = loss_function(recon_feature, feature.view(-1, 1))
            test_loss += loss.item()
    print('====> Epoch: {} Average loss on test set : {:.6f}'.format(
          epoch, test_loss / len(test_loader.dataset)))

