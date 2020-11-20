import torch
from torch import nn, optim
from torch.nn import functional as F


def loss_function(recon_x, x):
    loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    #loss = F.mse_loss(recon_x, x, reduction='sum')
    return loss


def train(epoch, model, optimizer, device, train_loader):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (label, data, feature) in enumerate(train_loader):
        data = data.to(device)
        feature = feature.to(device)
        optimizer.zero_grad()
        recon_feature = model(data)
        #loss = loss_function(recon_feature, feature.view(-1, 1))
        loss = loss_function(recon_feature, feature.view(-1, 2))
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        correct += (recon_feature.topk(1)[1] == feature.topk(1)[1]).sum().item()
    print('====> Epoch: {} Average loss on train set: {:.6f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    print('====> Epoch: {} Accuracy on train set: {:.2f}'.format(
        epoch, correct / len(train_loader.dataset)))



def test(epoch, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (label, data, feature) in enumerate(test_loader):
            data = data.to(device)
            feature = feature.to(device)
            recon_feature = model(data)
            #loss = loss_function(recon_feature, feature.view(-1, 1))
            loss = loss_function(recon_feature, feature.view(-1, 2))
            test_loss += loss.item()
            correct += (recon_feature.topk(1)[1] == feature.topk(1)[1]).sum().item()
    print('====> Epoch: {} Average loss on test set : {:.6f}'.format(
          epoch, test_loss / len(test_loader.dataset)))
    print('====> Epoch: {} Accuracy on test set: {:.2f}'.format(
        epoch, correct / len(test_loader.dataset)))
