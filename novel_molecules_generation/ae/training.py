import torch
from torch.nn import functional as F


def loss_function(recon_x, x):
    bce = F.binary_cross_entropy(recon_x, x, reduction='sum')
    return bce


def train(epoch, model, optimizer, device, train_loader):
    model.train()
    train_loss = 0
    for batch_idx, (_, data) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch = model(data)
        loss = loss_function(recon_batch, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader),
            loss.item() / len(data)))
    print('====> Epoch: {} Average loss on train set: {:.6f}'.format(
          epoch, train_loss / len(train_loader.dataset)))



def test(epoch, model, device, test_loader):
    model.eval()

    test_loss = 0
    with torch.no_grad():
        for i, (_, data) in enumerate(test_loader):
            data = data.to(device)
            recon_batch = model(data)
            test_loss += loss_function(recon_batch, data).item()
    test_loss /= len(test_loader.dataset)
    print('====> Epoch: {} Average loss on test set:  {:.6f}'.format(epoch, test_loss))

    # esol_loss = 0
    # with torch.no_grad():
    #     for i, (_, data) in enumerate(esol_loader):
    #         data = data.to(device)
    #         recon_batch = model(data)
    #         esol_loss += loss_function(recon_batch, data).item()
    # esol_loss /= len(esol_loader.dataset)
    # print('====> Epoch: {} Average loss on ESol set:  {:.6f}'.format(epoch, esol_loss))
    #
    # esol_test_loss = 0
    # with torch.no_grad():
    #     for i, (_, data) in enumerate(esol_test_loader):
    #         data = data.to(device)
    #         recon_batch = model(data)
    #         esol_test_loss += loss_function(recon_batch, data).item()
    # esol_test_loss /= len(esol_test_loader.dataset)
    # print('====> Epoch: {} Average loss on ESol test set:  {:.6f}'.format(epoch, esol_test_loss))
