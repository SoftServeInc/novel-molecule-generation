import os
import yaml
import torch
from data_processing.embeddings import EmbDataset
from data_processing.smiles import SMILESDataset
import sys
from copy import deepcopy

try:
    assert len(sys.argv) > 1
except AssertionError:
    'Please specify path to the config file for training of the desired model'
    exit()

with open(sys.argv[1]) as f:
    config = yaml.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config['device'] = device

if config['model'] == 'ae':
    from ae.autoencoder import AE as Model
    from ae.training import train, test
elif config['model'] == 'c2f':
    from c2f.classifier import C2F as Model
    from c2f.training import train, test
elif config['model'] == 'e2f':
    from e2f.regressor import E2F as Model
    from e2f.training import train, test
elif config['model'] == 'seq2seq':
    from seqtoseq.seqtoseq import Seq2Seq as Model
    from seqtoseq.training import train, test

print('Loading data...')

if 'path_to_embd_dataset_train' in config and 'path_to_labels_train' in config:
    config['path_to_emb'] = config['path_to_embd_dataset_train']
    config['path_to_labels'] = config['path_to_labels_train']
    train_dataset = EmbDataset(config)
else:
    config['path_to_datafile'] = config['path_to_datafile_train']
    train_dataset = SMILESDataset(config)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

if 'path_to_embd_dataset_test' in config and 'path_to_labels_test' in config:
    config['path_to_emb'] = config['path_to_embd_dataset_test']
    config['path_to_labels'] = config['path_to_labels_test']
    test_dataset = EmbDataset(config)
else:
    config['path_to_datafile'] = config['path_to_datafile_test']
    test_dataset = SMILESDataset(config)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
print('Done loading')
print('Training on {} samples, testing on {} samples'.format(len(train_dataset), len(test_dataset)))



model = Model(deepcopy(config)).to(device)
path_to_model = config['path_to_save_model']

if os.path.exists(path_to_model):
    model.load_state_dict(torch.load(path_to_model))

optimizer = torch.optim.Adam(model.parameters(), lr=config['l_r'], weight_decay=0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['decay_lr_every'], gamma=config['gamma'])

for epoch in range(1, config['num_epochs'] + 1):
    train(epoch, model, optimizer, device, train_loader)
    test(epoch, model, device, test_loader)
    if epoch > 0 and epoch % config['save_model_every'] == 0:
        torch.save(model.state_dict(), path_to_model)
    scheduler.step(epoch)

torch.save(model.state_dict(), path_to_model)
