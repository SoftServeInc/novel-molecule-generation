# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.utils.data
from data_processing.smiles import decode_smiles_from_seq2seq


def train(epoch, model, optimizer, device, train_loader):
    criterion = nn.NLLLoss(ignore_index=train_loader.dataset.pad_token)
    train_loss = 0

    for input_batch, trg_batch in train_loader:
            input_batch = input_batch.to(device)
            trg_batch = trg_batch.to(device)
            decoder_outputs = model(input_batch, trg_batch)
            optimizer.zero_grad()
            loss = 0
            for p,t in zip(decoder_outputs, trg_batch.transpose(0,1)):
                loss += criterion(p, t)
            loss.backward()
            optimizer.step()
            train_loss += loss
    model.curr_teacher_forcing_p *= model.teacher_forcing_decay
    print('====> Epoch: {} Average loss(avg over sum of losses from each batch) on train set: {:.6f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch, model, device, test_loader):
    test_loss = 0
    criterion = nn.NLLLoss(ignore_index=test_loader.dataset.pad_token)
    
    with open('seq2seq_test_results', 'w') as file:
        with torch.no_grad():
            for input_batch, trg_batch in test_loader:
                input_batch = input_batch.to(device)
                trg_batch = trg_batch.to(device)

                decoder_outputs = model(input_batch, trg_batch)
                loss = 0
                for p, t in zip(decoder_outputs, trg_batch.transpose(0, 1)):
                    loss += criterion(p, t)
                test_loss += loss

                decoded_smiles = decode_smiles_from_seq2seq(decoder_outputs.cpu(), test_loader.dataset)

                for i in range(input_batch.size()[0]):
                    input_sm = ''.join([test_loader.dataset.indx2char[indx.item()] if indx != test_loader.dataset.eos_token else ' '
                                        for indx in input_batch[i]]).split(' ')[0]
                    target_sm = ''.join([test_loader.dataset.indx2char[indx.item()] if indx != test_loader.dataset.eos_token else ' '
                                         for indx in trg_batch[i]]).split(' ')[0]
                    pred = ''.join(decoded_smiles[i]).split('EOS')[0]
                    if i == 10:
                        print(f'With errors: {input_sm}')
                        print(f'Correct:     {target_sm}')
                        print(f'Predicted:   {pred}')
                        print('')

                    file.write(f'With errors: {input_sm}\n')
                    file.write(f'Correct:     {target_sm}\n')
                    file.write(f'Predicted:   {pred}\n')
                    file.write('\n')

                    # utils.plot_attention([smiles_lang.indx2char[indx.item()] for indx in input_batch[i]],
                    #                      decoded_smiles[i], decoder_attentions[:, i, :], f"{config['exp_path']}attentions/attn_{i}.png", log=False)
                    # utils.plot_attention(input_sm, pred, decoder_attentions[:, i, :],
                    #                      f"{config['exp_path']}attentions/attn_log_{j}.png")
    print('====> Epoch: {} Average loss(avg over sum of losses from each batch) on test set: {:.6f}'.format(
        epoch, test_loss / len(test_loader.dataset)))














