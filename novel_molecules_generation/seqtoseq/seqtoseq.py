import torch
import torch.nn as nn
import torch.nn.functional as F


class Seq2Seq(nn.Module):
    def __init__(self, config):
        super(Seq2Seq, self).__init__()
        self.config = config
        self.config['max_smiles_len'] += 1
        self.config['smiles_dict_size'] += 2
        self.encoder = EncoderRNN(config['smiles_dict_size'], config['cell_type'], config['hidden_size'],
                                  config['num_layers'], config['encoder_dropout_p'])
        self.decoder = AttnDecoderRNN(config['smiles_dict_size'], config['cell_type'], config['hidden_size'],
                                      config['num_layers'], config['max_smiles_len'], config['decoder_dropout_p'])
        self.curr_teacher_forcing_p = self.config['start_teacher_forcing_p']
        self.teacher_forcing_decay = self.config['teacher_forcing_decay']

    def forward(self, input, target):
        encoder_outputs, encoder_hidden = self.encode(input)
        decoder_outputs = self.decode(target, encoder_hidden, encoder_outputs)
        return decoder_outputs

    def encode(self, input_batch):
        batch_size = input_batch.size()[0]
        encoder_hidden = self.encoder.init_hidden(self.config['device'], batch_size=batch_size)
        input_batch = input_batch.transpose(0, 1)
        input_length = input_batch.size()[0]
        encoder_outputs = torch.zeros(self.config['max_smiles_len'], batch_size, self.encoder.hidden_size, device=self.config['device'])
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_batch[ei], encoder_hidden, batch_size=batch_size)
            # print(encoder_output.size())
            encoder_outputs[ei] = encoder_output[0]
        return encoder_outputs, encoder_hidden

    def decode(self, target_batch, encoder_hidden, encoder_outputs):
        batch_size = target_batch.size()[0]
        target_batch = target_batch.transpose(0, 1)
        target_length = target_batch.size()[0]

        decoder_input = torch.tensor([58] * batch_size, device=self.config['device'])  # todo: add sos toekn as param

        decoder_hidden = encoder_hidden
	
        use_teacher_forcing = True if torch.rand(1).item() < self.curr_teacher_forcing_p else False # ToDo: add decay

        decoder_outputs = torch.zeros(target_length,  batch_size, self.config['smiles_dict_size'],
                                      device=self.config['device'])

        for di in range(self.config['max_smiles_len']):
            decoder_output, decoder_hidden, attention_weights = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs, batch_size=batch_size)
            decoder_outputs[di] = decoder_output
            if use_teacher_forcing:
                decoder_input = target_batch[di]  # Teacher forcing
            else:
                topv, topi = decoder_output.data.topk(1)
                decoder_input = torch.squeeze(topi).detach()

        return decoder_outputs


class EncoderRNN(nn.Module):
    def __init__(self, input_size, cell, hidden_size, n_layers, dropout_p): # input_size  - vocabulary size
        super(EncoderRNN, self).__init__()
        self.cell = cell
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)  #emmbedding layer is trained also
        if self.cell == 'GRU':
            self.recurrent = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout_p)
        elif self.cell == 'LSTM':
            self.recurrent = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=dropout_p)

    def forward(self, input, hidden, batch_size):
        embedded = self.embedding(input).view(1, batch_size, -1)
        output = embedded
        output, hidden = self.recurrent(output, hidden)
        return output, hidden

    def init_hidden(self, device, batch_size=1):
        if self.cell == 'LSTM':
            return (torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device),
                    torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
                    )
        return torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)

    def get_num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([torch.prod(torch.Tensor(list(p.size()))) for p in model_parameters]).item()


class DecoderRNN(nn.Module):
    def __init__(self, cell, hidden_size, output_size, n_layers):
        super(DecoderRNN, self).__init__()
        self.cell = cell
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_size, hidden_size)
        if self.cell == 'GRU':
            self.recurrent = nn.GRU(hidden_size, hidden_size, n_layers)
        elif self.cell == 'LSTM':
            self.recurrent = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, batch_size=1):
        output = self.embedding(input).view(1, batch_size, -1)
        output = F.relu(output)
        output, hidden = self.recurrent(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self, device, batch_size=1):
        if self.cell == 'LSTM':
            return (torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device),
                    torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
                    )
        return torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)

    def get_num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([torch.prod(torch.Tensor(list(p.size()))) for p in model_parameters]).item()


class AttnDecoderRNN(nn.Module):
    def __init__(self, output_size, cell, hidden_size, n_layers, max_length, dropout_emb=0.5):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_emb = dropout_emb
        # self.dropout_r = dropout_r
        self.max_length = max_length
        self.cell = cell
        self.n_layers = n_layers

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_emb)
        if self.cell == 'GRU':
            self.recurrent = nn.GRU(hidden_size, hidden_size, n_layers,)
        elif self.cell == 'LSTM':
            self.recurrent = nn.LSTM(hidden_size, hidden_size, n_layers,)

        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, batch_size):
        embedded = self.embedding(input).view(1, batch_size, -1)
        embedded = self.dropout(embedded)

        if self.cell == 'LSTM':
            cell_state = hidden[0]
        else:
            cell_state = hidden
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], cell_state[0]), 1)), dim=1) #instead of embedded - encoder outputs

        # print(encoder_outputs.shape)
        # print(embedded.shape)
        #
        # attn_weights = F.softmax(
        #     self.attn(torch.cat((encoder_outputs, cell_state[0].repeat(self.max_length, 1, 1)), 1)), dim=1)


        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs.view(batch_size, encoder_outputs.shape[0], self.hidden_size))

        output = torch.cat((embedded[0], attn_applied.view(1, batch_size, self.hidden_size)[0]), 1)

        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.recurrent(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self, device, batch_size=1):
        if self.cell == 'LSTM':
            return (torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device),
                    torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
                    )
        return torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)

    def get_num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([torch.prod(torch.Tensor(list(p.size()))) for p in model_parameters]).item()
