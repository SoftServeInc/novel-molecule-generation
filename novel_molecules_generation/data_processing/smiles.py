import torch
import torch.utils.data
import random
import re
from rdkit import Chem
from rdkit import rdBase


# class SMILESDataset(torch.utils.data.Dataset):
#     def __init__(self, path_to_csv_file, size_of_smiles_dict):
#         data_file = open(path_to_csv_file, 'r')
#         item = data_file.readline().split(',')
#         irow = 0
#         icolumn = len(item)
#         position = data_file.tell()
#         for _ in data_file:
#             irow += 1
#
#         label = []
#         representation_as_tensor = torch.empty(irow, (icolumn - 1) * size_of_smiles_dict)
#
#         irow = 0
#         number_of_columns = icolumn
#         data_file.seek(position)
#         for line in data_file:
#             item = line.split(',')
#             lbl, representation_as_tensor[irow] = csv_to_onehot(item, size_of_smiles_dict)
#             label.append(lbl)
#             irow += 1
#         data_file.close()
#
#         self.length = irow
#         self.label = label
#         self.representation = representation_as_tensor
#         self.number_of_columns = number_of_columns - 1
#
#     def __len__(self):
#         return self.length
#
#     def __getitem__(self, index):
#         return self.label[index], self.representation[index]
#
#     def col_num(self):
#         return self.number_of_columns


class SMILESDataset:
    def __init__(self, config):
        self.config = config

        with open(config['path_to_smiles_dict'], 'r') as f:
            self.char2indx = {l.strip().split('\t')[1]: int(l.strip().split('\t')[0]) for l in f.readlines()}
        self.indx2char = {v: k for k, v in self.char2indx.items()}
        self.max_smiles_len = self.config['max_smiles_len']
        self.pad_token = 0
        self.sos_token = len(self.indx2char)
        self.eos_token = len(self.indx2char) + 1

        self.smiles = []
        self.smiles_strings = []
        self.labels = []

        with open(self.config['path_to_datafile'], 'r') as f:
            lines = list(f.readlines())
            transformed = list(map(self.transform, lines))
            # self.smiles_strings.extend([tr[0] for tr in transformed if tr])
            # self.smiles.extend([tr[1] for tr in transformed if tr])
            # if self.config['mode'] == 'predictor':
            #     self.labels.extend([tr[2] for tr in transformed if tr])
            for tr in transformed:
                if tr and tr not in self.smiles_strings:
                    self.smiles_strings.append(tr[0])
                    self.smiles.append(tr[1])
                    if self.config['mode'] == 'predictor':
                        self.labels.append(tr[2])

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, index):
        if self.config['mode'] == 'ae' or self.config['mode'] == 'seq2seq':
            if self.config['rn_in_input']:
                return torch.FloatTensor(add_random_noise(self.smiles[index], self.config['max_smiles_len'], len(self.char2indx), self.eos_token)), \
                       torch.FloatTensor(self.smiles[index])
            return torch.FloatTensor(self.smiles[index]), torch.FloatTensor(self.smiles[index])
        elif self.config['mode'] == 'predictor':
            return torch.Tensor(self.smiles[index]), torch.Tensor(self.labels[index])
        elif self.config['mode'] == 'denovo':
            return torch.Tensor(self.smiles[index])

    def transform(self, l):
        sm_str = re.split('[,\t]', l.strip())[0]
        # sm_str = Chem.MolToSmiles(Chem.MolFromSmiles(sm_str))  # convert to canonical
        sm = split_smiles_string(sm_str, self.char2indx)
        real_sm_len = len(sm)
        if len(sm) > self.config['max_smiles_len']:
            return []
        try:
            sm = [self.char2indx[ch] for ch in sm]
        except KeyError:
            print(sm_str)
            return []
        sm = pad_smile(sm, self.config['max_smiles_len'])

        if self.config['smiles_one_hot']:
            sm = [indx_to_onehot(indx, self.char2indx) for indx in sm]

        lbl = None
        if self.config['mode'] == 'predictor':
            lbl = float(l.strip().split(',')[1])
            if self.config['labels_one_hot']:
                lbl = indx_to_onehot(l, [0, 0])

        return sm_str, sm, lbl



def indx_to_onehot(indx, smiles_dict):
    vec = [0] * len(smiles_dict)
    vec[indx] = 1
    return vec


def pad_smile(smile_indices, max_smiles_len):
    while len(smile_indices) < max_smiles_len:
        smile_indices.append(0)
    return smile_indices


def split_smiles_string(smiles_string, smiles_dict):
    chars = []
    for ch in smiles_dict:
        if ch == '++':
            chars.append('\+\+')
        elif ch in ('$', '\\', '*', '[', ']', '(', ')', '.', '+'):
            chars.append('\\' + ch)
        else:
            chars.append(ch)
    delimiter = '|'.join(chars)
    splited = re.split('(' + delimiter + ')', smiles_string)
    splited = list(filter(None, splited))
    return splited


def onehot_to_smiles(as_tensor, smiles_dict, number_of_columns):
    as_smiles = ''
    for i in as_tensor.reshape(number_of_columns, len(smiles_dict)).max(1)[1]:
        as_smiles += smiles_dict[i.item()]
    return as_smiles.lstrip('0')


def csv_to_onehot(csv_string, size_of_smiles_dict):
    representation_as_tensor = torch.zeros((len(csv_string) - 1) * size_of_smiles_dict)
    label = csv_string[0]
    for i in range(1, len(csv_string)):
        representation_as_tensor[(i - 1) * size_of_smiles_dict + int(csv_string[i])] = 1
    return label, representation_as_tensor


def is_valid_smiles(smiles_string):
    rdBase.DisableLog('rdApp.error')
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is not None:
        valid = True
    else:
        valid = False
    return valid


def is_new_smiles(smiles_string ,smiles_ds):
    return smiles_string in smiles_ds.smiles_strings


def add_random_noise(smile_lst, max_smiles_len, n_chars, eos_token):
    real_sm_len = smile_lst.index(eos_token)
    num_edits = random.choices([0, 1, 2, 3], [0.1, 0.3, 0.3, 0.3], k=1)[0]
    edit_indxs = [random.randint(0, real_sm_len - 1) for _ in range(num_edits)]

    if real_sm_len >= (max_smiles_len - num_edits):
        edit_op = random.choices(['repl', 'del'], [0.95, 0.05], k=num_edits)
    else:
        edit_op = random.choices(['repl', 'ins', 'del'], [0.9, 0.05, 0.05], k=num_edits)
    smile_with_rand = []
    for i in range(real_sm_len):
        if i in edit_indxs:
            if edit_op[edit_indxs.index(i)] == 'repl':
                smile_with_rand.append(random.randint(1, n_chars - 1))
            elif edit_op[edit_indxs.index(i)] == 'ins':
                smile_with_rand.append(random.randint(1, n_chars - 1))
                smile_with_rand.append(smile_lst[i])
        else:
            smile_with_rand.append(smile_lst[i])
    smile_with_rand.append(eos_token)
    smile_with_rand = pad_smile(smile_with_rand, max_smiles_len + 1)
    # if len(smile_with_rand) != len(smile_lst):
    #     print(smile_lst)
    #     print(smile_with_rand)
    return smile_with_rand


def decode_smiles_from_seq2seq(decoder_output, smiles_ds):
    decoder_output = decoder_output.transpose(0, 1)
    decoded_smiles = []
    _, decoder_output_indices = decoder_output.topk(1)
    decoder_output_indices = decoder_output_indices.squeeze(2)
    for di in range(decoder_output_indices.size()[0]):
        decoded_smile = []
        for indx in decoder_output_indices[di]: #skip SOS token
            if indx.item() == smiles_ds.sos_token:
                continue
            elif indx.item() == smiles_ds.eos_token:
                decoded_smile.append('EOS')
                break
            decoded_smile.append(smiles_ds.indx2char[indx.item()])
        decoded_smiles.append(decoded_smile)
    return decoded_smiles
