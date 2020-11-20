import torch
import torch.utils.data


class EmbDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config

        # load embeddings
        data_file = open(self.config['path_to_emb'], 'r')
        item = data_file.readline().split(',')
        irow = 1
        for _ in data_file:
            irow += 1
        number_of_columns = len(item)

        smiles = []
        representation_as_tensor = torch.zeros(irow, (number_of_columns - 1))

        if self.config['path_to_labels']:
            # load labels
            labels_file = open(self.config['path_to_labels'], 'r')
            label0 = {}
            for line in labels_file:
                item = line.split(",")
                label0[item[0].strip()] = float(item[1].strip())
            labels_file.close()
            if self.config['labels_one_hot']:
                self.labels_as_tensor = torch.zeros(irow, 2)
            else:
                self.labels_as_tensor = torch.zeros(irow, 1)

        irow = 0
        data_file.seek(0, 0)
        for line in data_file:
            item = line.split(",")
            smiles.append(item[0])
            for icolumn in range(1, number_of_columns):
                representation_as_tensor[irow][(icolumn - 1)] = float(item[icolumn])

            if self.config['path_to_labels']:
                if self.config['labels_one_hot']:
                    if label0[item[0]] == 0.:
                        self.labels_as_tensor[irow][0] = 1.0
                    else:
                        self.labels_as_tensor[irow][1] = 1.0
                else:
                    self.labels_as_tensor[irow] = label0[item[0]]
            irow += 1

        data_file.close()

        self.length = irow
        self.smiles_strings = smiles
        self.representation = representation_as_tensor

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.smiles_strings[index], self.representation[index], self.labels_as_tensor[index]
