from aifordrugdiscovery.data_processing.smiles import SMILESDataset, onehot_to_smiles, is_valid_smiles, is_new_smiles, decode_smiles_from_seq2seq
from aifordrugdiscovery.ae.autoencoder import AE
from aifordrugdiscovery.seqtoseq.seqtoseq import Seq2Seq
from aifordrugdiscovery.c2f.classifier import C2F
from aifordrugdiscovery.e2f.regressor import E2F

import torch
from imblearn.combine import SMOTETomek

from copy import deepcopy


class Pipeline():
    def __init__(self, config, smiles_ds=None):
        self.config = config
        if smiles_ds:
            self.SMILES_ds = smiles_ds
        else:
            self.SMILES_ds = SMILESDataset(config)

        self.ae = AE(config).to(config['device'])
        self.ae.load_state_dict(torch.load(config['path_to_ae'], map_location=config['device']))

        self.predictors = []
        for path in config['paths_to_predictors']:
            if 'c2f' in path.split('/'):
                p = C2F(config).to(config['device'])
            elif 'e2f' in path.split('/'):
                p = E2F(config).to(config['device'])
            # p.load_state_dict(torch.load(path, map_location=config['device'])))
            self.predictors.append(p)

        self.seq2seq = Seq2Seq(deepcopy(config)).to(config['device'])
        # self.seq2seq.load_state_dictionary(config['path_to_seq2seq'])

        self.smt = SMOTETomek(random_state=42)

    def smiles_to_embeddings(self):
        SMILES_dl = torch.utils.data.DataLoader(self.SMILES_ds, batch_size=self.config['batch_size'], shuffle=False)
        all_embds = torch.zeros((len(self.SMILES_ds), self.config['embd_size']), dtype=torch.float32)

        with torch.no_grad():
            for batch_indx, data in enumerate(SMILES_dl):
                data = data.to(self.config['device'])
                embds = self.ae.encode(data)
                all_embds[batch_indx * self.config['batch_size']:batch_indx * self.config['batch_size'] + embds.shape[0]] = embds.cpu()
        return all_embds

    def generate_new_embeddings(self, ref_embds):
        embds_for_smote = torch.cat((ref_embds, torch.ones((len(self.SMILES_ds) * 2, self.config['embd_size']))))
        new_embds, _ = self.smt.fit_sample(embds_for_smote, torch.cat((torch.ones((len(self.SMILES_ds), 1)),
                                                            torch.zeros((len(self.SMILES_ds) * 2, 1)))))
        return new_embds[-len(self.SMILES_ds):]

    def predict_properties(self, embds):
        embd_dl = torch.utils.data.DataLoader(embds, batch_size=self.config['batch_size'], shuffle=False)
        predictions = {i: [] for i in range(len(self.predictors))}
        with torch.no_grad():
            for batch_indx, data in enumerate(embd_dl):
                data = data.to(self.config['device'])
                for i, prdct in enumerate(self.predictors):
                    pred = prdct(data).cpu()
                    predictions[i].extend(pred.topk(1)[1].squeeze(1).tolist() if pred.shape[-1] > 1
                                          else list(map(lambda x: round(x, 3), pred.squeeze(1).tolist())))
        return predictions

    def embeddings_to_smiles(self, embds):
        embd_dl = torch.utils.data.DataLoader(embds, batch_size=self.config['batch_size'], shuffle=False)
        all_smiles_onehot = torch.zeros((len(embds), self.config['max_smiles_len'], len(self.SMILES_ds.char2indx)), dtype=torch.float32)
        for batch_indx, data in enumerate(embd_dl):
            smiles = self.ae.decode(data).cpu().view(-1, self.config['max_smiles_len'], len(self.SMILES_ds.char2indx))
            all_smiles_onehot[batch_indx * self.config['batch_size']:batch_indx * self.config['batch_size'] + smiles.shape[0]] = smiles
        return all_smiles_onehot

    def correct_smiles(self, smiles_onehot):

        smiles = [onehot_to_smiles(onehot, self.SMILES_ds.indx2char, self.config['max_smiles_len']) for onehot in
                      smiles_onehot]
        smiles2indx = {sm: i for i, sm in enumerate(smiles)}

        valid, invalid = [], []

        for sm in smiles:
            if is_valid_smiles(sm):
                valid.append(sm)
            else:
                invalid.append(sm)

        invalid_as_tensor = torch.zeros((len(invalid), self.config['max_smiles_len'] + 1), dtype=torch.int64)
        invalid_as_tensor[:, -1] = torch.full((1, len(invalid)), self.SMILES_ds.eos_token).squeeze()
        for i, sm in enumerate(invalid):
            indx = smiles_onehot[smiles2indx[sm]].topk(1)[1]
            invalid_as_tensor[i, :-1] = indx.squeeze(1)

        invalid_smiles_dl = torch.utils.data.DataLoader(invalid_as_tensor, batch_size=self.config['batch_size'], shuffle=False)
        attempted_to_correct = []
        for batch_indx, data in enumerate(invalid_smiles_dl):
            data = data.to(torch.int64).to(self.config['device'])
            decoder_outputs = self.seq2seq(data, data)
            attempted_to_correct.extend([''.join(sm).split('EOS')[0] for sm in
                                        decode_smiles_from_seq2seq(decoder_outputs.cpu(), self.SMILES_ds)])

        corrected = []
        for i, sm in enumerate(attempted_to_correct):
            if is_valid_smiles(sm):
                corrected.append(sm)
                smiles2indx[sm] = smiles2indx[invalid[i]]
                del smiles2indx[invalid[i]]
        unique_valid = [sm for sm in valid if is_new_smiles(sm, self.SMILES_ds)]
        unique_corrected = [sm for sm in corrected if is_new_smiles(sm, self.SMILES_ds) and sm not in unique_valid] # ToDO: add chem DBs check if unique

        all_new_smiles = deepcopy(unique_valid)
        all_new_smiles.extend(unique_corrected)

        return all_new_smiles, smiles2indx

    def write_results_to_file(self, smiles, predictions, smiles2indx):
        with open('smiles_with_predicted_properties.csv', 'w') as f:
            f.write(','.join(['SMILES', ','.join(['preds_{}'.format(i) for i in range(len(self.predictors))])]))
            f.write('\n')
            for i, sm in enumerate(smiles):
                if smiles2indx:
                    preds = [predictions[j][smiles2indx[sm]] for j in range(len(self.predictors))]
                else:
                    preds = [predictions[j][i] for j in range(len(self.predictors))]

                f.write(','.join([sm, ','.join(map(str, preds))]))
                f.write('\n')

    def write_embeddings_to_file(self, smiles, embds, filepath):
        with open(filepath, 'w') as f:
            for sm, embd in zip(smiles, embds):
                f.write(','.join([sm, ','.join(map(lambda x: str(x.item()), embd))]))
                f.write('\n')




