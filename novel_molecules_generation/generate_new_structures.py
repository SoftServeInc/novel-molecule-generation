from pipeline.pipeline import Pipeline

import torch

import yaml
import warnings
import time

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
        st = time.time()
        with open('pipeline/pipeline_config.yaml') as f:
            config = yaml.load(f)
        config['device'] = device

        p = Pipeline(config)

        embds = p.smiles_to_embeddings()

        new_embds = p.generate_new_embeddings(embds)
        properties = p.predict_properties(new_embds)

        new_smiles_one_hot = p.embeddings_to_smiles(new_embds)

        new_correct_smiles, smiles2indx = p.correct_smiles(new_smiles_one_hot)

        p.write_results_to_file(new_correct_smiles, properties, smiles2indx)

        end = time.time() - st
        print('Generated {} new SMILES in {:.2f}s'.format(len(new_correct_smiles), end))


