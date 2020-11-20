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

        p.write_embeddings_to_file(p.SMILES_ds.smiles_strings, embds, 'data/ames/ames_embds_train.csv')

        end = time.time() - st


