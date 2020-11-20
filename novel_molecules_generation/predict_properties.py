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

        properties = p.predict_properties(embds)

        p.write_results_to_file(p.SMILES_ds.smiles_strings, properties, {})

        end = time.time() - st
        print('Predicted properties for {} structures in {:.2f}s'.format(len(embds), end))

