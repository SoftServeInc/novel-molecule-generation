# Novel molecules generation

Code for the paper [Towards Efficient Generation, Correction and Properties Control of Unique Drug-Like Structures](https://chemrxiv.org/articles/Towards_Efficient_Generation_Correction_and_Properties_Control_of_Unique_Drug-like_Structures/9941858).
Enables fast generation and correction of novel chemical structures based on small reference dataset as well as
their properties prediction(predictor architectures were trained and tested for log solubility, bbbp(blood-brain barrier permeability), and ames).

## Getting Started

Install all necessary packages by running ``` pip3 install -r requirements.txt ```

## Usage

To train models

1. specify dataset paths and hyperparameters in the corresponding config file (i.e ae/train_config.yaml)
2. run ```python3 train.py [path to config file] ```

To generate new structures

1. specify the path to the reference dataset of SMILES in pipeline/pipeline_config.yaml
2. run ```python3 generate_new_structures.py```

To analyze the novel structures

1. use ```novel_smiles_analysis.ipynb```

## Data

All models were trained on publicly available data only: autoencoders on data from eMolecules database https://www.emolecules.com/info/plus/download-database, solubility dataset was compiled from sets provided by Huuskonen, Hou et al., Delaney, and Mitchell (see the paper for more details). 

## Authors
Maksym Druchok mdruc@softserveinc.com

Dzvenymyra Yarish dyari@softserveinc.com

Oleksandr Gurbych ogurb@softserveinc.com

Mykola Maksymenko mmaks@softserveinc.com
