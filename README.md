# About

Efficient design and screening of novel molecules is a major challenge in drug and materials design. This report focuses on a multi-stage pipeline in which several deep neural network (DNN)models are combined to map discrete molecular representations into continuous vector space to later generate from it new molecular structures with desired properties. Here the Attention-based Sequence-to-Sequence model is added to “spellcheck” and correct generated structures while the oversampling in the continuous space allows generating candidate structures with desired distribution for properties and molecular descriptors even for small reference datasets. We further use computer simulation to validate the desired properties in the numerical experiment. With the focus on the drug design, such a pipeline allows generating novel structures with control of SAS(Synthetic Accessibility Score) and a series of ADME metrics that assess the drug-likeliness.

# Novel molecules generation

Code for the paper [Towards Efficient Generation, Correction and Properties Control of Unique Drug-Like Structures](https://chemrxiv.org/articles/Towards_Efficient_Generation_Correction_and_Properties_Control_of_Unique_Drug-like_Structures/9941858).
Enables fast generation and correction of novel chemical structures based on small reference dataset as well as
their properties prediction(predictor architectures were trained and tested for log solubility, bbbp(blood-brain barrier permeability), and ames).

## Installation instructions

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

- Maksym Druchok mdruc@softserveinc.com
- Dzvenymyra Yarish dyari@softserveinc.com
- Oleksandr Gurbych ogurb@softserveinc.com
- Mykola Maksymenko mmaks@softserveinc.com
