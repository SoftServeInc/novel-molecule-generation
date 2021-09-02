# Novel drug-like molecules generation

This repository implements the proposed pipelines and models in the paper [Towards Efficient Generation, Correction and Properties Control of Unique Drug-Like Structures](https://onlinelibrary.wiley.com/doi/10.1002/jcc.26494). It also contains scripts to reproduce the results of the models reported in the paper.

## About

Novel drug-like molecules generation project enables fast generation and correction of novel chemical structures based on small reference dataset as well as
their properties prediction (predictor architectures were trained and tested for log solubility, bbbp (blood-brain barrier permeability)).
Efficient design and screening of novel molecules is a major challenge in drug and materials design. The project focuses on a multi-stage pipeline in which several deep neural network (DNN) models are combined to map discrete molecular representations into continuous vector space to later generate from it new molecular structures with desired properties. Here the Attention-based Sequence-to-Sequence model is added to “spellcheck” and correct generated structures while the oversampling in the continuous space allows generating candidate structures with desired distribution for properties and molecular descriptors even for small reference datasets. With the focus on the drug design, such a pipeline allows generating novel structures with control of SAS (Synthetic Accessibility Score) and a series of ADME metrics that assess the drug-likeliness.

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

## Contributing

- Fork it
- Create your feature branch (git checkout -b my-new-feature)
- Commit your changes (git commit -am 'Added some feature')
- Push to the branch (git push origin my-new-feature)
- Create new Pull Request to us

## Authors

- Maksym Druchok mdruc@softserveinc.com
- Dzvenymyra Yarish dyari@softserveinc.com
- Oleksandr Gurbych ogurb@softserveinc.com
- Mykola Maksymenko mmaks@softserveinc.com
