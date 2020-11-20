# Novel molecules generation

Code for the paper [Towards Efficient Generation, Correction and Properties Control of Unique Drug-Like Structures](https://chemrxiv.org/articles/Towards_Efficient_Generation_Correction_and_Properties_Control_of_Unique_Drug-like_Structures/9941858).
Enables fast generation and correction of novel chemical structures based on small reference dataset as well as
their properties prediction(for now, log solubility, bbbp(blood-brain barrier permeability), and ames).

## Getting Started

Install all necessary packages by running ``` pip3 install -r requirements.txt ```

## Usage

To generate new structures
1. specify the path to the reference dataset of SMILES in pipeline/pipeline_config.yaml
2. run ```python3 generate_new_structures.py```

To train models

1. specify dataset paths and hyperparameters in the corresponding config file (i.e ae/train_config.yaml)
2. run ```python3 train.py [path to config file] ```

## Data

Data used to train models will be included in */data* folder.

## Authors



## License



## Acknowledgments


