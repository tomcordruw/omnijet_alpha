# Thesis Project: Prediction of particle jet evolution with OmniJet-α

This project was carried out using the original OmniJet-α model, please refer to the original authors' paper and their GitHub repository:
<div align="center">

Joschka Birk, Anna Hallin, Gregor Kasieczka

[![GitHub](https://img.shields.io/badge/GitHub-Omnijet--%CE%B1-blue)](https://github.com/uhh-pd-ml/omnijet_alpha)
[![arXiv](https://img.shields.io/badge/arXiv-2403.05618-b31b1b.svg)](https://arxiv.org/abs/2403.05618)
[![pytorch](https://img.shields.io/badge/PyTorch_2.2-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.2.1-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)

</div>

**Table of contents**:

- [OmniJet-α: Jet Evolution](#how-to-run-the-code)
  - [Overview](#overview)
    - [Dataset](#dataset)
    - [Jet Evolution Task](#jet-evolution-task)
    - [Data Preprocessing](#data-preprocessing)
    - [Tokenisation](#tokenisation)
    - [Generative Model](#generative-model)
  - [Citation](#citation)

## Overview

Changes were made to the data preprocessing, tokenisation and generative model.
Please refer to the respective sections to find out more.
Some of the files for tested configurations are included in `test_configs`.

### Dataset

The data used for this project was the JetClass dataset, please refer to it here:
[jet-universe/particle_transformer](https://github.com/jet-universe/particle_transformer).

### Jet Evolution Task

The goal was to study if, and how the evolution of particle jets could be predicted by modifying OmniJet-α.

In this case, "jet evolution" means the progressive particle decays that occur in a jet, 
starting from the initiating parton after a particle collision and ending with hadronisation. 
For this research, the studied jets were exclusively of the hadronic top quark decay channel, 
denoted by the `TTBar_XXX.root` files in the JetClass dataset.

### Data Preprocessing

The notebook `tokenise_and_to_parquet.ipynb` in `datasets` processes the JetClass data and stores the tokenised files in `.parquet`-format.
The code currently expects the JetClass data to be located in the same folder. In addition, a model checkpoint and config file for the VQ-VAE tokeniser model are required, see the chapter [Tokenisation](#tokenisation) for how to create the VQ-VAE tokeniser model.

Essentially, the preprocessing encompasses:
- Reclustering of loaded particles per jet with FastJet
- Obtaining per-jet declustering steps from the unique history order
- Tokenising particles and pseudojets with VQ-VAE
- Inserting the appropriate tokens into the declustering steps for each jet

### Tokenisation

The VQ-VAE encodes particle and pseudojet features into tokens which are used for training and inference of the generative model. The VQ-VAE model can be trained by running: 
```bash
python gabbro/train.py experiment=experiment_tokenization_reclustered_transformer
```
Changes were made to the data loading in a separate file: `gabbro/data/iterable_dataset_jetclass_recluster.py` to incorporate pseudojets in the tokenisation.

### Generative Model

The generative model `jet_evolution.py` was extended to possess two distinct heads for predicting tokens:

- Left head, making the first prediction for the left "child" of a particle decay
- Right head, making the second prediction for the right child, using cross-attention to condition its predictions on the left

Furthermore, since jet evolution in principle follows a sequence (from jet initiation to hadronisation), positional embeddings were included in the backbone model `gabbro/models/gpt_model.py`.

For batch generation, the heads may predict stop tokens (i.e. particles that don't split further) which need to be filtered out or handled properly by the VQ-VAE. For now, they are filtered out in `gabbro/data/data_tokenization.py`.

Optionally, to address imbalanced token distributions in tokenised data, weights can be loaded by passing a `token_weights_path` in the experiment's config file: `configs/experiment/example_experiment_jet_evolution.yaml`.

### Miscellaneous Changes

- Several config files were added for the new tokenisation strategy and generative model
- `utils/arrays.py` got an additional function for only applying cuts on particle data during preprocessing, leaving out the transformation. This could likely be done more smoothly.

## Citation

Not yet available.
