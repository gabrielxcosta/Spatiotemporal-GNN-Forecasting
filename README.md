# Spatiotemporal GNN Forecasting

Repository for the master's research project on benchmarking Spatio-Temporal Graph Neural Networks (STGNNs) for node-level time series forecasting over heterogeneous graph domains.

This repository supports the experiments developed for a master's dissertation in Computer Science at the Federal University of Ouro Preto (UFOP). The project also serves as the code base for a paper for BRACIS 2026 and for poster/oral presentations at Enredando 2026 and Encontro de Saberes 2026 at UFOP.

## Overview

The project evaluates STGNN architectures under a controlled and unified experimental pipeline. The main goal is to compare how different temporal modeling mechanisms affect forecasting performance across graph-based time series datasets.

The architectures are organized according to the temporal module used by each model:

* attention-based models;
* convolutional models;
* recurrent models.

The experiments include different datasets, temporal contexts, forecasting horizons, random seeds, input projection settings, computational time analysis, result aggregation, and graph spectral analysis.

## Repository Structure

```text
Spatiotemporal-GNN-Forecasting/
│
├── data/
│   ├── chickenpox.json
│   ├── england_covid.json
│   ├── montevideo_bus.json
│   ├── pedalme_london.json
│   ├── pems_bay_adj_mat.npy
│   ├── twitter_tennis_rg17.json
│   ├── twitter_tennis_uo17.json
│   └── wikivital_mathematics.json
│
├── loaders/
│   ├── chickenpox_loader.py
│   ├── englandcovid_loader.py
│   ├── montevideobus_loader.py
│   ├── pedalme_loader.py
│   ├── pemsbay_loader.py
│   ├── twittertennis_loader.py
│   └── wikimaths_loader.py
│
├── models/
│   ├── attention/
│   │   └── Attention-based STGNN architectures.
│   │
│   ├── convolutional/
│   │   └── Convolutional STGNN architectures.
│   │
│   └── recurrent/
│       └── Recurrent STGNN architectures.
│
├── utils/
│   ├── computational_time.py
│   ├── dataloaders.py
│   ├── metrics.py
│   ├── plotting.py
│   ├── results.py
│   ├── results_long.py
│   ├── training.py
│   └── utilities.py
│
├── main.py
│   └── Main script for running the controlled experiments.
│
├── get_the_graphml.py
│   └── Script for exporting or generating graph representations.
│
├── verify_datasets.py
│   └── Script for checking dataset loading and formatting.
│
└── README.md
```

## Model Taxonomy

The `models/` directory is divided according to the temporal modeling mechanism used by each architecture.

### Attention-based models

Located in:

```text
models/attention/
```

These models use attention or transformer-based mechanisms to capture temporal or spatio-temporal dependencies. This family includes architectures designed to model adaptive and non-local interactions over time, nodes, or both.

### Convolutional models

Located in:

```text
models/convolutional/
```

These models use temporal convolutions, graph convolutions, gated convolutions, dilated convolutions, or adaptive graph convolutional mechanisms. This family represents architectures based mainly on local filtering and convolutional propagation.

### Recurrent models

Located in:

```text
models/recurrent/
```

These models use recurrent mechanisms such as GRU or LSTM combined with graph convolution, diffusion convolution, message passing, or dynamic graph updates. This family represents architectures that propagate temporal information through hidden states.

## Datasets

The repository includes loaders and data files for multiple graph-based time series datasets, including:

* Chickenpox Hungary;
* England COVID-19;
* Montevideo Bus;
* WikiMaths / WikiVital Mathematics;
* PedalMe London;
* PeMS-Bay;
* Twitter Tennis datasets.

The main experiments reported in the dissertation and paper focus on a controlled benchmark over selected heterogeneous datasets.

## Experimental Pipeline

The experiments follow a unified setup:

* chronological train/validation/test split;
* sliding-window forecasting formulation;
* multiple temporal context lengths;
* multiple forecasting horizons;
* multiple seeds;
* evaluation with RMSE, MAE, MSE, and R²;
* comparison between architectural families;
* input projection analysis;
* computational time analysis;
* graph spectral analysis.

The `main.py` file is the main entry point for running the controlled experiments.

## Spectral Analysis

The spectral analysis is computed from the graph structure and graph signals. It is not separated by forecasting regime because the graph spectrum characterizes the dataset/graph itself, not a specific Short-Mid or Long-Range configuration.

The spectral analysis is used to describe properties such as low-, mid-, and high-frequency energy and Dirichlet energy. These descriptors are then compared with the behavior of different STGNN families.

## Results

The repository includes utilities for processing experimental results in different forecasting regimes.

Relevant files include:

```text
utils/results.py
utils/results_long.py
utils/computational_time.py
utils/plotting.py
```

These scripts are used to aggregate metrics, compare architectures, analyze training time, and generate plots or tables for the dissertation and paper.

## Running the Experiments

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the main experimental pipeline:

```bash
python main.py
```

Additional scripts may be used for dataset verification, graph export, result processing, computational time analysis, and spectral analysis.

## Reproducibility

To ensure reproducibility, the experiments are controlled by:

* fixed chronological data splits;
* fixed seeds;
* standardized preprocessing;
* common training protocol;
* shared evaluation metrics;
* consistent model grouping by temporal module.

## Authors

Gabriel F. Costa¹, Vander L. S. Freitas¹, Eduardo J. S. Luz¹

¹ Federal University of Ouro Preto (UFOP)<br>
CEP: 35400-000 -- Ouro Preto -- MG -- Brazil<br>
PPGCC -- Postgraduate Program in Computer Science<br>
secretaria.ppgcc@ufop.edu.br<br>
http://www3.decom.ufop.br/pos/inicio/

Repository: https://github.com/gabrielxcosta/Spatiotemporal-GNN-Forecasting

## License

This repository is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this repository, please cite it using the metadata available in `CITATION.cff`.
