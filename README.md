# Spatiotemporal GNN Forecasting

Research repository for benchmarking **Spatio-Temporal Graph Neural Networks (STGNNs)** for node-level time series forecasting across heterogeneous graph domains.

This repository contains the experimental framework developed during the master's research in Computer Science at the **Federal University of Ouro Preto (UFOP)**. It is the official code repository associated with the BRACIS 2026 paper **“Benchmarking Spatio-Temporal Graph Neural Networks for Time Series Forecasting Across Heterogeneous Graph Domains”** and is being extended for the master's dissertation with additional datasets, baselines, temporal characterization, graph structural and spectral analysis, statistical comparisons, and forecasting regimes.

## Published Paper

**Gabriel F. Costa, Eduardo J. S. Luz, Vander L. S. Freitas**  
**Benchmarking Spatio-Temporal Graph Neural Networks for Time Series Forecasting Across Heterogeneous Graph Domains**

- Conference: **Brazilian Conference on Intelligent Systems (BRACIS 2026)**
- Proceedings: **Intelligent Systems**
- Series: **Lecture Notes in Computer Science (LNCS/LNAI)**
- Volume: **17106**
- Pages: **19–33**
- Publisher: **Springer, Cham**
- First online: **2 October 2026**
- DOI: [10.1007/978-3-032-39892-5_2](https://doi.org/10.1007/978-3-032-39892-5_2)
- Springer: [https://link.springer.com/chapter/10.1007/978-3-032-39892-5_2](https://link.springer.com/chapter/10.1007/978-3-032-39892-5_2)

The published study systematically evaluates **21 STGNN architectures** using a unified experimental pipeline across heterogeneous graph domains. The architectures are grouped according to their main temporal modeling mechanism: recurrent, convolutional, and attention-based.

The current repository goes beyond the original BRACIS experiment and supports the ongoing extended benchmark developed for the master's dissertation.

---

## Research Goals

The main objective is to understand how the interaction between **temporal dynamics, graph topology, graph spectra, forecasting context, and architectural design** affects the performance of STGNNs.

The current research framework includes:

- benchmarking of 21 STGNN architectures;
- comparison of recurrent, convolutional, and attention-based temporal mechanisms;
- graph-free forecasting baselines;
- heterogeneous graph domains;
- short-to-mid-range and long-range forecasting;
- scalar and lagged input representations;
- input-projection ablation experiments;
- multiple random seeds;
- multiple hidden dimensions;
- chronological train/validation/test splits;
- computational-time analysis;
- temporal characterization of graph signals;
- graph structural characterization;
- graph spectral characterization;
- graph-signal frequency analysis;
- statistical comparison of architectures and architectural families;
- Critical Difference diagrams;
- analysis of context length and forecasting horizon;
- network visualization;
- reproducible result auditing.

---

## Model Taxonomy

The benchmark currently contains **21 STGNN architectures**, grouped by the dominant temporal mechanism.

### Attention-based Models

| Model | Implementation |
|---|---|
| CaST | `models/attention/cast.py` |
| GMAN | `models/attention/gman.py` |
| STAEformer | `models/attention/staeformer.py` |
| STGNN | `models/attention/stgnn.py` |
| STGraformer | `models/attention/stgraformer.py` |
| TGAT | `models/attention/tgat.py` |

These architectures use attention, transformer-like mechanisms, adaptive embeddings, spectral components, or non-local interactions to represent temporal and/or spatio-temporal dependencies.

### Convolutional Models

| Model | Implementation |
|---|---|
| AAGCN | `models/convolutional/aagcn.py` |
| GraphWaveNet | `models/convolutional/graph_wavenet.py` |
| LSGCN | `models/convolutional/lsgcn.py` |
| MTGNN | `models/convolutional/mtgnn.py` |
| SLCNN | `models/convolutional/slcnn.py` |
| STGCN | `models/convolutional/stgcn.py` |

These models combine graph operations with temporal convolutions, gated convolutions, dilated convolutions, adaptive graph learning, or related local filtering mechanisms.

### Recurrent Models

| Model | Implementation |
|---|---|
| DCRNN | `models/recurrent/dcrnn.py` |
| DyGrAE | `models/recurrent/dygrae.py` |
| EvolveGCN-O | `models/recurrent/egcno.py` |
| EvolveGCN-H | `models/recurrent/egcnh.py` |
| GCLSTM | `models/recurrent/gclstm.py` |
| GConvGRU | `models/recurrent/gconvgru.py` |
| GConvLSTM | `models/recurrent/gconvlstm.py` |
| MPNNLSTM | `models/recurrent/mpnnlstm.py` |
| TGCN | `models/recurrent/tgcn.py` |

These architectures propagate temporal information through recurrent hidden states combined with graph convolution, diffusion, message passing, or evolving graph representations.

### No-Input-Projection Variants

The trainable architectures also contain paired `_w.py` implementations used by the `noipj-long` regime.

Examples:

```text
models/attention/gman.py
models/attention/gman_w.py

models/convolutional/stgcn.py
models/convolutional/stgcn_w.py

models/recurrent/dcrnn.py
models/recurrent/dcrnn_w.py
```

These variants support the long-range **input-projection ablation**, using parameter-free feature alignment where required instead of the standard learned input projection.

---

## Baselines

The extended benchmark includes four graph-free or non-STGNN baselines:

| Baseline | Type |
|---|---|
| Persistence | Deterministic |
| Seasonal Persistence | Deterministic |
| Least Squares | Deterministic |
| xLSTM | Trainable |

Implementations are available in:

```text
models/baseline/
├── least_squares.py
├── persistence.py
├── seasonal_persistence.py
├── xLSTM.py
└── xLSTM_w.py
```

`Persistence`, `SeasonalPersistence`, and `LeastSquares` are analytical/deterministic baselines and therefore do not require repeated random seeds.

`xLSTM` follows the same multi-seed experimental framework used for the trainable neural architectures.

---

## Datasets

The research has evolved from the four-dataset benchmark reported in the BRACIS paper toward a substantially broader heterogeneous evaluation.

The current descriptive temporal, structural, and spectral analysis covers the following graph-based datasets:

| Dataset | Domain |
|---|---|
| Chickenpox Hungary | Epidemiology |
| WikiMaths / WikiVital Mathematics | Online knowledge / page activity |
| England COVID-19 | Epidemiology |
| Montevideo Bus | Transportation |
| PedalMe London | Urban mobility |
| Twitter Tennis RG17 | Social / interaction network |
| Twitter Tennis UO17 | Social / interaction network |
| PeMS-Bay | Traffic |
| AQI36 | Air quality |
| AQI437 | Air quality |
| RioNegro | Hydrology |
| Grid2Op | Power systems |

The unified forecasting runner also contains entries for the ongoing **Windmill Output** experiments at multiple graph sizes:

```text
windmill_small
windmill_medium
windmill_large
```

### Data Currently Versioned in the Repository

```text
data/
├── AQI36.h5
├── AQI437.h5
├── AQI_dist.npy
├── chickenpox.json
├── england_covid.json
├── grid2op_ieee11.json
├── montevideo_bus.json
├── pedalme_london.json
├── pems_bay_adj_mat.npy
├── twitter_tennis_rg17.json
├── twitter_tennis_uo17.json
└── wikivital_mathematics.json
```

Some large or externally distributed raw files are intentionally not versioned directly in the repository.

For example:

```text
data/pems_bay_node_values.npy
```

is excluded through `.gitignore`, while the PeMS-Bay adjacency matrix is stored in the repository.

Other loaders may expect additional local data files depending on the experiment being executed.

---

## Dataset Loaders

Dataset-specific preprocessing is organized under:

```text
loaders/
├── aqi_loader.py
├── chickenpox_loader.py
├── englandcovid_loader.py
├── grid2op_loader.py
├── montevideobus_loader.py
├── pedalme_loader.py
├── pemsbay_loader.py
├── rionegro_loader.py
├── twittertennis_loader.py
└── wikimaths_loader.py
```

The loaders standardize the datasets into graph-temporal representations compatible with the unified forecasting pipeline.

The current training runner also contains configuration entries for the Windmill variants used in the ongoing extended experiments.

---

## Forecasting Formulation

The benchmark treats forecasting as a node-level multi-step prediction problem.

Given a graph

\[
\mathcal{G}=(\mathcal{V},\mathcal{E}),
\]

and graph signals observed over a temporal context of length \(L\), the models predict the next \(H\) observations for the graph nodes.

The experimental framework separates:

- **graph structure**;
- **temporal context \(L\)**;
- **forecasting horizon \(H\)**;
- **input representation**;
- **model family**;
- **hidden dimension**;
- **random seed**.

---

## Input Representations

Two representations are currently supported.

### Scalar Representation

Each temporal snapshot contains one feature per node:

```text
(batch, lags, nodes, 1)
```

This representation preserves the explicit temporal dimension while each node carries a scalar observation at each time step.

### Lagged Representation

Each snapshot contains a lagged feature vector:

```text
(batch, lags, nodes, lags)
```

For an external context \(L\), this construction exposes an effective temporal support of up to:

\[
2L-1
\]

observations.

The comparison between `scalar` and `lagged` representations is part of the current experimental investigation into how temporal information is exposed to STGNN architectures.

---

## Forecasting Regimes

The current pipeline defines three main regimes.

### Short-Mid Range

For most datasets:

```text
L = [2, 4, 8, 12]
H = [1, 5, 10]
```

PedalMe uses a shorter feasible context grid:

```text
L = [2, 4, 6, 8]
H = [1, 5, 10]
```

### Long Range

The default long-range configuration is:

```text
L = 50
H = 20
```

Dataset-specific long-range configurations are used when required by series length:

```text
EnglandCOVID:
L = 20
H = 10

PedalMe:
L = 9
H = 10

TwitterTennis RG17:
L = 20
H = 20

TwitterTennis UO17:
L = 20
H = 20
```

### Long Range Without Input Projection

The `noipj-long` regime uses the same long-range temporal configuration while replacing the standard trainable model modules with their `_w` variants.

This allows the effect of the initial learned input projection to be isolated from the remainder of the architecture.

Explicit `--lags` and `--horizons` arguments can override the default grids.

---

## Unified Experimental Pipeline

The main current experiment runner is:

```text
main_family.py
```

It provides a common execution framework across model families, datasets, representations, forecasting regimes, configurations, and random seeds.

The default trainable-model configuration is:

```text
Seeds:          10 (0–9)
Hidden sizes:   [32, 64]
Learning rate:  1e-3
Batch size:     32
Dropout:        0.2
Edge dropout:   0.1
Epochs:         200
Warmup:         5 epochs
Patience:       20 epochs
```

Training uses:

- AdamW;
- weight decay of `1e-4`;
- learning-rate warmup;
- cosine annealing;
- validation-based early stopping;
- restoration of the best validation state.

### Chronological Split

Forecasting windows are split chronologically into:

```text
Training:    70%
Validation:  15%
Test:        15%
```

Training batches are shuffled after the chronological split.

---

## Evaluation Metrics

Each experiment stores the following predictive metrics:

- MSE;
- RMSE;
- MAE;
- MAPE;
- global \(R^2\);
- mean \(R^2\) across forecasting horizons;
- \(R^2\) for each individual forecasting horizon.

Additional metadata includes:

- seed;
- dataset;
- input representation;
- number of input channels;
- model implementation;
- training configuration;
- runtime;
- number of epochs executed;
- graph policy;
- edge-weight policy.

The principal error metric used throughout the current comparative study is **RMSE**.

---

## Graph Handling

The unified training pipeline extracts the graph representation from the dataset and constructs a shared adjacency representation for model execution.

The current benchmark records the graph policy as:

```text
first_snapshot
```

For datasets with temporal edge information, the forecasting runner currently uses the graph from the first snapshot during model training.

For TwitterTennis, values are transformed with `log1p`, while the graph is fixed from the first snapshot in the current forecasting benchmark.

For PeMS-Bay, the prediction pipeline uses the physical channel corresponding to traffic speed.

When explicit edge weights are unavailable, unit edge weights are used.

---

## Missing-Value Handling

Datasets exposing observation masks can be represented with masked arrays.

For masked datasets, evaluation metrics are computed only over observed targets.

Normalization metadata is preserved with the results when supplied by the dataset loader, allowing consistency checks when existing experiment results are reused.

---

## Temporal Characterization

The extended dissertation experiments include a dedicated temporal characterization pipeline:

```text
utils/time_series_analysis.py
```

The current implementation analyzes twelve datasets without training forecasting models.

The characterization includes:

- Autocorrelation Function (ACF);
- Partial Autocorrelation Function (PACF);
- Average Mutual Information (AMI);
- Hurst exponent;
- Detrended Fluctuation Analysis (DFA);
- Welch power spectrum;
- spectral entropy;
- permutation/ordinal entropy;
- Augmented Dickey-Fuller test;
- KPSS test;
- exploratory Largest Lyapunov Exponent analysis;
- multivariate first-order relaxation analysis.

The relaxation analysis includes descriptors such as:

```text
relax_spectral_radius
relax_tau_max
relax_tau_median
relax_stable_fraction
```

These descriptors are used to characterize temporal persistence, dependence scales, memory, stationarity, complexity, and effective dynamical behavior across heterogeneous graph domains.

The temporal descriptors are intended as **dataset characterization variables**, not as automatic rules for selecting forecasting contexts.

---

## Graph Structural and Spectral Analysis

The current structural and spectral characterization pipeline is:

```text
utils/graph_spectra_analysis.py
```

It analyzes graph structure and graph signals independently of forecasting-model training.

The analysis uses the graph topology, an undirected analysis view where required, and the normalized graph Laplacian.

The current analysis includes structural descriptors such as:

- number of nodes;
- number of edges;
- degree statistics;
- strength statistics;
- connected components;
- largest connected component;
- diameter;
- average shortest-path behavior;
- global efficiency;
- clustering;
- transitivity;
- assortativity.

The spectral characterization investigates quantities derived from the graph Laplacian and graph signals, including spectral structure, graph-frequency behavior, smoothness, and the distribution of signal energy across graph frequencies.

A previous spectral-analysis implementation is also retained at:

```text
spectral_analysis.py
```

for reproducibility of earlier stages of the project.

---

## Graph-Signal Frequency Analysis

The project studies graph signals through the graph Fourier perspective.

The signal energy is separated into approximately:

- low graph frequencies;
- intermediate graph frequencies;
- high graph frequencies.

Additional quantities such as Dirichlet energy, spectral concentration, spectral entropy, and related smoothness descriptors are used to characterize how node signals vary over the graph structure.

These graph descriptors are subsequently compared with the predictive behavior of different STGNN architectural families.

---

## Statistical Analysis

The repository includes utilities for aggregating results across architectures, datasets, forecasting configurations, and seeds.

Current statistical procedures include:

- average ranks;
- Friedman tests;
- Nemenyi post-hoc comparisons;
- Critical Difference diagrams;
- Wilcoxon-based pairwise analyses in the extended result scripts;
- Pearson correlation;
- Spearman correlation;
- architectural-family comparisons;
- context/horizon comparisons.

Relevant files include:

```text
utils/results.py
utils/results_long.py
utils/results_x.py
utils/results_long_x.py
utils/context_horizon_reporting.py
```

The goal is to distinguish descriptive performance differences from statistically supported differences across repeated experimental configurations.

---

## Context and Horizon Analysis

The project explicitly investigates how architectural behavior changes with:

- context length \(L\);
- forecasting horizon \(H\);
- temporal representation;
- architectural family.

The reporting utilities compare family-level performance gaps such as:

```text
Attention - Convolution
Recurrent - Convolution
```

across datasets and forecasting configurations.

This supports the broader research question of whether particular temporal mechanisms become more or less effective as the forecasting dependency range changes.

---

## Computational-Time Analysis

Runtime is stored for each experiment and can be aggregated independently of predictive performance.

Relevant utilities include:

```text
utils/computational_time.py
time_cpx.py
```

This enables comparison of predictive performance against computational cost across models, families, regimes, and datasets.

---

## Result Auditing

Experiment completeness can be checked with:

```text
utils/check_results.py
```

The auditing utility verifies `metrics.json` files across:

- datasets;
- models;
- contexts;
- horizons;
- hidden dimensions;
- seeds;
- forecasting regimes.

It detects missing or invalid results before statistical aggregation.

---

## GPU Scheduling

Large experiment grids can be scheduled through:

```text
utils/gpu_queue.py
```

The scheduler executes experiments in isolated processes and can:

- inspect available GPU memory using `nvidia-smi`;
- control global concurrency;
- associate jobs with visible GPUs;
- retry jobs after CUDA out-of-memory failures;
- maintain per-job logs;
- record pending, running, completed, failed, interrupted, and OOM states.

The scheduler does not replace model-level checkpointing. A failed training run is restarted from the beginning.

---

## Result Storage

Results are organized hierarchically by:

```text
dataset
└── architecture
    └── configuration
        └── seed
            └── metrics.json
```

Depending on representation and regime, directories follow conventions such as:

```text
results_<dataset>_scalar/
results_<dataset>/
results_<dataset>_scalar_long/
results_<dataset>_long/
results_<dataset>_scalar_noipj_long/
results_<dataset>_noipj_long/
```

When plotting is enabled, experiments may additionally generate:

```text
loss_curve.png
regression.png
temporal.png
```

An existing valid experiment directory containing `metrics.json` is reused by the runner to avoid unnecessary retraining.

The current pipeline stores metrics and figures but does not use epoch-level checkpoints to resume interrupted training.

---

## Network Visualization

Network visualization is part of the extended dataset characterization.

The combined visualization utility is:

```text
utils/network_visualization.py
```

The intended benchmark figure contains the heterogeneous graph domains in a common layout and supports individual debug plots for dataset-level inspection.

The visualization framework is designed to represent graph topology together with normalized node centrality information.

---

## Repository Structure

```text
Spatiotemporal-GNN-Forecasting/
│
├── BRACIS_2026/
│   └── 2026_BRACIS_Gabriel.zip
│
├── ENREDANDO_2026/
│   ├── Contributions.png
│   ├── Gap_and_Methodology.png
│   ├── Lightning Talk - Gabriel F. Costa.pdf
│   ├── poster_Gabriel_ENREDANDO_2026.pdf
│   └── qrcode_spatiotemporal_gnn_forecasting.png
│
├── master_thesis/
│   └── Dissertação_DECOM_Gabriel_2026.zip
│
├── data/
│   ├── AQI36.h5
│   ├── AQI437.h5
│   ├── AQI_dist.npy
│   ├── chickenpox.json
│   ├── england_covid.json
│   ├── grid2op_ieee11.json
│   ├── montevideo_bus.json
│   ├── pedalme_london.json
│   ├── pems_bay_adj_mat.npy
│   ├── twitter_tennis_rg17.json
│   ├── twitter_tennis_uo17.json
│   └── wikivital_mathematics.json
│
├── loaders/
│   ├── aqi_loader.py
│   ├── chickenpox_loader.py
│   ├── englandcovid_loader.py
│   ├── grid2op_loader.py
│   ├── montevideobus_loader.py
│   ├── pedalme_loader.py
│   ├── pemsbay_loader.py
│   ├── rionegro_loader.py
│   ├── twittertennis_loader.py
│   └── wikimaths_loader.py
│
├── models/
│   │
│   ├── attention/
│   │   ├── cast.py
│   │   ├── cast_w.py
│   │   ├── gman.py
│   │   ├── gman_w.py
│   │   ├── staeformer.py
│   │   ├── staeformer_w.py
│   │   ├── stgnn.py
│   │   ├── stgnn_w.py
│   │   ├── stgraformer.py
│   │   ├── stgraformer_w.py
│   │   ├── tgat.py
│   │   └── tgat_w.py
│   │
│   ├── convolutional/
│   │   ├── aagcn.py
│   │   ├── aagcn_w.py
│   │   ├── graph_wavenet.py
│   │   ├── graph_wavenet_w.py
│   │   ├── lsgcn.py
│   │   ├── lsgcn_w.py
│   │   ├── mtgnn.py
│   │   ├── mtgnn_w.py
│   │   ├── slcnn.py
│   │   ├── slcnn_w.py
│   │   ├── stgcn.py
│   │   └── stgcn_w.py
│   │
│   ├── recurrent/
│   │   ├── dcrnn.py
│   │   ├── dcrnn_w.py
│   │   ├── dygrae.py
│   │   ├── dygrae_w.py
│   │   ├── egcnh.py
│   │   ├── egcnh_w.py
│   │   ├── egcno.py
│   │   ├── egcno_w.py
│   │   ├── gclstm.py
│   │   ├── gclstm_w.py
│   │   ├── gconvgru.py
│   │   ├── gconvgru_w.py
│   │   ├── gconvlstm.py
│   │   ├── gconvlstm_w.py
│   │   ├── mpnnlstm.py
│   │   ├── mpnnlstm_w.py
│   │   ├── tgcn.py
│   │   └── tgcn_w.py
│   │
│   └── baseline/
│       ├── least_squares.py
│       ├── persistence.py
│       ├── seasonal_persistence.py
│       ├── xLSTM.py
│       └── xLSTM_w.py
│
├── utils/
│   ├── check_results.py
│   ├── computational_time.py
│   ├── context_horizon_reporting.py
│   ├── dataloaders.py
│   ├── gpu_queue.py
│   ├── graph_spectra_analysis.py
│   ├── input_features.py
│   ├── metrics.py
│   ├── network_visualization.py
│   ├── plotting.py
│   ├── results.py
│   ├── results_long.py
│   ├── results_long_x.py
│   ├── results_x.py
│   ├── time_series_analysis.py
│   ├── training.py
│   └── utilities.py
│
├── network_analysis_and_EDA.ipynb
│
├── main.py
│   └── Earlier experimental runner retained for reproducibility.
│
├── main_family.py
│   └── Current unified STGNN experimental runner.
│
├── main_family_baselines.py
│   └── Dedicated runner for the four forecasting baselines.
│
├── results_analysis.py
│   └── Short/mid-range result-analysis entry point.
│
├── results_analysis_long.py
│   └── Long-range result-analysis entry point.
│
├── spectral_analysis.py
│   └── Earlier graph spectral-analysis pipeline.
│
├── time_cpx.py
│   └── Computational-time analysis entry point.
│
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Frameworks and Dependencies

The project is primarily built on the PyTorch graph-learning ecosystem.

Core technologies include:

- Python;
- PyTorch;
- PyTorch Geometric;
- PyTorch Geometric Temporal;
- NumPy;
- SciPy;
- pandas;
- NetworkX;
- scikit-learn;
- statsmodels;
- scikit-posthocs;
- Matplotlib;
- Joblib.

The development environment currently uses:

```text
PyTorch 2.4
CUDA 12.1
PyTorch Geometric 2.4
PyTorch Geometric Temporal 0.54
```

The complete environment is specified in:

```text
requirements.txt
```

Because the requirements include CUDA-specific PyTorch and PyG packages, installation may require the appropriate PyTorch/PyG wheel repositories for the target CUDA environment.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/gabrielxcosta/Spatiotemporal-GNN-Forecasting.git
cd Spatiotemporal-GNN-Forecasting
```

Create and activate a virtual environment if desired.

Install the dependencies:

```bash
pip install -r requirements.txt
```

For GPU execution, make sure the installed PyTorch and PyTorch Geometric packages are compatible with the local CUDA version.

---

## Running the Main Benchmark

The current unified experimental runner is:

```bash
python3 -u main_family.py
```

### Example: Recurrent Models on WikiMaths

```bash
python3 -u main_family.py \
    --dataset wikimaths \
    --family recurrent \
    --regime short-mid \
    --representation scalar
```

### Example: Long-Range Experiment

```bash
python3 -u main_family.py \
    --dataset wikimaths \
    --family attention \
    --regime long \
    --representation lagged
```

### Example: Input-Projection Ablation

```bash
python3 -u main_family.py \
    --dataset wikimaths \
    --family recurrent \
    --regime noipj-long \
    --representation scalar \
    --no-plots
```

### Run Individual Models

Models can be selected explicitly instead of selecting an entire family:

```bash
python3 -u main_family.py \
    --dataset pemsbay \
    --models DCRNN MPNNLSTM STGCN GMAN \
    --regime long \
    --representation scalar
```

### CPU Execution

CUDA is the default device for the main experimental runner.

To explicitly use the CPU:

```bash
python3 -u main_family.py \
    --dataset chickenpox \
    --family recurrent \
    --regime short-mid \
    --representation scalar \
    --device cpu
```

---

## Running the Baselines

Use:

```bash
python3 -u main_family_baselines.py
```

Example:

```bash
python3 -u main_family_baselines.py \
    --dataset wikimaths \
    --regime all \
    --representation all
```

Only analytical baselines:

```bash
python3 -u main_family_baselines.py \
    --dataset pemsbay \
    --models Persistence SeasonalPersistence LeastSquares \
    --regime short-mid \
    --representation scalar \
    --no-plots
```

xLSTM over a specific seed interval:

```bash
python3 -u main_family_baselines.py \
    --dataset aqi36 \
    --models xLSTM \
    --regime long \
    --representation lagged \
    --seed-start 3 \
    --seed-end 5
```

---

## Temporal Dataset Analysis

Run the complete temporal characterization with:

```bash
python3 -u utils/time_series_analysis.py --workers 4
```

Analyze a specific dataset:

```bash
python3 -u utils/time_series_analysis.py \
    --datasets PeMS-Bay \
    --workers 4
```

The default outputs are written under:

```text
datasets_time_series_analysis/
├── figures/
├── tables/
└── metadata/results files
```

---

## Structural and Spectral Graph Analysis

Use:

```bash
python3 -u utils/graph_spectra_analysis.py --help
```

This pipeline analyzes graph structure and graph-signal spectral characteristics without training STGNN models.

---

## Checking Experimental Results

Use:

```bash
python3 -u utils/check_results.py --help
```

This is useful before running statistical aggregation or generating final result tables.

---

## Exploratory Data Analysis

The repository also contains:

```text
network_analysis_and_EDA.ipynb
```

which is used for exploratory graph and time-series analysis during dataset inspection and experimental development.

---

## Reproducibility

The benchmark is designed around a controlled experimental protocol.

The main reproducibility mechanisms include:

- shared dataset loaders;
- chronological splits;
- standardized temporal windows;
- explicit forecasting horizons;
- fixed random seeds;
- repeated experiments;
- shared hyperparameter grids;
- common metrics;
- common training utilities;
- consistent family taxonomy;
- deterministic analytical baselines;
- explicit scalar/lagged representations;
- explicit input-projection ablation;
- machine-readable `metrics.json` outputs;
- experiment auditing;
- automatic reuse of completed experiments.

The experiment runner records enough configuration metadata to identify the dataset, architecture, representation, context, horizon, hidden dimension, seed, runtime, and optimization configuration associated with each result.

---

## Research Outputs

The repository supports several outputs associated with the project.

### BRACIS 2026

The work resulted in the published paper:

> Gabriel F. Costa, Eduardo J. S. Luz, and Vander L. S. Freitas.  
> **Benchmarking Spatio-Temporal Graph Neural Networks for Time Series Forecasting Across Heterogeneous Graph Domains.**  
> Intelligent Systems, BRACIS 2026, Lecture Notes in Computer Science, vol. 17106, Springer.

DOI: [10.1007/978-3-032-39892-5_2](https://doi.org/10.1007/978-3-032-39892-5_2)

Materials associated with the conference submission are stored in:

```text
BRACIS_2026/
```

### Enredando 2026

Poster, lightning-talk, methodology, contribution, and QR-code materials are available under:

```text
ENREDANDO_2026/
```

### Master's Dissertation

Dissertation-related material is stored under:

```text
master_thesis/
```

The current dissertation extends the original BRACIS benchmark toward a broader analysis of heterogeneous datasets, forecasting regimes, temporal characteristics, graph structure, graph spectra, graph-signal frequency behavior, baselines, and architecture–dataset relationships.

---

## Authors

**Gabriel F. Costa**  
**Eduardo J. S. Luz**  
**Vander L. S. Freitas**

Federal University of Ouro Preto — UFOP  
Postgraduate Program in Computer Science — PPGCC  
Ouro Preto, Minas Gerais, Brazil

Repository:  
[https://github.com/gabrielxcosta/Spatiotemporal-GNN-Forecasting](https://github.com/gabrielxcosta/Spatiotemporal-GNN-Forecasting)

---

## Citation

If you use this repository or the associated benchmark, please cite the published paper.

```bibtex
@inproceedings{costa2027benchmarking,
  author    = {Costa, Gabriel F. and
               Luz, Eduardo J. S. and
               Freitas, Vander L. S.},
  title     = {Benchmarking Spatio-Temporal Graph Neural Networks for
               Time Series Forecasting Across Heterogeneous Graph Domains},
  booktitle = {Intelligent Systems},
  series    = {Lecture Notes in Computer Science},
  volume    = {17106},
  pages     = {19--33},
  publisher = {Springer},
  address   = {Cham},
  year      = {2027},
  doi       = {10.1007/978-3-032-39892-5_2},
  note      = {BRACIS 2026; first online 2 October 2026}
}
```

The repository also contains citation metadata in:

```text
CITATION.cff
```

### PyTorch Geometric Temporal

Several datasets, interfaces, and implementation ideas build upon the PyTorch Geometric Temporal ecosystem.

```bibtex
@inproceedings{rozemberczki2021pytorch,
  title     = {PyTorch Geometric Temporal: Spatiotemporal Signal Processing
               with Neural Machine Learning Models},
  author    = {Rozemberczki, Benedek and
               Davies, Ryan and
               Sarkar, Rik and
               Sutton, Charles},
  booktitle = {Proceedings of the 30th ACM International Conference on
               Information and Knowledge Management},
  pages     = {4564--4573},
  year      = {2021},
  doi       = {10.1145/3459637.3482014}
}
```

---

## License

This repository is distributed under the **MIT License**.

See:

```text
LICENSE
```

for details.

---

## Acknowledgments

This work builds upon the open-source **PyTorch**, **PyTorch Geometric**, and **PyTorch Geometric Temporal** ecosystems.

We acknowledge the developers and research communities responsible for these libraries and benchmark datasets, which provide essential infrastructure for reproducible research in graph machine learning and spatio-temporal forecasting.

This study was financed in part by the **Coordenação de Aperfeiçoamento de Pessoal de Nível Superior — Brasil (CAPES) — Finance Code 001**.
