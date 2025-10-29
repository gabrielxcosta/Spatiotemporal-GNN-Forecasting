# Spatiotemporal GNN Forecasting
**Exploring Neural Network Architectures for Time Series Prediction on Graphs**

This repository contains the implementation and experiments from my Master's research, which investigates how different spatial and temporal architectural modules in Graph Neural Networks (GNNs) affect forecasting performance over static and dynamic graphs.

---

## Abstract
The growing availability of time series data from domains such as epidemiological surveillance, smart cities, and environmental monitoring has increased the need for accurate forecasting models. Traditional approaches often fail to capture the complex spatial-temporal dependencies inherent in interconnected systems.

By representing time series as nodes in graphs—where relationships evolve over time—this project explores the integration of **spatial and temporal learning** using GNN-based architectures such as **DCRNN**, **STGCN**, **Graph WaveNet**, and **GMAN**.  
The study systematically compares architectures that combine **attention-based**, **recurrent**, and **convolutional** temporal modules, as well as different spatial embedding strategies.

---

## Project Structure

│  
├── main_experiments.py  
├── requirements.txt  
│  
├── 📂 models/  
│   ├── DCRNN.py  
│   ├── STGCN.py  
│   ├── GraphWaveNet.py  
│   └── GMAN.py  
│  
├── 📂 utils/  
│   ├── data_utils.py  
│   ├── gpu_utils.py  
│   ├── training_utils.py  
│   ├── plotting_utils.py  
│   ├── metrics_utils.py  
│   └── time_utils.py  
│  
├── 📂 results/  
│   ├── EnglandCovid_lags16/  
│   ├── EnglandCovid_lags20/  
│   └── comparativo_lags_rmse.png  
│  
└── 📄 README.md
