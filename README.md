# icc2023-qot-forecasting
Code repository for the paper "Uncertainty-Aware QoT Forecasting in Optical Networks with Bayesian Recurrent Neural Networks", Di Cicco N., Talpini J., et al., published at IEEE ICC 2023.

# Structure
The repository is structured as follows:
- `models/`: contains the PyTorch implementations of the Variational LSTM layers, the Bayesian Seq2Seq model and the MLP baseline.
- `common/`: contains utility functions for train/test splitting and data loading.

# Dataset
The dataset used in this paper is the publicly-available "Wide-Area Optical Backbone Performance" dataset, published in Ghobadi, M., Mahajan, R, "Optical Layer Failures in a Large Backbone", IMC'16

# Usage
The notebook lstm_seq2seq.ipynb loads the dataset and illustrates the usage of the Bayesian Seq2Seq model for a toy example.