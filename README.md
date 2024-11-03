# Federated Learning for Image Classification

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)

## Overview

This repository provides a Python-based implementation of Federated Learning for Image Classification, using neural networks with PyTorch. Federated learning enables collaborative model training across multiple devices (clients) without requiring data centralization, preserving privacy and maintaining data locality.

### Key Features
- **Federated Training**: Implements federated averaging for distributed model updates across clients.
- **Non-IID Data Handling**: Supports partitioning of data in both IID and non-IID configurations, providing realistic scenarios for federated learning.
- **Customizable Hyperparameters**: Control the number of clients, communication rounds, and other aspects of federated learning.

---

## Datasets

The following datasets are used in this project:
- **[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)**: Contains MRI images for tumor detection.
- **[COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)**: A set of chest X-ray images for COVID-19 detection.

### Dataset Details
- **COVID-19 Dataset**: Split into training and testing subsets using an approximate 70-30 ratio.
- **Federated Setup**: Each client is assigned a subset of the training data, emulating a real-world distributed data environment.

## Hyperparameters

The following hyperparameters can be adjusted to customize the federated learning process:

- **Number of Clients**: Determines the number of clients participating in federated training.
- **Number of Communication Rounds**: Specifies the number of rounds of communication between clients and the server.
- **Data Distribution**: Supports IID (independent and identically distributed) and non-IID data partitioning.
- **Beta Parameter**: Used for controlling the degree of data non-IID distribution when using Dirichlet distribution.

## Getting Started

### Prerequisites

- Python 3.7+
- Pandas
- PyTorch
- scikit-learn
- Matplotlib
- Seaborn

### Usage

#### Training Baseline Model
To train a baseline model without federated learning:

```bash
python baseline.py --dataset tumor
```

#### Training Federated Model
To initiate federated training:

```bash
python federated.py --dataset tumor --n_parties 10 --comm_round 10
```

Replace `tumor` with `covid` to use the COVID-19 dataset. Adjust `--n_parties` and `--comm_round` to set the number of clients and communication rounds, respectively.

### Related Projects
For a similar implementation of federated learning applied to Sentiment Analysis, please refer to [this repository](https://github.com/Atul-AI08/Federated-Learning-for-NLP-Tasks).