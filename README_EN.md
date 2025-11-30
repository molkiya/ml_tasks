# ITOIB - Educational Project

This repository contains work and completed educational tasks on machine learning and data processing.

**[Русская версия](README.md)**

## Description

The project is a collection of various assignments and experiments completed as part of an educational course. It includes implementations of various machine learning algorithms, data processing methods, and result visualization.

## Project Structure

```
ITOIB/
├── scripts/                    # Main scripts with educational tasks
│   ├── affinity_propagation_clusterization.py  # Clustering using Affinity Propagation
│   ├── fraud.py                # Credit card fraud detection
│   ├── LDA.py                  # Topic modeling and classification with LDA
│   ├── pca_classification.py   # Classification using PCA and SVD
│   ├── tree_classification.py  # Classification with decision tree
│   ├── single_perceptron_regression.py  # Regression with single-layer perceptron
│   └── metrics_visualizer.py   # Model metrics visualization
│
├── utils/                      # Helper utilities
│   ├── affinity_propagation.py
│   ├── decision_tree.py
│   ├── single_perceptron.py
│   ├── metrics.py
│   └── ...                     # Other utilities
│
├── data/                       # Training data
│   ├── fraud/                  # Fraud data
│   └── imdb/                   # IMDB data
│
├── models/                     # Saved models
├── logs/                       # Logs and result graphs
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

## Installation and Setup

1. Clone the repository or download the project
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # for Linux/Mac
   # or
   venv\Scripts\activate  # for Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

All main scripts are located in the `scripts/` folder. To run a script, use:

```bash
python scripts/script_name.py
```

### Running Examples:

- **Clustering**: `python scripts/affinity_propagation_clusterization.py`
- **Fraud Detection**: `python scripts/fraud.py`
- **LDA Classification**: `python scripts/LDA.py`
- **PCA Classification**: `python scripts/pca_classification.py`
- **Decision Tree**: `python scripts/tree_classification.py`
- **Perceptron Regression**: `python scripts/single_perceptron_regression.py`

## Task Contents

### 1. Clustering (Affinity Propagation)
- Clustering iris data with added noise
- Visualization of clustering results

### 2. Fraud Detection
- Analysis of credit card transactions
- Classification using GaussianNB and DecisionTree
- ROC curve construction and metrics analysis

### 3. Topic Modeling (LDA)
- Processing IMDB text data
- Training Word2Vec models (CBOW, Skip-gram)
- Classification using LDA

### 4. PCA Classification
- Application of PCA and Truncated SVD for dimensionality reduction
- Comparison of models with and without feature compression

### 5. Decision Tree
- Implementation of custom decision tree algorithm
- Classification on iris data

### 6. Perceptron Regression
- Time series forecasting
- Using different activation functions (tanh, sigmoid)

## Notes

- This is an educational project containing work and completed assignments
- Code may contain experimental solutions and unoptimized sections
- Some scripts require data to be present in the corresponding `data/` folders

## License

Educational project. Used for educational purposes.

