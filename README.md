# Data-Centric Machine Learning Experiments

This project investigates how preprocessing quality, representation design, and inductive bias influence predictive performance and robustness in tabular machine learning.

The project compares multiple machine learning models across a range of controlled degradation conditions, including:

- missing values,
- feature noise,
- irrelevant and redundant features,
- class imbalance,
- dataset size reduction,
- and computational efficiency.

The experiments are conducted using the Adult Income dataset and are structured as a reproducible and modular machine learning research framework.

---

# Project Goals

The primary goal of the project is to investigate whether improvements in data representation and preprocessing can significantly influence predictive robustness and reduce the apparent need for increasingly complex models.

The project focuses particularly on:

- preprocessing effectiveness,
- inductive bias,
- robustness,
- computational trade-offs,
- and representation quality.

---

# Models

The following models are evaluated throughout the experiments:

- Logistic Regression
- Decision Tree
- Naive Bayes
- Multi-Layer Perceptron (MLP)

---

# Experimental Conditions

The project evaluates model behavior under multiple controlled conditions:

| Experiment | Conditions |
|---|---|
| Missing Values | 10% missing, 30% missing |
| Noise | Low noise, High noise |
| Feature Relevance | Irrelevant features, Redundant features |
| Class Imbalance | Moderate imbalance, Severe imbalance |
| Dataset Size | Small, Medium, Large |
| Computational Cost | Training time, Prediction time, Efficiency ratio |

---

# Project Structure

```text
project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── results/
│   ├── metrics/
│   ├── plots/
│   └── tables/
│
├── src/
│   ├── data/
│   ├── experiments/
│   ├── models/
│   ├── visualization/
│   └── utils/
│
├── thesis/
│
├── main.py
│
└── requirements.txt
```

---

# Environment Setup

## 1. Clone Repository

```bash
git clone <repository-url>
cd project
```

---

## 2. Create Virtual Environment

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\\Scripts\\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset Setup

The project uses the Adult Income dataset.

## Automatic Download

Run:

```bash
python src/data/download_data.py
```

This creates:

```text
data/raw/adult.csv
```

---

# Running the Project

## Run Entire Experimental Pipeline

```bash
python main.py
```

This will:

- execute all notebooks,
- regenerate results,
- regenerate plots,
- regenerate summary tables,
- and rebuild the cross-experiment analysis.

---

# Processed Data

The `processed/` directory stores transformed datasets generated through the preprocessing pipeline.

Examples include:
- cleaned datasets,
- engineered datasets,
- encoded feature representations.

Processed datasets can be regenerated using:

```bash
python -m src.data.export_processed_data
```

---

# Notebooks

| Notebook | Description |
|---|---|
| 01 | Baseline experiment |
| 02 | Preprocessing comparison |
| 03 | Missing values experiment |
| 04 | Noise experiment |
| 05 | Feature relevance experiment |
| 06 | Class imbalance experiment |
| 07 | Dataset size experiment |
| 08 | Computational cost analysis |
| 09 | Cross-experiment analysis |

---

# Results

The generated outputs are automatically stored in:

```text
results/
```

including:

- metrics,
- plots,
- tables,
- and comparative experiment summaries.

---

# Research Focus

A central motivation of the project is the relationship between:

- representation quality,
- preprocessing,
- inductive bias,
- model complexity,
- and predictive robustness.

The project investigates whether carefully designed preprocessing pipelines can significantly improve robustness and reduce the need for substantially more complex learning architectures.

---

# Reproducibility

The project is designed to support reproducible experimentation through:

- centralized preprocessing,
- modular experiment pipelines,
- automated notebook execution,
- centralized result storage,
- and explicit dependency management.

---

# Author

Camille Kirstine Larsson

MSc Computer Science Student @ RUC

Data-centric machine learning and experimental robustness analysis.