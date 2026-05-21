# Data Directory

This directory contains the datasets used throughout the project.

## Structure

```text
data/
├── raw/
└── processed/
```

---

# Raw Data

The `raw/` directory stores the original source dataset.

The project uses the Adult Income dataset from OpenML / UCI.

The raw dataset is intentionally excluded from version control through `.gitignore`.

To regenerate the dataset locally:

```bash
python src/data/download_data.py
```

This creates:

```text
data/raw/adult.csv
```

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

# Purpose

Separating raw and processed data improves:
- reproducibility,
- pipeline clarity,
- transparency,
- and experimental consistency.