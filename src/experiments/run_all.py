from pathlib import Path

import subprocess


NOTEBOOKS = [

    "01_baseline_experiment.ipynb",

    "02_preprocessing_levels.ipynb",

    "03_missing_values_experiment.ipynb",

    "04_noise_experiment.ipynb",

    "05_feature_relevance_experiment.ipynb",

    "06_class_imbalance_experiment.ipynb",

    "07_dataset_size_experiment.ipynb",

    "08_computational_cost_analysis.ipynb",

    "09_cross_experiment_analysis.ipynb"
]


def run_all_notebooks():
    """
    Execute all experiment notebooks.
    """

    notebook_dir = Path(
        "notebooks"
    )

    for notebook in NOTEBOOKS:

        notebook_path = (
            notebook_dir / notebook
        )

        print(
            f"\nRunning: {notebook}"
        )

        subprocess.run([
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            str(notebook_path)
        ])


if __name__ == "__main__":
    run_all_notebooks()