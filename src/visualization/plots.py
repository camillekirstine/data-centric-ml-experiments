import matplotlib.pyplot as plt

from src.utils.saving import (
    save_plot
)


def plot_f1_comparison(
    results_df,
    title,
    filename
):
    """
    Create grouped bar plot comparing
    F1-scores across preprocessing levels.
    """
    
    pivot_df = results_df.pivot_table(
        index="model",
        columns="preprocessing",
        values="f1_score",
        aggfunc="mean"
    )
    
    pivot_df.plot(
        kind="bar",
        figsize=(10, 6)
    )
    
    plt.title(title)
    
    plt.ylabel("F1-Score")
    
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    
    save_plot(
        plt,
        filename
    )
    
    plt.show()


def plot_efficiency_ratio(
    results_df,
    filename
):
    """
    Plot computational efficiency ratio.
    """
    
    pivot_df = results_df.pivot_table(
        index="model",
        columns="preprocessing",
        values="efficiency_ratio",
        aggfunc="mean"
    )
    
    pivot_df.plot(
        kind="bar",
        figsize=(10, 6)
    )
    
    plt.title(
        "Efficiency Ratio Across Models"
    )
    
    plt.ylabel("Efficiency Ratio")
    
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    
    save_plot(
        plt,
        filename
    )
    
    plt.show()


def plot_training_time(
    results_df,
    filename
):
    """
    Plot training time comparison.
    """
    
    pivot_df = results_df.pivot_table(
        index="model",
        columns="preprocessing",
        values="training_time_seconds",
        aggfunc="mean"
    )
    
    pivot_df.plot(
        kind="bar",
        figsize=(10, 6)
    )
    
    plt.title(
        "Training Time Across Models"
    )
    
    plt.ylabel("Training Time (seconds)")
    
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    
    save_plot(
        plt,
        filename
    )
    
    plt.show()