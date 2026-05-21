from pathlib import Path


RESULTS_DIR = Path("../results")

METRICS_DIR = RESULTS_DIR / "metrics"

TABLES_DIR = RESULTS_DIR / "tables"

PLOTS_DIR = RESULTS_DIR / "plots"


def ensure_directories():
    """
    Create results directories if missing.
    """
    
    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )
    
    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )
    
    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def save_metrics(
    results_df,
    filename
):
    """
    Save raw experiment metrics.
    """
    
    ensure_directories()
    
    filepath = METRICS_DIR / filename
    
    results_df.to_csv(
        filepath,
        index=False
    )
    
    print(f"Metrics saved to: {filepath}")


def save_table(
    table_df,
    filename
):
    """
    Save summary table.
    """
    
    ensure_directories()
    
    filepath = TABLES_DIR / filename
    
    table_df.to_csv(
        filepath,
        index=False
    )
    
    print(f"Table saved to: {filepath}")


def save_plot(
    plt,
    filename
):
    """
    Save matplotlib figure.
    """
    
    ensure_directories()
    
    filepath = PLOTS_DIR / filename
    
    plt.savefig(
        filepath,
        bbox_inches="tight"
    )
    
    print(f"Plot saved to: {filepath}")