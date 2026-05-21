from src.utils.saving import (
    save_table
)


def create_summary_table(
    results_df,
    metric="f1_score"
):
    """
    Create summary table grouped by
    model and preprocessing.
    """
    
    summary_df = (
        results_df
        .groupby(
            ["model", "preprocessing"]
        )[[metric]]
        .mean()
        .reset_index()
    )
    
    return summary_df


def save_summary_table(
    results_df,
    metric,
    filename
):
    """
    Create and save summary table.
    """
    
    summary_df = create_summary_table(
        results_df,
        metric
    )
    
    save_table(
        summary_df,
        filename
    )
    
    return summary_df