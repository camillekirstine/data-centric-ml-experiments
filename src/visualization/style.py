import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# GLOBAL COLOR PALETTE
# ==========================================

COLOR_PALETTE = {
    "dark_blue": "#1E3A5F",
    "teal": "#2A9D8F",
    "light_teal": "#76C7C0",
    "grey": "#6C757D",
    "light_grey": "#D9E1E8"
}


# ==========================================
# PREPROCESSING COLORS
# ==========================================

PREPROCESSING_COLORS = {
    "Raw": COLOR_PALETTE["dark_blue"],
    "Cleaned": COLOR_PALETTE["teal"],
    "Engineered": COLOR_PALETTE["light_teal"]
}


# ==========================================
# MODEL COLORS
# ==========================================

MODEL_COLORS = {
    "Logistic Regression": COLOR_PALETTE["dark_blue"],
    "Decision Tree": COLOR_PALETTE["teal"],
    "Naive Bayes": COLOR_PALETTE["light_teal"],
    "MLP": COLOR_PALETTE["grey"]
}


# ==========================================
# GLOBAL PLOT STYLE
# ==========================================

def set_plot_style():
    """
    Apply global plotting style
    across all experiments.
    """

    sns.set_theme(
        style="whitegrid",
        context="talk"
    )

    plt.rcParams.update({

        # Figure sizing
        "figure.figsize": (10, 6),

        # Titles
        "axes.titlesize": 18,
        "axes.labelsize": 14,

        # Tick labels
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,

        # Grid styling
        "grid.alpha": 0.3,
        "grid.linestyle": "--",

        # Legend styling
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,

        # Line width
        "lines.linewidth": 2.5,

        # Export quality
        "savefig.dpi": 300,
        "savefig.bbox": "tight"
    })


# ==========================================
# CLEAN LEGEND STYLING
# ==========================================

def apply_legend_style(ax):
    """
    Apply consistent legend styling.
    """

    ax.legend(
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )