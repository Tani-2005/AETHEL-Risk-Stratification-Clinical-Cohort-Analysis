"""
publication_layout.py
=====================
Defines visual styling and layout guidelines matching high-tier medical journals
(e.g., Nature Digital Medicine, Lancet Digital Health).
Sets up standard fonts, colors, and layout configurations.
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import seaborn as sns

# Color palettes suitable for publication and colorblind friendly
CLINICAL_COLORS = {
    "primary": "#1f77b4",      # Steel Blue
    "secondary": "#ff7f0e",    # Warm Orange
    "success": "#2ca02c",      # Soft Green
    "danger": "#d62728",       # Crimson Red
    "warning": "#bcbd22",      # Olive
    "info": "#17becf",         # Muted Cyan
    "accent": "#9467bd",       # Soft Purple
    "neutral_dark": "#2c3e50", # Slate Grey
    "neutral_light": "#ecf0f1",# Light Grey
}

COLOR_PALETTE = list(CLINICAL_COLORS.values())

def apply_publication_theme() -> None:
    """
    Applies standard journal visual style configs to matplotlib rcParams.
    """
    # Use Helvetica if available, otherwise DejaVu Sans
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans", "Liberation Sans"]
    
    # Font sizes matching journal guidelines
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 8
    plt.rcParams["ytick.labelsize"] = 8
    plt.rcParams["legend.fontsize"] = 8
    plt.rcParams["figure.titlesize"] = 12
    
    # Layout and grids
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.color"] = "#cccccc"
    
    # Line widths
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.linewidth"] = 0.5
    plt.rcParams["lines.linewidth"] = 1.2
    plt.rcParams["patch.linewidth"] = 0.8
    
    # Ticks
    plt.rcParams["xtick.major.width"] = 0.8
    plt.rcParams["ytick.major.width"] = 0.8
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    
    # DPI & Saving
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.transparent"] = False

def clean_plot(ax: plt.Axes) -> None:
    """
    Cleans up redundant axes elements to achieve a clean publication layout.
    Removes top and right spines, and lightens grids.
    """
    sns.despine(ax=ax, top=True, right=True)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.xaxis.grid(False)  # Often, horizontal grids are enough
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#cccccc")

def get_figure_size(columns: int = 1) -> tuple[float, float]:
    """
    Returns standard figure dimensions in inches.
    1 column = 3.5 inches
    2 columns = 7.0 inches
    """
    if columns == 1:
        return (3.5, 3.5)
    return (7.0, 5.0)
