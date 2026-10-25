"""
visualizations.py
-----------------
Matplotlib + Seaborn chart generation.
Every function returns a base64-encoded PNG string.
Figures are always closed after rendering to prevent memory leaks.
"""

import io
import base64

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
LIGHT_BG = "#ffffff"
ACCENT   = "#4f6ef7"


def _encode(fig) -> str:
    """Save figure to base64 PNG and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110, facecolor=LIGHT_BG)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def histogram(df: pd.DataFrame, column: str) -> str:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")
    series = df[column].dropna()
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError(f"Column '{column}' is not numeric.")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(series, kde=True, ax=ax, color=ACCENT, edgecolor="white")
    ax.set_title(f"Distribution of {column}", fontsize=13, fontweight="bold")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return _encode(fig)


def boxplot(df: pd.DataFrame, column: str) -> str:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")
    series = df[column].dropna()
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError(f"Column '{column}' is not numeric.")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(x=series, ax=ax, color=ACCENT)
    ax.set_title(f"Box Plot of {column}", fontsize=13, fontweight="bold")
    ax.set_xlabel(column)
    fig.tight_layout()
    return _encode(fig)


def scatter(df: pd.DataFrame, x_col: str, y_col: str) -> str:
    for col in (x_col, y_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' must be numeric for a scatter plot.")

    tmp = df[[x_col, y_col]].dropna()
    # Downsample to avoid huge render times
    if len(tmp) > 2000:
        tmp = tmp.sample(2000, random_state=42)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(tmp[x_col], tmp[y_col], alpha=0.55, color=ACCENT, edgecolors="none", s=25)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{x_col} vs {y_col}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _encode(fig)


def correlation_heatmap(df: pd.DataFrame) -> str:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        raise ValueError("Need at least 2 numeric columns for a correlation heatmap.")

    corr = numeric_df.corr(numeric_only=True)
    n = corr.shape[0]
    size = max(6, min(1.1 * n + 2, 14))
    fig, ax = plt.subplots(figsize=(size, size * 0.8))
    sns.heatmap(
        corr,
        annot=n <= 15,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
        square=True,
        linewidths=0.5,
    )
    ax.set_title("Correlation Heatmap", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _encode(fig)


def line_chart(df: pd.DataFrame, x_col: str, y_col: str) -> str:
    for col in (x_col, y_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")
    if not pd.api.types.is_numeric_dtype(df[y_col]):
        raise ValueError(f"Column '{y_col}' must be numeric for a line chart.")

    tmp = df[[x_col, y_col]].dropna().sort_values(x_col)
    if len(tmp) > 2000:
        tmp = tmp.iloc[::max(1, len(tmp) // 2000)]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(tmp[x_col], tmp[y_col], color=ACCENT, linewidth=1.5)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} over {x_col}", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return _encode(fig)


def count_plot(df: pd.DataFrame, column: str) -> str:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    vc = df[column].value_counts().head(20)
    if vc.empty:
        raise ValueError(f"Column '{column}' has no data to plot.")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=vc.values, y=vc.index.astype(str), ax=ax, color=ACCENT)
    ax.set_title(f"Category Counts: {column}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Count")
    ax.set_ylabel(column)
    fig.tight_layout()
    return _encode(fig)


def bar_chart(df: pd.DataFrame, x_col: str, y_col: str) -> str:
    for col in (x_col, y_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")
    if not pd.api.types.is_numeric_dtype(df[y_col]):
        raise ValueError(f"Column '{y_col}' must be numeric for a bar chart.")

    tmp = df[[x_col, y_col]].dropna()
    grouped = tmp.groupby(x_col)[y_col].mean().sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(x=grouped.index.astype(str), y=grouped.values, ax=ax, color=ACCENT)
    ax.set_title(f"Mean {y_col} by {x_col}", fontsize=13, fontweight="bold")
    ax.set_xlabel(x_col)
    ax.set_ylabel(f"Mean {y_col}")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return _encode(fig)


# ── Auto-overview charts (used on the overview page) ─────────────────────────

def auto_histograms(df: pd.DataFrame, max_cols: int = 9) -> str:
    """Grid of histograms for the first few numeric columns."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
    if not num_cols:
        raise ValueError("No numeric columns to plot.")

    n = len(num_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(num_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color=ACCENT, edgecolor="white")
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel("")
    for j in range(len(num_cols), len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle("Numeric Distributions", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return _encode(fig)


def auto_boxplots(df: pd.DataFrame, max_cols: int = 9) -> str:
    """Grid of boxplots for outlier detection."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
    if not num_cols:
        raise ValueError("No numeric columns to plot.")

    n = len(num_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.0 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(num_cols):
        sns.boxplot(x=df[col].dropna(), ax=axes[i], color="#DD8452")
        axes[i].set_title(col, fontsize=10)
    for j in range(len(num_cols), len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle("Outlier Check (Box Plots)", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return _encode(fig)
