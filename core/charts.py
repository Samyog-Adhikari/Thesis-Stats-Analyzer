"""Matplotlib and seaborn charts for analysis pages and export."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def histogram(df: pd.DataFrame, variable: str, normal_curve: bool = True):
    """Create a labelled histogram with an optional normal density curve."""
    import numpy as np
    from scipy.stats import norm
    x = pd.to_numeric(df[variable], errors="coerce").dropna()
    if x.empty: raise ValueError("No numeric observations available for the histogram.")
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white"); sns.histplot(x, kde=False, stat="density", ax=ax, color="#2563eb")
    if normal_curve and len(x) > 1:
        xx = np.linspace(x.min(), x.max(), 200); ax.plot(xx, norm.pdf(xx, x.mean(), x.std(ddof=1)), color="#dc2626", label="Normal curve"); ax.legend()
    ax.set(title=f"Distribution of {variable}", xlabel=variable, ylabel="Density"); fig.tight_layout(); return fig


def bar_chart(df: pd.DataFrame, variable: str):
    """Create a categorical frequency bar chart."""
    counts = df[variable].dropna().value_counts().sort_values(ascending=False); fig, ax = plt.subplots(figsize=(7,4), facecolor="white"); counts.plot.bar(ax=ax, color="#0f766e"); ax.set(title=f"Frequencies of {variable}", xlabel=variable, ylabel="Count"); ax.tick_params(axis="x", rotation=30); fig.tight_layout(); return fig


def boxplot(df: pd.DataFrame, variable: str, grouping: str | None = None):
    """Create a boxplot for a scale variable, optionally grouped."""
    fig, ax = plt.subplots(figsize=(7,4), facecolor="white"); sns.boxplot(data=df, x=grouping, y=variable, ax=ax, color="#93c5fd"); ax.set(title=f"Boxplot of {variable}", xlabel=grouping or "", ylabel=variable); fig.tight_layout(); return fig
