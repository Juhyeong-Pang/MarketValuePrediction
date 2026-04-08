import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DISCRETE_COLUMN_LIST = [
    "Age",
    "Born",
    "Match Played",
    "Match Started",
    "Goals",
    "Assists",
    "Goals + Assists",
    "Non-Penality Goals",
    "Penalty Kick Goals",
    "Penalty Kick Attempted",
    "Yellow Cards",
    "Red Cards",
    "Experience Level", 
    "Penalty Kicker"
]


def plot_cardinality(df):
    cardinality = df.nunique().sort_values(ascending=False)

    plt.figure(figsize=(14, 8))

    ax = sns.barplot(
        x=cardinality.values,
        y=cardinality.index,
        hue=cardinality.index,
        legend=False,
        palette="magma",
    )

    for container in ax.containers:
        ax.bar_label(container, padding=5)

    plt.title("Unique Values (Cardinality)")
    plt.xlabel("Number of Unique Values")
    plt.ylabel("Columns")
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()


def plot_distribution(
    df,
    column_name,
    figsize=(10, 6),
    color="rebeccapurple",
    discrete=False,
    ax=None,
    show_count=False,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    data = df[column_name].dropna()
    is_numeric = pd.api.types.is_numeric_dtype(data)

    if is_numeric:
        sns.histplot(data, kde=False, color=color, discrete=discrete, ax=ax)
        ax.set_title(f"{column_name} Distribution", fontsize=12)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")

        d_min = data.min()
        d_mean = data.mean()
        d_max = data.max()

        ax.axvline(d_min, color="red", linestyle="--", linewidth=1, 
                   label=f"Min: {d_min:.2f}", alpha=0.5)
        ax.axvline(d_mean, color="black", linestyle="-", linewidth=2, 
                   label=f"Mean: {d_mean:.2f}", alpha=0.3)
        ax.axvline(d_max, color="blue", linestyle="--", linewidth=1, 
                   label=f"Max: {d_max:.2f}", alpha=0.5)
        
        ax.legend()

    else:
        sns.countplot(x=data, color=color, ax=ax, order=data.value_counts().index)
        ax.set_title(f"{column_name} Frequency", fontsize=12)
        ax.set_xlabel("Category")
        ax.set_ylabel("Frequency")
        
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    if not is_numeric or (show_count and len(ax.containers) > 0):
        for container in ax.containers:
            ax.bar_label(container, fmt='%.0f', padding=3, 
                         fontsize=10, fontweight='bold', color='black')

    heights = [p.get_height() for p in ax.patches if p.get_height() > 0]
    if heights:
        max_height = max(heights)
        ax.set_ylim(0, max_height * 1.2)


def plot_entire_distribution(df, color="rebeccapurple"):
    cols_to_plot = [col for col in df.columns if col not in ["DF", "FW", "GK", "MF"]]

    n_cols = 5
    n_rows = math.ceil(len(cols_to_plot) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten()

    for i, column in enumerate(cols_to_plot):
        plot_distribution(
            df,
            column,
            color=color,
            discrete=(column in DISCRETE_COLUMN_LIST),
            ax=axes[i],
        )

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_distribution_stats(df):
    """
    Skewness measures the symmetry of a variable's distribution.
    If the distribution stretches toward the right or left tail, it's skewed.
    Negative skewness indicates more larger values, while positive skewness indicates more smaller values.
    A skewness value between -1 and +1 is excellent, while -2 to +2 is generally acceptable.
    Values beyond -2 and +2 suggest substantial nonnormality.

    Kurtosis indicates whether the distribution is too peaked or flat compared to a normal distribution.
    Positive kurtosis means a more peaked distribution, while negative kurtosis means a flatter one.
    A kurtosis greater than +2 suggests a too peaked distribution, while less than -2 indicates a too flat one.

    In the rare scenario where both skewness and kurtosis are zero, the pattern of responses is considered a normal distribution.
    """
    numeric_df = df.select_dtypes(include=["number"])

    stats = pd.DataFrame(
        {"Skewness": numeric_df.skew(), "Kurtosis": numeric_df.kurtosis()}
    ).sort_values(by="Skewness", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    titles = ["Skewness", "Kurtosis"]
    palettes = ["coolwarm", "magma"]

    for i, col in enumerate(stats.columns):
        ax = axes[i]
        sns.barplot(
            x=stats[col],
            y=stats.index,
            ax=ax,
            hue=stats.index,
            palette=palettes[i],
            legend=False,
        )

        ax.set_title(f"{titles[i]}")
        ax.axvline(0, color="black", linestyle="-", linewidth=1)

        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", padding=7)

        x_min, x_max = ax.get_xlim()
        margin = (x_max - x_min) * 0.15
        ax.set_xlim(x_min - margin if x_min < 0 else x_min, x_max + margin)

    plt.tight_layout()
    plt.show()


def plot_outlier_ratio(df):
    numeric_df = df.select_dtypes(include=[np.number])
    outlier_ratios = {}

    for col in numeric_df.columns:
        data = numeric_df[col].dropna()
        if len(data) == 0:
            continue

        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = data[(data < lower_bound) | (data > upper_bound)]
        ratio = len(outliers) / len(data)
        outlier_ratios[col] = ratio

    outlier_series = pd.Series(outlier_ratios).sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=outlier_series.values,
        y=outlier_series.index,
        hue=outlier_series.index,
        legend=False,
        palette="magma",
    )

    plt.title("Outlier Ratio per Column (IQR Method)", pad=20)
    plt.xlabel("Ratio of Outliers")
    plt.ylabel("Numeric Columns")
    plt.grid(axis="x", linestyle=":", alpha=0.6)

    max_val = outlier_series.max()
    if max_val > 0:
        plt.xlim(0, max_val * 1.15)
    else:
        plt.xlim(0, 0.1)

    for i, v in enumerate(outlier_series.values):
        plt.text(v, i, f" {v:.2%}", va="center", fontweight="bold")

    plt.tight_layout()
    plt.show()


def plot_boxplot(df, column_name, figsize=(10, 4), color="rebeccapurple", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    data = df[column_name].dropna()
    if data.empty:
        return

    sns.boxplot(
        x=data,
        ax=ax,
        color=color,
        flierprops={
            "marker": "o",
            "markersize": 4,
            "markerfacecolor": "red",
            "markeredgecolor": "none",
            "alpha": 0.5,
        },
        linewidth=1.5,
    )

    ax.set_title(f"{column_name}", fontsize=12, pad=15)
    ax.set_xlabel("Value")

    d_min = data.min()
    d_mean = data.mean()
    d_max = data.max()

    ax.axvline(
        d_min,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"Min: {d_min:.1f}",
        alpha=0.6,
    )
    ax.axvline(
        d_mean,
        color="black",
        linestyle="-",
        linewidth=2,
        label=f"Mean: {d_mean:.1f}",
        alpha=0.8,
    )
    ax.axvline(
        d_max,
        color="blue",
        linestyle="--",
        linewidth=1,
        label=f"Max: {d_max:.1f}",
        alpha=0.6,
    )

    ax.legend(fontsize=8, loc="upper right")

    current_xlim = ax.get_xlim()
    ax.set_xlim(
        current_xlim[0], current_xlim[1] + (current_xlim[1] - current_xlim[0]) * 0.1
    )


def plot_entire_boxplot(df, color="rebeccapurple"):
    cols_to_plot = [col for col in df.columns if col not in ["DF", "FW", "GK", "MF"]]

    n_cols = 5
    n_rows = math.ceil(len(cols_to_plot) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3))
    axes = axes.flatten()

    for i, column in enumerate(cols_to_plot):
        if pd.api.types.is_numeric_dtype(df[column]):
            plot_boxplot(df, column, color=color, ax=axes[i])
        else:
            axes[i].set_title(f"{column} (Not Numeric)")

    for j in range(len(cols_to_plot), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def get_cv_summary(df):
    """
    The coefficient of variation (CV) is a statistical measure of relative dispersion,
    calculating the ratio of the standard deviation to the mean, typically expressed as a percentage.

    It indicates how large the standard deviation is relative to the mean,
    allowing for comparison of variability between datasets with different units or scales

    < 10% -> High precision; desirable in fields like quality control
    10% ~ 30% -> Moderate, acceptable in some contexts
    > 30% -> Considered inconsistent, volatile, or highly dispersed
    > 100% -> Standard Deviation > Mean (Tooo much sparse (0), or extremely affected by outlier)
    """
    cols_to_analyze = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in ["DF", "FW", "GK", "MF"]
    ]

    numeric_df = df[cols_to_analyze]

    means = numeric_df.mean()
    stds = numeric_df.std()

    cv = np.where(means != 0, (stds / means) * 100, 0)

    cv_df = pd.DataFrame(
        {"Mean": means, "Std Dev": stds, "CV (%)": cv}, index=cols_to_analyze
    )

    return cv_df.sort_values(by="CV (%)", ascending=False)


def plot_cv(df):
    """
    The coefficient of variation (CV) is a statistical measure of relative dispersion,
    calculating the ratio of the standard deviation to the mean, typically expressed as a percentage.

    It indicates how large the standard deviation is relative to the mean,
    allowing for comparison of variability between datasets with different units or scales

    < 10% -> High precision; desirable in fields like quality control
    10% ~ 30% -> Moderate, acceptable in some contexts
    > 30% -> Considered inconsistent, volatile, or highly dispersed
    > 100% -> Standard Deviation > Mean (Tooo much sparse (0), or extremely affected by outlier)
    """
    cv_summary = get_cv_summary(df)

    cv_summary = cv_summary[cv_summary["CV (%)"] > 0]

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        x="CV (%)",
        y=cv_summary.index,
        data=cv_summary,
        hue=cv_summary.index,
        legend=False,
        palette="magma",
    )

    plt.title("Coefficient of Variation (Relative Volatility) per Feature", pad=20)
    plt.xlabel("CV (%) - Higher means more relative variation")
    plt.ylabel("Features")
    plt.grid(axis="x", linestyle=":", alpha=0.6)

    max_val = cv_summary["CV (%)"].max()
    plt.xlim(0, max_val * 1.15)

    for i, v in enumerate(cv_summary["CV (%)"]):
        ax.text(v, i, f" {v:.1f}%", va="center", fontweight="bold")

    plt.tight_layout()
    plt.show()


def plot_corr(df):
    correlation_matrix = df.corr()

    plt.figure(figsize=(20, 18))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1)
    plt.show()
