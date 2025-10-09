import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TOP_N_CATEGORIES

def setup_plotting_style():
    sns.set_style("whitegrid")
    plt.rcParams['axes.unicode_minus'] = False

def plot_numerical_distributions(df: pd.DataFrame):
    for col in NUMERICAL_FEATURES:
        plt.figure(figsize=(10, 6))

        sns.histplot(
            data=df,
            x=col,
            kde=True,
            bins=30,
            color='skyblue',
            edgecolor='black',
            alpha=0.7
        )

        plt.title(f'Distribution of {col}', fontsize=16)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Count / Density', fontsize=12)
        plt.show()

def plot_categorical_counts(df: pd.DataFrame):
    for col in CATEGORICAL_FEATURES:
        plt.figure(figsize=(12, 6))

        # 筛选出最常见的 N 个类别
        top_categories = df[col].value_counts().nlargest(TOP_N_CATEGORIES).index
        df_filtered = df[df[col].isin(top_categories)]

        sns.countplot(
            data=df_filtered,
            y=col,
            order=top_categories,
            hue=col,
            legend=False,
            palette='viridis'
        )

        plt.title(f'Most Common {col} Distribution (Top {TOP_N_CATEGORIES} )', fontsize=16)
        plt.xlabel('Count', fontsize=12)
        plt.ylabel(col, fontsize=12)
        plt.tight_layout()
        plt.show()

def plot_figures(df: pd.DataFrame):
    setup_plotting_style()
    plot_numerical_distributions(df)
    plot_categorical_counts(df)