import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TOP_N_CATEGORIES

def setup_plotting_style():
    sns.set_style("whitegrid")
    plt.rcParams['axes.unicode_minus'] = False

def plot_numerical_distributions(df: pd.DataFrame, log_transform: bool = True):
    for col in NUMERICAL_FEATURES:
        plt.figure(figsize=(10, 6))

        if log_transform and col == 'RESALE_PRICE':
            df['RESALE_PRICE_LOG'] = np.log1p(df['RESALE_PRICE'])
            col = 'RESALE_PRICE_LOG'

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

def plot_categorical_vs_numeric(
    df: pd.DataFrame,
    cat_col: str,
    num_col: str,
    top_n: int = None,
    agg: str = 'median',          # 'mean' or 'median'
    show_violin: bool = True,
    rotate_xticks: bool = False,
    figsize=(14,6),
    log_y: bool = False,
    sample_frac: float = None     # 若数据量巨大, 可按比例随机采样
):
    """
    常用的类别 vs 数值对比（箱线图/小提琴图 + 均值线/条形）
    - top_n: 若为 None 则使用 config 中 TOP_N_CATEGORIES；若为整数则显示该 top n 类别，其他合并为 'Other'
    - agg: 用于条形图聚合函数
    - sample_frac: 若提供, 先随机抽样（以提升绘图速度）
    """
    if sample_frac is not None and 0 < sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=42)

    top_n = TOP_N_CATEGORIES if top_n is None else top_n

    # 如果类别太多，只保留 top_n（其余标为 Other）
    vc = df[cat_col].value_counts()
    keep = vc.nlargest(top_n).index
    df_plot = df.copy()
    df_plot[cat_col] = df_plot[cat_col].where(df_plot[cat_col].isin(keep), other='Other')

    # 可选对数变换
    if log_y:
        df_plot[num_col] = np.log1p(df_plot[num_col])

    plt.figure(figsize=figsize)
    if show_violin:
        sns.violinplot(data=df_plot, x=cat_col, y=num_col, order=sorted(df_plot[cat_col].unique()),
                       cut=0, inner='quartile')
    else:
        sns.boxplot(data=df_plot, x=cat_col, y=num_col, order=sorted(df_plot[cat_col].unique()))

    # 绘制每类的聚合统计条（均值或中位数）
    agg_func = 'median' if agg == 'median' else 'mean'
    stats = df_plot.groupby(cat_col)[num_col].agg(agg_func).reindex(sorted(df_plot[cat_col].unique()))
    # 在次坐标系绘制条形（透明）
    ax = plt.gca()
    ax2 = ax.twinx()
    x_positions = np.arange(len(stats))
    ax2.bar(x_positions, stats.values, alpha=0.15, width=0.6)
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([])  # 不显示右侧刻度
    ax.set_title(f'{cat_col} vs {num_col} (top {top_n} + Other)')
    if rotate_xticks:
        plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_numeric_vs_numeric(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    sample_frac: float = 1.0,
    figsize=(8,6),
    kind: str = 'scatter',   # 'scatter', 'hex', 'reg' (regression line)
    log_x: bool = False,
    log_y: bool = False,
    alpha: float = 0.5
):
    """
    数值 vs 数值 的可视化
    - kind: scatter (散点), hex (六边箱/密度), reg (带回归线)
    - sample_frac: 采样比例（例如 0.2）
    """
    df_plot = df.copy()
    if sample_frac is not None and 0 < sample_frac < 1:
        df_plot = df_plot.sample(frac=sample_frac, random_state=42)

    if log_x:
        df_plot[x_col] = np.log1p(df_plot[x_col])
    if log_y:
        df_plot[y_col] = np.log1p(df_plot[y_col])

    plt.figure(figsize=figsize)
    if kind == 'scatter':
        sns.scatterplot(data=df_plot, x=x_col, y=y_col, alpha=alpha, s=20)
    elif kind == 'hex':
        plt.hexbin(df_plot[x_col], df_plot[y_col], gridsize=60, cmap='Blues', bins='log')
        cb = plt.colorbar()
        cb.set_label('log(count)')
    elif kind == 'reg':
        sns.regplot(data=df_plot, x=x_col, y=y_col, scatter_kws={'alpha':alpha, 's':20}, line_kws={'color':'red'})
    else:
        raise ValueError("kind must be one of ['scatter','hex','reg']")

    plt.title(f'{y_col} vs {x_col} ({kind})')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.show()

def plot_town_vs_price(df, top_n=None, log_price=True):
    plot_categorical_vs_numeric(df, 'TOWN', 'RESALE_PRICE', top_n=top_n, agg='median', show_violin=False, rotate_xticks=True, log_y=log_price)

def plot_area_vs_price(df, sample_frac=0.2, kind='hex', log_price=True, log_area=False):
    plot_numeric_vs_numeric(df, 'FLOOR_AREA_SQM', 'RESALE_PRICE', sample_frac=sample_frac, kind=kind, log_x=log_area, log_y=log_price)

def plot_figures(df: pd.DataFrame):
    setup_plotting_style()
    plot_numerical_distributions(df)
    plot_categorical_counts(df)

    plot_town_vs_price(df, top_n=15, log_price=False)
    plot_area_vs_price(df, sample_frac=0.1, kind='reg', log_price=False, log_area=False)