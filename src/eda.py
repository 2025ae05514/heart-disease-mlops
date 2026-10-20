"""EDA plots saved to reports/figures."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 

from src.data import CLEAN, NUM 

OUT = Path("reports/figures")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLEAN)
    sns.set_theme(style="whitegrid")

    ax = sns.countplot(x="target", data=df)
    ax.set_title("Class balance (0 = no disease, 1 = disease)")
    for c in ax.containers:
        ax.bar_label(c)
    plt.savefig(OUT / "class_balance.png", dpi=120, bbox_inches="tight")
    plt.close()

    df[NUM].hist(bins=20, figsize=(11, 7), color="#3b7ddd")
    plt.suptitle("Numeric feature distributions")
    plt.savefig(OUT / "histograms.png", dpi=120, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(df.drop(columns="source").corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation heatmap")
    plt.savefig(OUT / "correlation.png", dpi=120, bbox_inches="tight")
    plt.close()

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for a, col in zip(axs, ["age", "thalach", "oldpeak"]):
        sns.boxplot(x="target", y=col, data=df, ax=a)
    plt.savefig(OUT / "boxplots.png", dpi=120, bbox_inches="tight")
    plt.close()
    plt.figure(figsize=(7, 4))
    sns.countplot(x="source", hue="target", data=df)
    plt.title("Rows and class split per source file")
    plt.savefig(OUT / "source_split.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("saved figures to", OUT)


if __name__ == "__main__":
    main()
