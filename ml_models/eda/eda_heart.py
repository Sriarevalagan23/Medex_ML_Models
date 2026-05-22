import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

data_path = Path(__file__).resolve().parents[1] / "processed" / "heart_cleaned.csv"

df = pd.read_csv(data_path)

print(df.info())

print(df.describe())

# TARGET DISTRIBUTION
sns.countplot(
    x="num",
    data=df
)

plt.title(
    "Heart Disease Distribution"
)
plots_dir = Path(__file__).resolve().parent / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(plots_dir / "heart_target_distribution.png", bbox_inches="tight")
plt.close()

# CORRELATION MATRIX
plt.figure(
    figsize=(12,8)
)

numeric_corr = df.select_dtypes(include=["number"]).corr()

sns.heatmap(
    numeric_corr,
    annot=True,
    cmap="coolwarm"
)

plt.title(
    "Feature Correlation"
)
plt.savefig(plots_dir / "heart_correlation.png", bbox_inches="tight")
plt.close()