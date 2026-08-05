"""
eda_analysis.py
---------------
Beginner-friendly Exploratory Data Analysis (EDA) for the
Student Health Risk competition dataset.

Run:
    python ml/eda_analysis.py
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Project paths
ROOT = Path(__file__).resolve().parents[2]
DATA_CANDIDATES = [ROOT / "data" / "train.csv", ROOT / "train.csv"]
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "models" / "eda_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_train_csv() -> Path:
    """Find train.csv in data/ or project root."""
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "train.csv not found. Place it in data/train.csv or project root."
    )


def load_data() -> pd.DataFrame:
    path = find_train_csv()
    print(f"Loading: {path}")
    df = pd.read_csv(path)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def print_basic_overview(df: pd.DataFrame) -> None:
    print("\n=== COLUMNS & DTYPES ===")
    print(df.dtypes)

    print("\n=== TARGET DISTRIBUTION ===")
    counts = df["health_condition"].value_counts(dropna=False)
    perc = df["health_condition"].value_counts(normalize=True) * 100
    summary = pd.DataFrame({"count": counts, "percent": perc.round(2)})
    print(summary)

    print("\n=== MISSING VALUES ===")
    missing = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_percent": (df.isna().mean() * 100).round(2),
        }
    ).sort_values("missing_percent", ascending=False)
    print(missing)


def plot_target_balance(df: pd.DataFrame) -> None:
    """Bar chart of class imbalance — important for balanced accuracy."""
    plt.figure(figsize=(7, 4))
    order = df["health_condition"].value_counts().index
    sns.countplot(data=df, x="health_condition", order=order, color="#1f6f8b")
    plt.title("Target Class Distribution (health_condition)")
    plt.xlabel("Health Condition")
    plt.ylabel("Count")
    plt.tight_layout()
    out = OUTPUT_DIR / "01_target_distribution.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"Saved: {out}")


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    """Histograms for numeric lifestyle features."""
    numeric_cols = [
        "sleep_duration",
        "heart_rate",
        "bmi",
        "calorie_expenditure",
        "step_count",
        "exercise_duration",
        "water_intake",
    ]
    existing = [c for c in numeric_cols if c in df.columns]

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.flatten()
    for i, col in enumerate(existing):
        sns.histplot(df[col].dropna(), bins=40, ax=axes[i], color="#3d84a8")
        axes[i].set_title(col)
    for j in range(len(existing), len(axes)):
        axes[j].axis("off")
    plt.suptitle("Numeric Feature Distributions")
    plt.tight_layout()
    out = OUTPUT_DIR / "02_numeric_distributions.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"Saved: {out}")


def plot_boxplots_by_target(df: pd.DataFrame) -> None:
    """Compare key numeric features across health classes."""
    key_cols = ["bmi", "sleep_duration", "heart_rate", "step_count"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    for i, col in enumerate(key_cols):
        sns.boxplot(
            data=df,
            x="health_condition",
            y=col,
            ax=axes[i],
            color="#99c24d",
        )
        axes[i].set_title(f"{col} by health_condition")
    plt.tight_layout()
    out = OUTPUT_DIR / "03_boxplots_by_target.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"Saved: {out}")


def plot_categorical_vs_target(df: pd.DataFrame) -> None:
    """Stacked percentage bars for categorical lifestyle factors."""
    cat_cols = [
        "stress_level",
        "sleep_quality",
        "physical_activity_level",
        "smoking_alcohol",
        "diet_type",
        "gender",
    ]
    for col in cat_cols:
        if col not in df.columns:
            continue
        ct = pd.crosstab(df[col], df["health_condition"], normalize="index") * 100
        ct.plot(kind="bar", stacked=True, figsize=(8, 4), colormap="Set2")
        plt.title(f"{col} vs health_condition (% within category)")
        plt.ylabel("Percent")
        plt.xlabel(col)
        plt.legend(title="health_condition", bbox_to_anchor=(1.02, 1))
        plt.tight_layout()
        out = OUTPUT_DIR / f"04_{col}_vs_target.png"
        plt.savefig(out, dpi=140)
        plt.close()
        print(f"Saved: {out}")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Correlation among numeric features only."""
    numeric = df.select_dtypes(include="number").drop(columns=["id"], errors="ignore")
    corr = numeric.corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Numeric Feature Correlation Heatmap")
    plt.tight_layout()
    out = OUTPUT_DIR / "05_correlation_heatmap.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"Saved: {out}")


def main() -> None:
    sns.set_theme(style="whitegrid")
    df = load_data()
    print_basic_overview(df)
    plot_target_balance(df)
    plot_numeric_distributions(df)
    plot_boxplots_by_target(df)
    plot_categorical_vs_target(df)
    plot_correlation_heatmap(df)
    print("\nEDA complete.")


if __name__ == "__main__":
    main()
