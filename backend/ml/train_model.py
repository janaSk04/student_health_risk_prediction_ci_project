"""
train_model.py
--------------
Trains a beginner-friendly Random Forest classifier for student health risk.

Why Random Forest?
- Ensemble method (allowed by assignment)
- Handles non-linear patterns
- class_weight='balanced' helps minority classes (fit / unhealthy)
- Easy to explain with feature importance

Run from backend folder:
    python ml/train_model.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parents[1]
ROOT = BACKEND_DIR.parent
MODEL_DIR = BACKEND_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "health_condition"
ID_COL = "id"

NUMERIC_FEATURES = [
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
]

CATEGORICAL_FEATURES = [
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def find_train_csv() -> Path:
    candidates = [ROOT / "data" / "train.csv", ROOT / "train.csv"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("train.csv not found in data/ or project root.")


def load_training_data() -> pd.DataFrame:
    path = find_train_csv()
    print(f"Loading training data from: {path}")
    df = pd.read_csv(path)
    print(f"Rows: {len(df):,}")
    print("Class balance:")
    print(df[TARGET].value_counts(normalize=True).mul(100).round(2))
    return df


def build_pipeline() -> Pipeline:
    """
    Build a full preprocessing + model pipeline.

    This SAME pipeline is used later by the FastAPI app, so training and
    prediction stay consistent.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    # Beginner ensemble model with balanced class weights
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=18,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )
    return pipeline


def evaluate_holdout(pipeline: Pipeline, X_test, y_test) -> dict:
    """Evaluate on a stratified hold-out set using competition-style metric."""
    y_pred = pipeline.predict(X_test)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_).tolist()

    print("\n=== HOLD-OUT EVALUATION ===")
    print(f"Balanced accuracy: {bal_acc:.4f}")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix labels:", list(pipeline.classes_))
    print(cm)

    return {
        "balanced_accuracy": float(bal_acc),
        "classification_report": report,
        "confusion_matrix": cm,
        "classes": list(pipeline.classes_),
    }


def save_feature_importance(pipeline: Pipeline) -> None:
    """Export Random Forest feature importances for the report."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Get final feature names after one-hot encoding
    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    fi = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    out = MODEL_DIR / "feature_importance.csv"
    fi.to_csv(out, index=False)
    print(f"Saved feature importance: {out}")
    print(fi.head(15).to_string(index=False))


def main() -> None:
    df = load_training_data()

    # Drop rows with missing target (should be rare)
    df = df.dropna(subset=[TARGET]).copy()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET]

    # Stratified split keeps class ratios similar in train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()

    print("\nFitting Random Forest pipeline...")
    pipeline.fit(X_train, y_train)

    metrics = evaluate_holdout(pipeline, X_test, y_test)

    # Optional quick cross-validation on a sample for stability check
    # (full CV on 690k rows can be slow on some laptops)
    sample_n = min(80_000, len(X_train))
    X_sample = X_train.sample(n=sample_n, random_state=42)
    y_sample = y_train.loc[X_sample.index]
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        build_pipeline(),
        X_sample,
        y_sample,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )
    metrics["cv_balanced_accuracy_mean"] = float(np.mean(cv_scores))
    metrics["cv_balanced_accuracy_std"] = float(np.std(cv_scores))
    metrics["cv_sample_size"] = int(sample_n)
    print(
        f"\n3-fold CV balanced accuracy on {sample_n:,} rows: "
        f"{metrics['cv_balanced_accuracy_mean']:.4f} "
        f"(+/- {metrics['cv_balanced_accuracy_std']:.4f})"
    )

    # Retrain on ALL training data before saving for Kaggle / app use
    print("\nRetraining on full training set for deployment...")
    final_pipeline = build_pipeline()
    final_pipeline.fit(X, y)

    model_path = MODEL_DIR / "health_risk_model.joblib"
    joblib.dump(
        {
            "pipeline": final_pipeline,
            "feature_columns": FEATURE_COLUMNS,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "target": TARGET,
            "classes": list(final_pipeline.classes_),
        },
        model_path,
    )
    print(f"Saved model: {model_path}")

    save_feature_importance(final_pipeline)

    metrics_path = MODEL_DIR / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
