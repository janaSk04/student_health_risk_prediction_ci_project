from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[2]
BACKEND = Path(__file__).resolve().parents[1]
OUT = BACKEND / "models" / "model_comparison.csv"

NUMERIC = [
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
]
CATEGORICAL = [
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
]
TARGET = "health_condition"


def find_train() -> Path:
    for p in [ROOT / "data" / "train.csv", ROOT / "train.csv"]:
        if p.exists():
            return p
    raise FileNotFoundError("train.csv not found")


def make_preprocessor() -> ColumnTransformer:
    num = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", num, NUMERIC),
            ("cat", cat, CATEGORICAL),
        ]
    )


def main() -> None:
    df = pd.read_csv(find_train())
    df = df.dropna(subset=[TARGET])

    # Sample keeps runtime beginner-friendly on large data
    sample_n = min(60_000, len(df))
    df = df.sample(n=sample_n, random_state=42)

    X = df[NUMERIC + CATEGORICAL]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "DecisionTree": DecisionTreeClassifier(
            max_depth=12, class_weight="balanced", random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=16,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "NeuralNet_MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=30,
            random_state=42,
        ),
    }

    rows = []
    for name, clf in models.items():
        pipe = Pipeline(
            [
                ("preprocessor", make_preprocessor()),
                ("classifier", clf),
            ]
        )
        print(f"\nTraining {name}...")
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        bal = balanced_accuracy_score(y_test, pred)
        print(f"{name} balanced accuracy: {bal:.4f}")
        print(classification_report(y_test, pred))
        rows.append({"model": name, "balanced_accuracy": round(float(bal), 4)})

    result = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)
    print(f"\nSaved comparison table: {OUT}")
    print(result.to_string(index=False))

if __name__ == "__main__":
    main()
