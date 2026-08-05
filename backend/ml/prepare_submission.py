"""
prepare_submission.py
---------------------
Creates Kaggle submission.csv using the trained pipeline.

Expected test file columns match train features + id.
Output format:
    id,health_condition
"""

from pathlib import Path

import joblib
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
ROOT = BACKEND_DIR.parent
MODEL_PATH = BACKEND_DIR / "models" / "health_risk_model.joblib"
OUT_PATH = BACKEND_DIR / "models" / "submission.csv"


def find_test_csv() -> Path:
    candidates = [ROOT / "data" / "test.csv", ROOT / "test.csv"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "test.csv not found. Download it from Kaggle and place it in data/test.csv"
    )


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run train_model.py first to create health_risk_model.joblib"
        )

    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]
    feature_columns = bundle["feature_columns"]

    test_path = find_test_csv()
    print(f"Loading test data: {test_path}")
    test_df = pd.read_csv(test_path)

    X_test = test_df[feature_columns]
    preds = pipeline.predict(X_test)

    submission = pd.DataFrame(
        {
            "id": test_df["id"],
            "health_condition": preds,
        }
    )
    submission.to_csv(OUT_PATH, index=False)
    print(f"Saved submission: {OUT_PATH}")
    print(submission.head())
    print("Upload this file to the Kaggle competition page.")


if __name__ == "__main__":
    main()
