"""
model_loader.py
---------------
Loads the trained Random Forest pipeline once at startup.
"""

from pathlib import Path
from typing import Any, Dict

import joblib

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "health_risk_model.joblib"

_model_bundle: Dict[str, Any] | None = None


def load_model_bundle() -> Dict[str, Any]:
    global _model_bundle
    if _model_bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. "
                "Run: python ml/train_model.py"
            )
        _model_bundle = joblib.load(MODEL_PATH)
    return _model_bundle


def get_pipeline():
    return load_model_bundle()["pipeline"]


def get_feature_columns():
    return load_model_bundle()["feature_columns"]
