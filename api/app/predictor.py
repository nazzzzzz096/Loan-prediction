"""Model loader and prediction wrapper for the API."""

import os
import joblib
import pandas as pd

def get_project_root():
    """
    Walk upward from current file until we find 'Loan-prediction' folder.
    This ensures reliability in pytest, docker, and local environments.
    """
    current = os.path.abspath(__file__)

    while True:
        current = os.path.dirname(current)
        if current.endswith("Loan-prediction"):
            return current
        if current == os.path.dirname(current):
            raise RuntimeError("❌ Could not locate project root: 'Loan-prediction'")

PROJECT_ROOT = get_project_root()

ARTIFACT_DIR = os.path.join(
    PROJECT_ROOT,
    "model_training",
    "model_training",
    "artifacts"
)

class CreditRiskModel:
    """Loads the trained model and preprocessor and exposes a predict() method."""


    def __init__(self):
        """Load model and preprocessor from artifacts on initialization."""

        print("📌 Loading model and preprocessor...")

        model_path = os.path.join(ARTIFACT_DIR, "model.pkl")
        pre_path = os.path.join(ARTIFACT_DIR, "preprocessor.pkl")

        print("Model path:", model_path)
        print("Preprocessor path:", pre_path)

        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(pre_path)

    def predict(self, input_dict: dict):
        """Preprocess input dict and return probability and binary decision."""
        df = pd.DataFrame([input_dict])

        x = self.preprocessor.transform(df)

        prob = self.model.predict_proba(x)[0][1]
        prediction = int(prob > 0.20)

        return {
            "risk_score": float(prob),
            "prediction": prediction
        }
