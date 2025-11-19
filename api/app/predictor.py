"""Model loader and prediction wrapper for the API."""

import os
import joblib
import pandas as pd

# Project-root discovery omitted here - keep your working version
def get_project_root():
    current = os.path.abspath(__file__)
    while True:
        current = os.path.dirname(current)
        if current.endswith("Loan-prediction"):
            return current
        if current == os.path.dirname(current):
            raise RuntimeError("Could not locate project root: 'Loan-prediction'")

PROJECT_ROOT = get_project_root()
ARTIFACT_DIR = os.path.join(
    PROJECT_ROOT,
    "model_training",
    "model_training",
    "artifacts",
)

class CreditRiskModel:
    """Loads the trained model and preprocessor and exposes a predict() method.

    Having an explicit load() method increases clarity and satisfies pylint's
    'too few public methods' rule in a harmless way.
    """

    def __init__(self) -> None:
        """Initialize and load model & preprocessor."""
        self.model = None
        self.preprocessor = None
        self.load()

    def load(self) -> None:
        """Load model and preprocessor from artifacts."""
        model_path = os.path.join(ARTIFACT_DIR, "model.pkl")
        pre_path = os.path.join(ARTIFACT_DIR, "preprocessor.pkl")
        # will raise if not present — acceptable for startup
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(pre_path)

    def predict(self, input_dict: dict) -> dict:
        """Preprocess input dict and return probability and binary decision."""
        df = pd.DataFrame([input_dict])
        x = self.preprocessor.transform(df)
        prob = float(self.model.predict_proba(x)[0][1])
        prediction = int(prob > 0.20)
        return {"risk_score": prob, "prediction": prediction}

