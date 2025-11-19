import os
import joblib
import pandas as pd

ARTIFACT_DIR = "/app/model_artifacts"   

class CreditRiskModel:
    """Model loader and prediction wrapper."""

    def __init__(self):
        model_path = os.path.join(ARTIFACT_DIR, "model.pkl")
        preprocessor_path = os.path.join(ARTIFACT_DIR, "preprocessor.pkl")

        print("📁 Using artifact directory:", ARTIFACT_DIR)
        print("📄 Loading model:", model_path)
        print("📄 Loading preprocessor:", preprocessor_path)

        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

    def predict(self, input_dict: dict):
        df = pd.DataFrame([input_dict])
        x = self.preprocessor.transform(df)
        prob = float(self.model.predict_proba(x)[0][1])
        prediction = int(prob > 0.20)
        return {"risk_score": prob, "prediction": prediction}

