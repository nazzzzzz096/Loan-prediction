import os
import joblib
import pandas as pd

def get_artifact_dir():
    """Return correct artifact directory depending on environment."""

    # 1️⃣ Running inside Docker
    if os.environ.get("DOCKER_ENV") == "1":
        return "/app/model_artifacts"

    # 2️⃣ Running in project locally
    local_path = os.path.join(
        os.path.dirname(__file__),
        "..", "..",
        "model_training", "model_training", "artifacts"
    )
    local_path = os.path.abspath(local_path)
    if os.path.exists(local_path):
        return local_path

    # 3️⃣ Running in CI (repo root)
    ci_path = os.path.abspath("model_training/model_training/artifacts")
    if os.path.exists(ci_path):
        return ci_path

    raise RuntimeError("Could not locate artifact directory")

ARTIFACT_DIR = get_artifact_dir()


class CreditRiskModel:
    def __init__(self):
        model_path = os.path.join(ARTIFACT_DIR, "model.pkl")
        pre_path = os.path.join(ARTIFACT_DIR, "preprocessor.pkl")

        print(f"📁 Using artifact directory: {ARTIFACT_DIR}")
        print(f"📄 Loading model: {model_path}")
        print(f"📄 Loading preprocessor: {pre_path}")

        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(pre_path)

    def predict(self, input_dict: dict):
        df = pd.DataFrame([input_dict])
        x = self.preprocessor.transform(df)
        prob = float(self.model.predict_proba(x)[0][1])
        return {"risk_score": prob, "prediction": int(prob > 0.20)}


