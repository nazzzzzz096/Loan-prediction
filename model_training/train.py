"""
Training script: build preprocessor, train XGBoost model, and save artifacts.

This file is intentionally compact; some functions may be refactored later
to reduce local variable count.
"""
# pylint: disable=R0914


import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, recall_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from model_training.preprocess import (
    load_raw_data,
    create_target,
    clean_term_column,
    apply_domain_imputation,
    build_preprocessor,
    FEATURES_TO_USE
)

# ------------------------------------------------------
# MLflow DagsHub Tracking
# ------------------------------------------------------

mlflow.set_tracking_uri("https://dagshub.com/nazzzzzz096/Loan-prediction.mlflow")
mlflow.set_experiment("Loan-Prediction")

# ------------------------------------------------------
# Paths
# ------------------------------------------------------

DATA_PATH = "model_training/data/sample_200k.csv"
MODEL_OUTPUT_DIR = "model_training/artifacts"
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)


def train_model():
    """Run full training: preprocess, SMOTE, fit XGBoost, evaluate, and log artifacts."""

    print("📌 Loading 200k sample...")
    df = load_raw_data(DATA_PATH)

    print("📌 Creating target...")
    df = create_target(df)

    print("📌 Cleaning term column...")
    df = clean_term_column(df)

    print("📌 Applying domain imputations...")
    df = apply_domain_imputation(df)

    print("📌 Selecting features...")
    df = df[FEATURES_TO_USE + ["is_default"]]

    x = df[FEATURES_TO_USE]
    y = df["is_default"]

    print(f"📌 Final data shape: {x.shape}")

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    print("📌 Building preprocessor...")
    preprocessor = build_preprocessor(x_train)

    print("📌 Fitting preprocessor...")
    preprocessor.fit(x_train)

    x_train_trans = preprocessor.transform(x_train)
    x_test_trans = preprocessor.transform(x_test)

    # SMOTE
    print("📌 Applying SMOTE...")
    sm = SMOTE(random_state=42)
    x_train_res, y_train_res = sm.fit_resample(x_train_trans, y_train)

    # Train XGBoost
    print("📌 Training XGBoost...")
    model = XGBClassifier(
        n_estimators=450,
        max_depth=6,
        learning_rate=0.06,
        eval_metric="logloss"
    )

    model.fit(x_train_res, y_train_res)

    # Evaluation
    print("📌 Evaluating model...")
    y_proba = model.predict_proba(x_test_trans)[:, 1]
    threshold = 0.20
    y_pred = (y_proba > threshold).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"AUC: {auc:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Recall: {recall:.4f}")

    # MLflow Logging
    with mlflow.start_run():

        mlflow.log_metric("auc", auc)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("recall", recall)

        mlflow.log_params({
            "n_estimators": 450,
            "max_depth": 6,
            "learning_rate": 0.06,
            "threshold": threshold,
            "features_used": len(FEATURES_TO_USE)
        })

        # Artifacts
        pre_path = f"{MODEL_OUTPUT_DIR}/preprocessor.pkl"
        model_path = f"{MODEL_OUTPUT_DIR}/model.pkl"

        joblib.dump(preprocessor, pre_path)
        joblib.dump(model, model_path)

        mlflow.log_artifact(pre_path)
        mlflow.log_artifact(model_path)

    print("🎉 Training complete! Model + Preprocessor saved.")


if __name__ == "__main__":
    train_model()

