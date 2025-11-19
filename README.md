 Loan Default Prediction – End-to-End MLOps Project

This project is an end-to-end Credit Risk / Loan Default Prediction system built using modern MLOps practices.
It includes data preprocessing, model training (XGBoost), MLflow tracking, FastAPI deployment, Dockerization, unit testing, linting, and CI/CD automation.

📌 Project Features Overview
🔹 Machine Learning

Uses Lending Club dataset (cleaned 200k sample)

    Feature engineering & preprocessing

    Handles missing values with domain rules

    Categorical encoding + numerical scaling

    Class imbalance solved using SMOTE

    Trained using XGBoost

    Evaluation metrics: AUC, F1, Recall

    MLflow tracking enabled

🔹 Model Serving (API)

    FastAPI endpoint: /predict

    Loads model + preprocessor artifacts

    Dockerized & ready for deployment

    Works inside GitHub Actions AND local

🔹 Code Quality

    Unit tests with pytest

    Linting with pylint

    90% code quality score (9.43/10)

🔹 CI/CD Pipeline

    GitHub Actions workflow includes:

    Install dependencies

    Run linting

    Run unit tests

    Build Docker image

    Push to DockerHub

🧠 Model Overview
   ⚙ Pipeline

    Load 200k rows → clean → feature selection

    Fix term column (“36 months” → 36)

    Fill missing values (domain-specific logic)

   Preprocessing:

     Numeric: SimpleImputer + StandardScaler

     Categorical: MostFrequentImputer + OneHotEncoder

  SMOTE oversampling

  Train XGBoostClassifier

  Evaluate (AUC, F1, Recall)

Save:

model.pkl

preprocessor.pkl
