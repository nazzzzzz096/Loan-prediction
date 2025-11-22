import pandas as pd
import joblib

def test_model_can_predict():
    model = joblib.load("model_training/model_training/artifacts/model.pkl")
    preprocessor = joblib.load("model_training/model_training/artifacts/preprocessor.pkl")

    sample = pd.DataFrame([{
        "loan_amnt": 15000,
        "term": 36,
        "int_rate": 13.5,
        "installment": 350,
        "grade": "C",
        "purpose": "credit_card",
        "annual_inc": 65000,
        "verification_status": "Verified",
        "home_ownership": "RENT",
        "dti": 18.5,
        "open_acc": 7,
        "total_acc": 19,
        "revol_bal": 8000,
        "mort_acc": 2,
        "inq_last_12m": 1,
        "mths_since_recent_inq": 5,
        "delinq_2yrs": 0,
        "mths_since_last_delinq": 999,
        "mths_since_last_major_derog": 999,
        "pub_rec_bankruptcies": 0
    }])

    X = preprocessor.transform(sample)
    prob = model.predict_proba(X)[0][1]

    assert 0 <= prob <= 1
