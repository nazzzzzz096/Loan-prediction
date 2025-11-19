import pandas as pd
from model_training.preprocess import (
    clean_term_column,
    apply_domain_imputation,
    build_preprocessor,
    FEATURES_TO_USE
)

def test_preprocess_pipeline():
    # Minimal sample input
    sample = pd.DataFrame([{
        "loan_amnt": 12000,
        "term": "36 months",
        "int_rate": 12.5,
        "installment": 400,
        "grade": "C",
        "purpose": "credit_card",
        "annual_inc": 55000,
        "verification_status": "Verified",
        "home_ownership": "RENT",
        "dti": 18.2,
        "open_acc": 4,
        "total_acc": 10,
        "revol_bal": 8000,
        "mort_acc": 0,
        "inq_last_12m": 1,
        "mths_since_recent_inq": 3,
        "delinq_2yrs": 0,
        "mths_since_last_delinq": None,
        "mths_since_last_major_derog": None,
        "pub_rec_bankruptcies": 0
    }])

    # Step 1: term cleaning
    sample = clean_term_column(sample)
    assert sample["term"][0] == 36.0

    # Step 2: domain imputation
    sample = apply_domain_imputation(sample)
    assert sample["mths_since_last_delinq"][0] == 999
    assert sample["mths_since_last_major_derog"][0] == 999

    # Step 3: preprocessing pipeline must produce a matrix
    pre = build_preprocessor(sample)
    transformed = pre.fit_transform(sample)

    assert transformed.shape[0] == 1
