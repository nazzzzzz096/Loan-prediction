"""
Preprocessing helpers for the LendingClub credit-risk project.

Contains:
- loading & cleaning functions
- domain-specific imputations
- preprocessing pipeline builder
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ------------------------------------------------------
# Target labels
# ------------------------------------------------------

DEFAULT_STATES = [
    "Charged Off",
    "Default",
    "Late (31-120 days)",
    "Late (16-30 days)"
]

# ------------------------------------------------------
# FINAL 20 FEATURES for Lending Club dataset
# ------------------------------------------------------

FEATURES_TO_USE = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "purpose",

    "annual_inc",
    "verification_status",
    "home_ownership",
    "dti",

    "open_acc",
    "total_acc",
    "revol_bal",
    "mort_acc",
    "inq_last_12m",
    "mths_since_recent_inq",

    "delinq_2yrs",
    "mths_since_last_delinq",
    "mths_since_last_major_derog",
    "pub_rec_bankruptcies",
]

# ------------------------------------------------------
# Load data
# ------------------------------------------------------

def load_raw_data(path: str):
    """fo loading the data"""
    return pd.read_csv(path, low_memory=False)

# ------------------------------------------------------
# Create target column
# ------------------------------------------------------

def create_target(df: pd.DataFrame):
    """ creating a new cloum which is used for target since there is no primary column can be used for classifying"""
    df["is_default"] = df["loan_status"].isin(DEFAULT_STATES).astype(int)
    return df

# ------------------------------------------------------
# Clean term column ("36 months" -> 36)
# ------------------------------------------------------

def clean_term_column(df: pd.DataFrame):
    """Convert '36 months' to integer 36."""
    if "term" in df.columns:
        df["term"] = df["term"].astype(str).str.extract(r"(\d+)").astype(float)
    return df

# ------------------------------------------------------
# Domain-specific imputations (credit bureau logic)
# ------------------------------------------------------

def apply_domain_imputation(df: pd.DataFrame):
    """ used this because they is issue with the metrics which was
     low so fill that with some values and zeros than null"""

    # Missing delinquency months = borrower is clean
    fill_999 = [
        "mths_since_last_delinq",
        "mths_since_last_major_derog"
    ]

    # Missing counts = no delinquencies
    fill_zero = [
        "delinq_2yrs",
        "pub_rec_bankruptcies",
        "inq_last_12m",
        "mths_since_recent_inq"
    ]

    for col in fill_999:
        if col in df.columns:
            df[col] = df[col].fillna(999)

    for col in fill_zero:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df

# ------------------------------------------------------
# Preprocessing Pipeline
# ------------------------------------------------------

def build_preprocessor(x: pd.DataFrame):
    """ used to build the pipeline which connects all the preprocess steps"""
    numeric_features = x.select_dtypes(include=["float64", "int64"]).columns.tolist()
    categorical_features = x.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ],
        remainder="drop"
    )


