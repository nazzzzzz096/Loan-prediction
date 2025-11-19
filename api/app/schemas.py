"""Pydantic request schema for loan application payloads."""

from pydantic import BaseModel

class LoanApplication(BaseModel):
    """Pydantic model describing the loan application input for /predict."""
    loan_amnt: float
    term: float
    int_rate: float
    installment: float
    grade: str
    purpose: str

    annual_inc: float
    verification_status: str
    home_ownership: str
    dti: float

    open_acc: float
    total_acc: float
    revol_bal: float
    mort_acc: float
    inq_last_12m: float
    mths_since_recent_inq: float

    delinq_2yrs: float
    mths_since_last_delinq: float
    mths_since_last_major_derog: float
    pub_rec_bankruptcies: float
