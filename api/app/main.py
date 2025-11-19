"""FastAPI app for loan default prediction service."""

from fastapi import FastAPI
from api.app.schemas import LoanApplication
from api.app.predictor import CreditRiskModel


app = FastAPI(title="Loan Default Prediction API")

# Load model once
credit_model = CreditRiskModel()

@app.post("/predict")
def predict_loan_default(payload: LoanApplication):
    """ used for prediction using trained ML model"""
    data = payload.dict()
    result = credit_model.predict(data)
    return result

