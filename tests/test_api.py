from fastapi.testclient import TestClient
from api.app.main import app

client = TestClient(app)

def test_api_predict():
    
    payload = {
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
        "total_acc": 20,
        "revol_bal": 8000,
        "mort_acc": 2,
        "inq_last_12m": 1,
        "mths_since_recent_inq": 5,
        "delinq_2yrs": 0,
        "mths_since_last_delinq": 999,
        "mths_since_last_major_derog": 999,
        "pub_rec_bankruptcies": 0
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "risk_score" in data
    assert "prediction" in data
