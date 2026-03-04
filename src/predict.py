import pandas as pd
import joblib
import os
import numpy as np

print("✅ Script started...")

# Load saved model
model_path = os.path.join(os.path.dirname(__file__), "best_churn_model.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError("❌ best_churn_model.pkl not found in src/")
else:
    print(f"✅ Model found at: {model_path}")
    model = joblib.load(model_path)
    print("✅ Model loaded successfully.")

def predict_churn(input_dict):
    """Predict churn for a new customer (dictionary input)."""
    new_data = pd.DataFrame([input_dict])

    # Clean categorical and numeric consistency
    for c in new_data.columns:
        if c.lower() in ["tenure", "monthlycharges", "totalcharges", "seniorcitizen"]:
            new_data[c] = pd.to_numeric(new_data[c], errors="coerce")
        if new_data[c].dtype == "object":
            new_data[c] = new_data[c].astype(str).str.strip().str.lower()

    new_data = new_data.fillna(0)

    # Make prediction
    prediction = model.predict(new_data)[0]
    probability = model.predict_proba(new_data)[0][1]

    result = "Churn" if prediction == 1 else "No Churn"
    print(f"\n🧾 Prediction: {result}")
    print(f"🔹 Churn Probability: {probability:.2f}\n")
    return result, probability

if __name__ == "__main__":
    # Example test input
    sample_customer = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'Yes',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 85.65,
        'TotalCharges': 1025.3
    }

    predict_churn(sample_customer)
