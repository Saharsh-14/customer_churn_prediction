import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import joblib

# Load data
data = pd.read_csv(r"C:\Users\Asus\OneDrive\Desktop\customer_churn_prediction\data\telco_customer_churn.csv")

# ✅ Convert numeric columns properly
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data["MonthlyCharges"] = pd.to_numeric(data["MonthlyCharges"], errors="coerce")
data["tenure"] = pd.to_numeric(data["tenure"], errors="coerce")
data["SeniorCitizen"] = pd.to_numeric(data["SeniorCitizen"], errors="coerce")

# Fill NaN numerics
data.fillna({"TotalCharges": data["TotalCharges"].median(),
             "MonthlyCharges": data["MonthlyCharges"].median(),
             "tenure": data["tenure"].median(),
             "SeniorCitizen": 0}, inplace=True)

# Drop ID column
if "customerID" in data.columns:
    data = data.drop("customerID", axis=1)

# Target column
data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

# ✅ Clean inconsistent categories
replace_cols = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
]
for col in replace_cols:
    data[col] = data[col].replace({"No internet service": "No", "No phone service": "No"})

# ✅ Clean category text
for col in data.select_dtypes(include="object").columns:
    data[col] = data[col].str.strip().str.lower()

# Split features/target
X = data.drop("Churn", axis=1)
y = data["Churn"]

# Detect columns correctly now
cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

print("✅ Detected categorical columns:", list(cat_cols))
print("✅ Detected numeric columns:", list(num_cols))

# Preprocessor
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", StandardScaler(), num_cols)
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(eval_metric="logloss")
}

results = {}

for name, model in models.items():
    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    results[name] = (acc, f1)

print("\n📊 Model Performance:")
for name, (acc, f1) in results.items():
    print(f"{name:<20} Accuracy: {acc:.3f} | F1: {f1:.3f}")

# Pick best model
best_model = max(results, key=lambda x: results[x][1])
best_acc, best_f1 = results[best_model]
print(f"\n🏆 Best Model: {best_model} (Accuracy={best_acc:.3f}, F1={best_f1:.3f})")

# Retrain on all data
final_model = Pipeline([("preprocess", preprocess), ("model", models[best_model])])
final_model.fit(X, y)

# Save new model
joblib.dump(final_model, "best_churn_model.pkl")
print("✅ Model retrained and saved successfully as best_churn_model.pkl")
