import joblib
import pandas as pd
import numpy as np
from datasets import load_dataset

print("🧪 Running dynamic model tests...")

# ==============================
# LOAD MODEL
# ==============================
model = joblib.load("models/best_model.pkl")

# ==============================
# GET EXPECTED SCHEMA
# ==============================
expected_cols = model.named_steps["preprocessor"].feature_names_in_

def align_input(data_dict):
    df = pd.DataFrame(columns=expected_cols)

    # Fill defaults
    for col in df.columns:
        df[col] = 0

    # Override with actual values
    for k, v in data_dict.items():
        if k in df.columns:
            df.loc[0, k] = v

    return df

# ==============================
# 1. TEST WITH REAL DATA
# ==============================
print("\n🔍 Real Data Samples:")

dataset = load_dataset("Mudit1984/tourism_project2")
df_real = dataset["train"].to_pandas().drop("ProdTaken", axis=1)

samples = df_real.sample(3, random_state=42)

for i, row in samples.iterrows():
    df = align_input(row.to_dict())
    prob = model.predict_proba(df)[0][1]
    pred = int(prob > 0.3)
    print(f"Prob: {prob:.3f} → Pred: {pred}")

# ==============================
# 2. LOW VALUE CUSTOMER
# ==============================
low_value = {
    "Age": 45,
    "TypeofContact": "Company Invited",
    "CityTier": 3,
    "DurationOfPitch": 5,
    "Occupation": "Unemployed",
    "Gender": "Male",
    "NumberOfPersonVisiting": 1,
    "NumberOfFollowups": 1,
    "PreferredPropertyStar": 2,
    "NumberOfTrips": 0,
    "Passport": 0,
    "PitchSatisfactionScore": 2,
    "OwnCar": 0,
    "NumberOfChildrenVisiting": 2,
    "MonthlyIncome": 15000
}

# ==============================
# 3. HIGH VALUE CUSTOMER
# ==============================
high_value = {
    "Age": 30,
    "TypeofContact": "Self Enquiry",
    "CityTier": 1,
    "DurationOfPitch": 20,
    "Occupation": "Professional",
    "Gender": "Female",
    "NumberOfPersonVisiting": 3,
    "NumberOfFollowups": 5,
    "PreferredPropertyStar": 5,
    "NumberOfTrips": 6,
    "Passport": 1,
    "PitchSatisfactionScore": 5,
    "OwnCar": 1,
    "NumberOfChildrenVisiting": 0,
    "MonthlyIncome": 150000
}

print("\n📉 Low Value Customer:")
df = align_input(low_value)
print("Prob:", model.predict_proba(df)[0][1])

print("\n📈 High Value Customer:")
df = align_input(high_value)
print("Prob:", model.predict_proba(df)[0][1])

print("\n✅ All tests completed")
