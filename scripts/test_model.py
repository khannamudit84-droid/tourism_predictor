import joblib
import pandas as pd
import numpy as np

print("🧪 Running model tests...")

model = joblib.load("models/best_model.pkl")

# Create synthetic input (safe for production)
sample = {
    "Age": 35,
    "TypeofContact": "Company Invited",
    "CityTier": 1,
    "DurationOfPitch": 10,
    "Occupation": "Salaried",
    "Gender": "Male"
}

df = pd.DataFrame([sample])

# Predict
probs = model.predict_proba(df)[:, 1]
preds = (probs > 0.3).astype(int)

# ==============================
# ASSERTIONS (IMPORTANT)
# ==============================
assert len(preds) == 1, "Prediction failed"
assert not np.isnan(preds).any(), "NaN predictions"
assert preds[0] in [0, 1], "Invalid class output"

print("✅ Test Passed")
print("Probability:", probs[0])
print("Prediction:", preds[0])
