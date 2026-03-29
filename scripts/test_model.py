import joblib
import pandas as pd
import numpy as np

print("🧪 Running model tests...")

# Load model
model = joblib.load("models/best_model.pkl")

# ==============================
# GET EXPECTED FEATURES
# ==============================
expected_cols = model.named_steps["preprocessor"].feature_names_in_

# Create empty dataframe with all columns
df = pd.DataFrame(columns=expected_cols)

# Fill with default values
for col in df.columns:
    df[col] = 0  # default numeric

# Override some realistic values
df.loc[0, "Age"] = 35
df.loc[0, "TypeofContact"] = "Company Invited"
df.loc[0, "CityTier"] = 1

# Convert categorical properly
df = df.astype(object)

# ==============================
# PREDICT
# ==============================
probs = model.predict_proba(df)[:, 1]
preds = (probs > 0.3).astype(int)

# ==============================
# ASSERTIONS
# ==============================
assert len(preds) == 1, "Prediction failed"
assert not np.isnan(preds).any(), "NaN predictions"
assert preds[0] in [0, 1], "Invalid output"

print("✅ Test Passed")
print("Probability:", probs[0])
print("Prediction:", preds[0])
