import pandas as pd
import joblib

print("🧪 Testing model...")

# Load model
model = joblib.load("models/best_model.pkl")

# Use random sample (better than head)
df = pd.read_csv("data/train.csv").sample(10, random_state=42)

X = df.drop("ProdTaken", axis=1, errors="ignore")

# Predict probabilities
probs = model.predict_proba(X)[:, 1]

# Business threshold (important)
threshold = 0.3
preds = (probs > threshold).astype(int)

print("\n📊 Results:")
print("Probabilities:", probs)
print("Predictions:", preds)
