import pandas as pd
import joblib

print("🧪 Testing model...")

# Load model
model = joblib.load("models/best_model.pkl")

# Load sample data
df = pd.read_csv("data/train.csv").head(5)

X = df.drop("ProdTaken", axis=1, errors="ignore")

# Predict
preds = model.predict(X)

print("✅ Predictions:", preds)
