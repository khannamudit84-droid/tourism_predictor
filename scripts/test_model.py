import pandas as pd
import joblib
import sklearn

print("🔍 sklearn version:", sklearn.__version__)

model = joblib.load("models/best_model.pkl")

df = pd.read_csv("data/train.csv").head(5)
X = df.drop("ProdTaken", axis=1, errors="ignore")

preds = model.predict(X)

print("✅ Predictions:", preds)
