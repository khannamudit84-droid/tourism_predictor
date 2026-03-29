import pandas as pd
import joblib
import os

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from huggingface_hub import login, upload_file


# ==============================
# 1. LOGIN
# ==============================
login(token=os.getenv("HF_TOKEN"))


# ==============================
# 2. LOAD DATA FROM HF
# ==============================
print("📥 Loading dataset from Hugging Face...")

dataset = load_dataset("Mudit1984/tourism_project2")
train_df = dataset["train"].to_pandas()

if "test" in dataset:
    test_df = dataset["test"].to_pandas()
    print("✅ Using train-test split from HF")
else:
    print("⚠️ No test split found → creating one")
    train_df, test_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,
        stratify=train_df["ProdTaken"]
    )


# ==============================
# 3. DATA CLEANING
# ==============================
train_df.drop(columns=["CustomerID"], errors="ignore", inplace=True)
test_df.drop(columns=["CustomerID"], errors="ignore", inplace=True)

numeric_cols = train_df.select_dtypes(include="number").columns
train_medians = train_df[numeric_cols].median()

train_df[numeric_cols] = train_df[numeric_cols].fillna(train_medians)
test_df[numeric_cols] = test_df[numeric_cols].fillna(train_medians)

train_df.fillna("Unknown", inplace=True)
test_df.fillna("Unknown", inplace=True)


# ==============================
# 4. FEATURE SPLIT
# ==============================
X_train = train_df.drop("ProdTaken", axis=1)
y_train = train_df["ProdTaken"]

X_test = test_df.drop("ProdTaken", axis=1)
y_test = test_df["ProdTaken"]


# ==============================
# 5. PIPELINE + IMPROVED MODEL
# ==============================
cat_cols = X_train.select_dtypes(include="object").columns
num_cols = X_train.select_dtypes(exclude="object").columns

preprocessor = ColumnTransformer([
    ("num", "passthrough", num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42
    ))
])


# ==============================
# 6. TRAIN
# ==============================
pipeline.fit(X_train, y_train)
print("✅ Model trained")


# ==============================
# 7. EVALUATION
# ==============================
y_pred = pipeline.predict(X_test)

print("\n📊 Model Evaluation:")
print(classification_report(y_test, y_pred))


# ==============================
# 8. SAVE MODEL
# ==============================
os.makedirs("models", exist_ok=True)

model_path = "models/best_model.pkl"
joblib.dump(pipeline, model_path)

print("✅ Model saved")


# ==============================
# 9. UPLOAD MODEL TO HF
# ==============================
upload_file(
    path_or_fileobj=model_path,
    path_in_repo="best_model.pkl",
    repo_id="Mudit1984/tourism_project2",
    repo_type="model"
)

print("✅ Model uploaded to Hugging Face")
