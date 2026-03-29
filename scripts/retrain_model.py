import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report

from huggingface_hub import login, upload_file


# ==============================
# 1. CONFIG
# ==============================
DATASET_NAME = "Mudit1984/tourism_project2"
MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/best_model.pkl"
F1_THRESHOLD = 0.6

os.makedirs(MODEL_DIR, exist_ok=True)

# ==============================
# 2. LOGIN (HF)
# ==============================
login(token=os.getenv("HF_TOKEN"))

# ==============================
# 3. MLflow Setup
# ==============================
mlflow.set_experiment("tourism_ml_pipeline")

# ==============================
# 4. LOAD DATA
# ==============================
print("📥 Loading dataset...")
dataset = load_dataset(DATASET_NAME)
train_df = dataset["train"].to_pandas()

if "test" in dataset:
    test_df = dataset["test"].to_pandas()
else:
    train_df, test_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,
        stratify=train_df["ProdTaken"]
    )

# ==============================
# 5. BASIC CLEANING
# ==============================
train_df.drop(columns=["CustomerID"], errors="ignore", inplace=True)
test_df.drop(columns=["CustomerID"], errors="ignore", inplace=True)

X_train = train_df.drop("ProdTaken", axis=1)
y_train = train_df["ProdTaken"]

X_test = test_df.drop("ProdTaken", axis=1)
y_test = test_df["ProdTaken"]

cat_cols = X_train.select_dtypes(include="object").columns
num_cols = X_train.select_dtypes(exclude="object").columns

print(f"Categorical: {len(cat_cols)}, Numerical: {len(num_cols)}")

# ==============================
# 6. PIPELINE (PRODUCTION SAFE)
# ==============================
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ]), num_cols),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# ==============================
# 7. TRAIN + MLflow LOGGING
# ==============================
with mlflow.start_run():

    print("🚀 Training model...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    print("\n📊 Evaluation:")
    print(classification_report(y_test, y_pred))
    print("F1 Score:", f1)

    # Log params
    mlflow.log_params({
        "model": "RandomForest",
        "n_estimators": 200,
        "max_depth": 10
    })

    # Log metric
    mlflow.log_metric("f1_score", f1)

    # Log model
    mlflow.sklearn.log_model(pipeline, "model")

    # ==============================
    # 8. PERFORMANCE GATE
    # ==============================
    if f1 < F1_THRESHOLD:
        raise Exception(f"❌ Model rejected. F1 {f1} < {F1_THRESHOLD}")

    print("✅ Model passed performance gate")

    # ==============================
    # 9. SAVE MODEL
    # ==============================
    joblib.dump(pipeline, MODEL_PATH)
    print("✅ Model saved locally")

    # ==============================
    # 10. PUSH TO HUGGING FACE
    # ==============================
    upload_file(
        path_or_fileobj=MODEL_PATH,
        path_in_repo="best_model.pkl",
        repo_id=DATASET_NAME,
        repo_type="model"
    )

    print("🚀 Model uploaded to Hugging Face (PRODUCTION)")
