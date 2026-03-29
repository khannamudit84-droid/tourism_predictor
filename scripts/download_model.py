from huggingface_hub import hf_hub_download
import shutil
import os

MODEL_REPO = "Mudit1984/tourism_project2"

print("📥 Downloading model...")

os.makedirs("models", exist_ok=True)

model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename="best_model.pkl"
)

shutil.copy(model_path, "models/best_model.pkl")

print("✅ Model ready")
