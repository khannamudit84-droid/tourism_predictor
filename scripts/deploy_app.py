from huggingface_hub import login, upload_file
import os

login(token=os.environ["HF_TOKEN"])

SPACE_REPO = "Mudit1984/tourism-predictor1"

upload_file("app.py", "app.py", SPACE_REPO, repo_type="space")
upload_file("requirements.txt", "requirements.txt", SPACE_REPO, repo_type="space")

print("✅ App deployed to Hugging Face Space")
