from huggingface_hub import login, upload_file
import os

login(token=os.environ["HF_TOKEN"])

upload_file(
    path_or_fileobj="tourism_project2/data/tourism.csv",
    path_in_repo="data/raw/tourism.csv",
    repo_id="Mudit1984/tourism_project2",
    repo_type="dataset"
)

print("✅ Dataset uploaded to Hugging Face")
