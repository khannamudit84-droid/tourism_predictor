from datasets import load_dataset
import os

print("📥 Downloading dataset from Hugging Face...")

DATASET_NAME = "Mudit1984/tourism_project2"

dataset = load_dataset(DATASET_NAME)

os.makedirs("data", exist_ok=True)

# Save train
train_df = dataset["train"].to_pandas()
train_df.to_csv("data/train.csv", index=False)

# Save test if exists
if "test" in dataset:
    test_df = dataset["test"].to_pandas()
    test_df.to_csv("data/test.csv", index=False)
    print("✅ Train & Test saved")
else:
    print("⚠️ No test split found")

print("Train shape:", train_df.shape)
