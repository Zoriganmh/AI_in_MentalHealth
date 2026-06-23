import sys
import os

# --- CHỈ ĐƯỜNG CHO PYTHON TÌM CONFIG TRONG THƯ MỤC SCRIPTS ---
current_dir = os.path.dirname(os.path.abspath(__file__)) # Đang ở PhoBERT
scripts_dir = os.path.dirname(current_dir)               # Lùi ra 1 cấp (scripts)
sys.path.insert(0, scripts_dir)                          # Ưu tiên tìm trong scripts
# -----------------------------------------------------------

from pyvi import ViTokenizer
from datasets import load_dataset
from transformers import AutoTokenizer

from config import DATA_DIR, LABEL_COL, FEATURES_DIR, PHOBERT_MODEL_NAME, MAX_LENGTH

def tokenize_vietnamese(text):
    return ViTokenizer.tokenize(str(text))

def main():
    print(f"Loading PhoBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)

    data_files = {
        "train": os.path.join(DATA_DIR, "processed", "train_vi.csv"),
        "validation": os.path.join(DATA_DIR, "processed", "val_vi.csv"),
        "test": os.path.join(DATA_DIR, "processed", "test_vi.csv"),
    }
    
    print("Loading dataset from processed directory...")
    dataset = load_dataset("csv", data_files=data_files)

    print("Applying ViTokenizer (Word Segmentation)...")
    dataset = dataset.map(lambda x: {"text": tokenize_vietnamese(x["text"])})

    print(f"Tokenizing dataset (Max Length: {MAX_LENGTH})...")
    def tokenize_func(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    tokenized_dataset = dataset.map(tokenize_func, batched=True)

    if LABEL_COL in tokenized_dataset["train"].column_names:
        tokenized_dataset = tokenized_dataset.rename_column(LABEL_COL, "labels")

    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    tokens_path = os.path.join(FEATURES_DIR, 'phobert_dataset_tokenized')
    os.makedirs(tokens_path, exist_ok=True)

    print(f"Saving tokenized dataset to {tokens_path} ...")
    tokenized_dataset.save_to_disk(tokens_path)
    print("\nDone. Tokenized dataset saved.")

if __name__ == "__main__":
    main()