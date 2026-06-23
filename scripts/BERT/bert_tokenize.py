# bert_tokenize.py
import sys
import os

# --- THÊM ĐOẠN NÀY LÊN TRÊN CÙNG ---
# Lấy đường dẫn hiện tại của file này
current_dir = os.path.dirname(os.path.abspath(__file__))
# Lùi ra 2 cấp (ví dụ: từ models/bert lùi ra models, lùi tiếp ra gốc)
root_dir = os.path.dirname(os.path.dirname(current_dir))
# Thêm thư mục gốc vào hệ thống để Python tìm thấy config.py
sys.path.append(root_dir)
# -----------------------------------
import pandas as pd
from datasets import load_dataset
from transformers import BertTokenizerFast
from config import (
    DATA_DIR,
    LABEL_COL,
    FEATURES_DIR,
)

def load_raw_dataset(prefix):
    """Load dataset linh hoạt theo tiền tố (both, posts, titles)"""
    data_files = {
        "train": os.path.join(DATA_DIR, "processed", f"{prefix}_train.csv"),
        "validation": os.path.join(DATA_DIR, "processed", f"{prefix}_val.csv"),
        "test": os.path.join(DATA_DIR, "processed", f"{prefix}_test.csv"),
    }
    dataset = load_dataset("csv", data_files=data_files)
    return dataset

def tokenize_dataset(dataset, tokenizer, text_col, label_col, max_length):
    def tokenize(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize, batched=True)

    if label_col in tokenized["train"].column_names:
        tokenized = tokenized.rename_column(label_col, "labels")

    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    return tokenized

def main():
    print("Loading BERT tokenizer...")
    # Sử dụng BERT base uncased
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # Cấu hình 3 luồng: (tên_tiền_tố, cột_văn_bản, chiều_dài_tối_đa)
    configs = [
        ("both", "text", 512),
        ("posts", "post", 512),
        ("titles", "title", 35)
    ]

    for prefix, text_col, max_len in configs:
        print(f"\n{'='*50}")
        print(f"PROCESSING DATASET: {prefix.upper()} (BERT)")
        print(f"{'='*50}")
        
        print(f"Loading raw CSV datasets for {prefix}...")
        dataset = load_raw_dataset(prefix)

        print(f"Tokenizing {prefix} datasets (Max Length: {max_len})...")
        tokenized_dataset = tokenize_dataset(
            dataset=dataset,
            tokenizer=tokenizer,
            text_col=text_col,
            label_col=LABEL_COL,
            max_length=max_len,
        )

        tokens_path = os.path.join(FEATURES_DIR, f'bert_{prefix}_tokenized')
        if not os.path.exists(tokens_path):  
            os.makedirs(tokens_path)

        print(f"Saving tokenized {prefix} dataset to {tokens_path} ...")
        tokenized_dataset.save_to_disk(tokens_path)

    print("\nDone. All BERT tokenized datasets saved.")

if __name__ == "__main__":
    main()