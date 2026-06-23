# lstm_preprocess.py
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
import torch
import pickle
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from tqdm import tqdm

from config import DATA_DIR, FEATURES_DIR, LABEL_COL

nltk.download('punkt', quiet=True)

def load_raw_dataset(prefix):
    """Load các file CSV theo tiền tố (both, posts, titles)"""
    train_df = pd.read_csv(os.path.join(DATA_DIR, "processed", f"{prefix}_train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "processed", f"{prefix}_val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "processed", f"{prefix}_test.csv"))
    return train_df, val_df, test_df

def build_vocab(train_texts):
    """Xây dựng từ điển, loại bỏ từ xuất hiện 1 lần"""
    print("Building vocabulary from training set...")
    all_tokens = []
    for text in tqdm(train_texts, desc="Tokenizing train texts"):
        all_tokens.extend(word_tokenize(str(text).lower()))
    
    token_counts = Counter(all_tokens)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for token, count in token_counts.items():
        if count > 1:
            vocab[token] = idx
            idx += 1
            
    print(f"Vocabulary size: {len(vocab)}")
    return vocab

def encode_and_pad(texts, labels, vocab, max_len):
    """Chuyển text thành số và đệm độ dài"""
    input_ids = []
    unk_idx = vocab['<UNK>']
    pad_idx = vocab['<PAD>']
    
    for text in tqdm(texts, desc="Encoding sequences"):
        tokens = word_tokenize(str(text).lower())
        token_ids = [vocab.get(token, unk_idx) for token in tokens]
        
        if len(token_ids) < max_len:
            token_ids = token_ids + [pad_idx] * (max_len - len(token_ids))
        else:
            token_ids = token_ids[:max_len]
            
        input_ids.append(token_ids)
        
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels.values, dtype=torch.long)

def main():
    configs = [
        ("both", "text", 512),
        ("posts", "post", 512),
        ("titles", "title", 35)
    ]

    for prefix, text_col, max_len in configs:
        print(f"\n{'='*50}")
        print(f"PREPROCESSING LSTM DATASET: {prefix.upper()}")
        print(f"{'='*50}")
        
        train_df, val_df, test_df = load_raw_dataset(prefix)
        
        # 1. Build & Save Vocab
        vocab = build_vocab(train_df[text_col].tolist())
        
        save_dir = os.path.join(FEATURES_DIR, f"lstm_{prefix}_preprocessed")
        os.makedirs(save_dir, exist_ok=True)
        
        vocab_path = os.path.join(save_dir, "vocab.pkl")
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)
            
        # 2. Encode & Pad
        print(f"Encoding datasets (Max Length: {max_len})...")
        train_inputs, train_labels = encode_and_pad(train_df[text_col], train_df[LABEL_COL], vocab, max_len)
        val_inputs, val_labels = encode_and_pad(val_df[text_col], val_df[LABEL_COL], vocab, max_len)
        test_inputs, test_labels = encode_and_pad(test_df[text_col], test_df[LABEL_COL], vocab, max_len)
        
        # 3. Save Tensors
        print(f"Saving tensors to {save_dir}...")
        torch.save((train_inputs, train_labels), os.path.join(save_dir, "train.pt"))
        torch.save((val_inputs, val_labels), os.path.join(save_dir, "val.pt"))
        torch.save((test_inputs, test_labels), os.path.join(save_dir, "test.pt"))

    print("\nDone. All LSTM datasets preprocessed and saved.")

if __name__ == "__main__":
    main()