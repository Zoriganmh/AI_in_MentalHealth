import pandas as pd
import os
import re
from config import DATA_DIR

def clean_text(text):
    text = str(text) if pd.notna(text) else ""
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    # Giữ lại các ký tự hợp lệ
    text = re.sub(r'[^a-zA-Z0-9\s.!?,\-;:()\[\]{}"]', '', text)
    return text

def preprocess_and_save(input_prefix, text_cols):
    """
    input_prefix: 'both', 'posts', hoặc 'titles'
    text_cols: danh sách các cột cần clean (vd: ['title', 'post'])
    """
    for split in ['train', 'val', 'test']:
        file_name = f"{input_prefix}_{split}.csv"
        input_path = os.path.join(DATA_DIR, "original", file_name)
        
        # Nếu đang chạy trên thư mục hiện tại không có DATA_DIR/original, đổi lại đường dẫn
        if not os.path.exists(input_path):
             input_path = file_name # fallback đọc trực tiếp file
        
        if not os.path.exists(input_path):
            print(f"Warning: Không tìm thấy file {input_path}")
            continue

        df = pd.read_csv(input_path)
        
        # Clean các cột văn bản
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].apply(clean_text)
                
        # Nếu là file "both", ta tự động tạo cột "text" gộp từ title và post theo logic của bài báo
        if input_prefix == 'both':
            df['text'] = df['title'] + ": " + df['post']
            
        # Lưu vào thư mục processed
        save_dir = os.path.join(DATA_DIR, "processed")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)
        
        df.to_csv(save_path, index=False)
        print(f"Processed and saved: {save_path}")

def preprocess_data():
    print("--- Preprocessing BOTH (Title + Post) ---")
    preprocess_and_save('both', ['title', 'post'])
    
    print("\n--- Preprocessing POSTS ---")
    preprocess_and_save('posts', ['post'])
    
    print("\n--- Preprocessing TITLES ---")
    preprocess_and_save('titles', ['title'])

if __name__ == "__main__":
    preprocess_data()