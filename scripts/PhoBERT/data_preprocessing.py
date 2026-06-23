import sys
import os
import pandas as pd
import re
import unicodedata

# 1. CHỈ ĐƯỜNG CHO PYTHON TÌM CONFIG TRONG THƯ MỤC SCRIPTS
current_dir = os.path.dirname(os.path.abspath(__file__)) # Thư mục PhoBERT
scripts_dir = os.path.dirname(current_dir)               # Lùi ra 1 cấp là thư mục scripts
sys.path.insert(0, scripts_dir)                          # Ưu tiên tìm trong scripts

# 2. Bây giờ import config sẽ thành công 100%
from config import DATA_DIR

def clean_text(text):
    text = str(text) if pd.notna(text) else ""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    vietnamese_chars = "ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ"
    pattern = r'[^a-zA-Z0-9\s.!?,\-;:()\[\]{}"' + vietnamese_chars + r']'
    text = re.sub(pattern, '', text)
    text = ' '.join(text.split())
    return text

def preprocess_file(input_filename, output_filename):
    input_path = os.path.join(DATA_DIR, "original", input_filename)
    
    if not os.path.exists(input_path):
         input_path = input_filename
         
    if not os.path.exists(input_path):
        print(f"Lỗi: Không tìm thấy file gốc {input_path}")
        return

    print(f"Đang xử lý: {input_filename}...")
    df = pd.read_csv(input_path)
    
    if 'title' in df.columns and 'post' in df.columns and 'text' not in df.columns:
        df['text'] = df['title'] + ": " + df['post']
        
    if 'text' in df.columns:
        df['text'] = df['text'].apply(clean_text)
    else:
        print("Cảnh báo: Không tìm thấy cột 'text' trong dữ liệu.")
        
    save_dir = os.path.join(DATA_DIR, "processed")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, output_filename)
    
    df.to_csv(save_path, index=False, encoding='utf-8')
    print(f"Đã lưu thành công: {save_path}\n")

def main():
    print("=== BẮT ĐẦU TIỀN XỬ LÝ DỮ LIỆU ===")
    
    file_mapping = [
        ("formatted_train_vi.csv", "train_vi.csv"),
        ("formatted_val_vi.csv", "val_vi.csv"),
        ("formatted_test_vi.csv", "test_vi.csv")
    ]
    
    for in_file, out_file in file_mapping:
        preprocess_file(in_file, out_file)
        
    print("=== HOÀN THÀNH TIỀN XỬ LÝ ===")

if __name__ == "__main__":
    main()