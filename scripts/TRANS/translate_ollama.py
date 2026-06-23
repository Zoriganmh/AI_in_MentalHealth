import os
import pandas as pd
import ollama
from tqdm import tqdm
import time

# 1. ĐƯỜNG DẪN THƯ MỤC CỦA BẠN
FOLDER_CHUA_CSV = r"D:\Tai lieu dai hoc\LAB\TramCam\Code\Models\RoBERTa\data\processed"

# 2. TÊN MÔ HÌNH ĐÃ TẢI TRONG OLLAMA
MODEL_NAME = "qwen2.5:7b" 

def translate_with_ollama(text, max_retries=3):
    """Hàm gọi AI Offline qua Ollama để dịch thuật"""
    if pd.isna(text) or str(text).strip() == "":
        return text
        
    prompt = f"""
    Bạn là một dịch giả chuyên nghiệp. Hãy dịch đoạn văn bản tiếng Anh sau sang tiếng Việt.
    Yêu cầu BẮT BUỘC:
    1. Dịch chuẩn xác ngữ nghĩa, không thêm bớt thông tin.
    2. Dùng cấu trúc câu tự nhiên, trôi chảy của người Việt.
    3. Giữ nguyên văn phong chia sẻ thực tế trên các diễn đàn.
    4. CHỈ TRẢ VỀ NỘI DUNG ĐÃ DỊCH. TUYỆT ĐỐI KHÔNG thêm bất kỳ từ ngữ nào khác.
    
    Văn bản cần dịch:
    "{text}"
    """
    
    for attempt in range(max_retries):
        try:
            # Gửi yêu cầu đến Ollama đang chạy trên máy
            response = ollama.chat(model=MODEL_NAME, messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ])
            
            # Lấy kết quả
            result = response['message']['content'].strip()
            
            # Làm sạch nếu AI lỡ thêm dấu ngoặc kép
            if result.startswith('"') and result.endswith('"'):
                result = result[1:-1]
            return result
            
        except Exception as e:
            tqdm.write(f" [!] Lỗi kết nối Ollama: {e}. Đang thử lại...")
            time.sleep(2)
                
    tqdm.write(" [!] Bỏ qua dòng này do lỗi liên tục.")
    return None

# 3. QUY TRÌNH XỬ LÝ FILE (Tương tự như trước)
files_to_process = ["test.csv", "val.csv", "train.csv"]
columns_to_translate = ['title', 'post', 'text'] 

for file_name in files_to_process:
    input_file_path = os.path.join(FOLDER_CHUA_CSV, file_name)
    output_file_path = os.path.join(FOLDER_CHUA_CSV, file_name.replace(".csv", "_vi.csv"))
    
    print(f"\n{'-'*50}\nĐANG XỬ LÝ FILE BẰNG AI OFFLINE: {file_name}\n{'-'*50}")
    
    try:
        df = pd.read_csv(input_file_path)
    except FileNotFoundError:
        print(f"Không tìm thấy file {input_file_path}.")
        continue

    if os.path.exists(output_file_path):
        print(f"Phát hiện file {file_name.replace('.csv', '_vi.csv')} dịch dở dang, đang tiếp tục...")
        df_translated = pd.read_csv(output_file_path)
    else:
        df_translated = df.copy()

    for col in columns_to_translate:
        if col in df_translated.columns:
            col_vi = f'{col}_vi'
            if col_vi not in df_translated.columns:
                df_translated[col_vi] = None
            df_translated[col_vi] = df_translated[col_vi].astype('object')

    pbar = tqdm(total=len(df_translated), desc=f"Tiến độ {file_name}", unit="dòng")
    
    for index, row in df_translated.iterrows():
        for col in columns_to_translate:
            if col in df_translated.columns:
                if pd.isna(df_translated.at[index, f'{col}_vi']) or str(df_translated.at[index, f'{col}_vi']).strip() == "":
                    original_text = row[col]
                    translated_text = translate_with_ollama(original_text)
                    
                    if translated_text:
                        df_translated.at[index, f'{col}_vi'] = translated_text

        pbar.update(1)

        # Lưu an toàn sau mỗi 10 dòng
        if index > 0 and index % 10 == 0:
            df_translated.to_csv(output_file_path, index=False, encoding='utf-8-sig')

    pbar.close() 
    
    df_translated.to_csv(output_file_path, index=False, encoding='utf-8-sig')
    print(f"-> Hoàn thành {file_name}! Kết quả lưu tại: {output_file_path}")

print("\nĐÃ HOÀN THÀNH TOÀN BỘ CÁC FILE BẰNG OLLAMA!")