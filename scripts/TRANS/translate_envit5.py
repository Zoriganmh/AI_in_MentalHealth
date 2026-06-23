import os
import time
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# =============================
# CONFIG
# =============================
FOLDER_CHUA_DATA = r"D:\Tai lieu dai hoc\LAB\TramCam\Code\Models\RoBERTa\data"
FILE_NAME = "dataframe_part_3.xlsx" 

OUTPUT_FILE = FILE_NAME.replace(".xlsx", "_vi.csv")
SAVE_INTERVAL = 10 

# Khởi tạo mô hình VietAI (Khoảng 1.1GB)
print("⏳ Đang tải mô hình VietAI/envit5-translation...")
MODEL_NAME = "VietAI/envit5-translation"

# T5 Tokenizer ít bị lỗi hơn MBart, có thể bật use_fast=True
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Ép kiểu torch.bfloat16 giúp tiết kiệm RAM và tăng tốc CPU
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
print("✅ Tải mô hình thành công!")

# =============================
# TRANSLATE FUNCTION (VietAI T5)
# =============================
def translate_with_envit5(text):
    if pd.isna(text) or str(text).strip() == "":
        return text

    text = str(text)
    chunk_size = 1500  

    def _translate_chunk(chunk):
        try:
            # QUAN TRỌNG: Thêm tiền tố "en: " để ra lệnh dịch tiếng Anh -> Việt
            input_text = "en: " + chunk
            
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids
            
            # Tối ưu tốc độ: dùng num_beams=1 (Greedy Search)
            output_ids = model.generate(
                input_ids,
                max_new_tokens=1024,
                num_return_sequences=1,
                num_beams=1 
            )
            
            vi_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            return " ".join(vi_text)
        except Exception as e:
            tqdm.write(f" [!] Lỗi khi dịch: {e}")
            return ""

    if len(text) <= chunk_size:
        result = _translate_chunk(text)
        return result if result else None

    # Chunking nếu văn bản quá dài
    translated_chunks = []
    while len(text) > 0:
        if len(text) <= chunk_size:
            translated_chunks.append(_translate_chunk(text))
            break

        cut_idx = max(
            text.rfind('. ', 0, chunk_size),
            text.rfind('\n', 0, chunk_size),
            text.rfind('? ', 0, chunk_size),
            text.rfind('! ', 0, chunk_size)
        )

        if cut_idx == -1:
            cut_idx = chunk_size
        else:
            cut_idx += 1

        chunk = text[:cut_idx].strip()
        translated_chunks.append(_translate_chunk(chunk))
        text = text[cut_idx:].strip()

    valid_chunks = [c for c in translated_chunks if c]
    if not valid_chunks:
        return None

    return " ".join(valid_chunks)

# =============================
# SAFE SAVE CSV
# =============================
def safe_save_csv(df, path):
    temp_file = path + ".tmp"
    df.to_csv(temp_file, index=False, encoding="utf-8-sig")
    os.replace(temp_file, path)

# =============================
# MAIN
# =============================
input_path = os.path.join(FOLDER_CHUA_DATA, FILE_NAME)
output_path = os.path.join(FOLDER_CHUA_DATA, OUTPUT_FILE)

print(f"\n{'-'*50}\nĐANG XỬ LÝ: {FILE_NAME} (Bằng VietAI EnViT5)\n{'-'*50}")

try:
    df = pd.read_excel(input_path, engine="openpyxl")
except FileNotFoundError:
    print("❌ Không tìm thấy file input")
    exit()

if os.path.exists(output_path):
    print("🔁 Phát hiện file CSV đang dịch dở → resume...")
    try:
        df_translated = pd.read_csv(output_path)
    except Exception as e:
        df_translated = df.copy()
else:
    df_translated = df.copy()

columns_to_translate = ["Title", "Content"]

for col in columns_to_translate:
    if col in df_translated.columns:
        col_vi = f"{col}_vi"
        if col_vi not in df_translated.columns:
            df_translated[col_vi] = None

pbar = tqdm(total=len(df_translated), desc="Tiến độ", unit="dòng")
processed_count = 0

for index, row in df_translated.iterrows():
    translated_flag = False

    for col in columns_to_translate:
        col_vi = f"{col}_vi"

        if col in df_translated.columns:
            if pd.isna(row.get(col_vi)) or str(row.get(col_vi)).strip() == "":
                text = row[col]
                result = translate_with_envit5(text)

                if result:
                    df_translated.at[index, col_vi] = result
                    translated_flag = True

    processed_count += 1
    pbar.update(1)

    if translated_flag and processed_count % SAVE_INTERVAL == 0:
        safe_save_csv(df_translated, output_path)

pbar.close()
safe_save_csv(df_translated, output_path)

print("\n✅ HOÀN THÀNH!")
print(f"📁 File output: {os.path.abspath(output_path)}")