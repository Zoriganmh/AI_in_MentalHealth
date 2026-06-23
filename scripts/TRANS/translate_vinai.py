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
SAVE_INTERVAL = 10 # Nên lưu thường xuyên hơn vì chạy local khá tốn thời gian

# Khởi tạo tokenizer và model từ Hugging Face
print("⏳ Đang tải mô hình VinAI (Khoảng 3.5GB, chỉ tải 1 lần duy nhất)...")
MODEL_NAME = "vinai/vinai-translate-en2vi-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang="en_XX", use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
print("✅ Tải mô hình thành công!")

# =============================
# TRANSLATE FUNCTION (VinAI Local)
# =============================
def translate_with_vinai(text):
    if pd.isna(text) or str(text).strip() == "":
        return text

    text = str(text)
    # Giới hạn độ dài do VinAI hỗ trợ tối đa 1024 tokens. 
    # Cắt khoảng 1000 ký tự để chừa biên độ an toàn.
    chunk_size = 1000  

    def _translate_chunk(chunk):
        try:
            # Tokenize văn bản
            input_ids = tokenizer(chunk, return_tensors="pt").input_ids
            
            # Generate bản dịch
            output_ids = model.generate(
                input_ids,
                decoder_start_token_id=tokenizer.lang_code_to_id["vi_VN"],
                num_return_sequences=1,
                num_beams=5, # Beam search giúp kết quả dịch mượt mà hơn
                early_stopping=True
            )
            
            # Giải mã kết quả trả về
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

print(f"\n{'-'*50}\nĐANG XỬ LÝ: {FILE_NAME} (Bằng VinAI Local)\n{'-'*50}")

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
                result = translate_with_vinai(text)

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