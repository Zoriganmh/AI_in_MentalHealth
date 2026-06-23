# AIzaSyC2qo2nimZLAcqpeB9TUHpl2t-PK4_PBWY
# AIzaSyDmCO2DjnX3gHPpr9bLgmCiRaiba0ivuJk
import os
import time
import pandas as pd
from google import genai
from tqdm import tqdm

# =============================
# CONFIG
# =============================
FOLDER_CHUA_DATA = r"D:\Tai lieu dai hoc\LAB\TramCam\Code\Models\RoBERTa\data"
FILE_NAME = "dataframe_part_2.xlsx"  

OUTPUT_FILE = FILE_NAME.replace(".xlsx", "_vi.csv")
SAVE_INTERVAL = 10  

# Điền API Key của bạn
GOOGLE_API_KEY = "AIzaSyDmCO2DjnX3gHPpr9bLgmCiRaiba0ivuJk" 
client = genai.Client(api_key=GOOGLE_API_KEY)

# =============================
# TRANSLATE FUNCTION (GEMINI)
# =============================
def translate_with_gemini(text, max_retries=5):
    if pd.isna(text) or str(text).strip() == "":
        return text

    prompt = f"""
    Bạn là một dịch giả chuyên nghiệp. Hãy dịch đoạn văn bản tiếng Anh sau sang tiếng Việt.
    Yêu cầu BẮT BUỘC:
    1. Dịch chuẩn xác ngữ nghĩa, không thêm bớt thông tin.
    2. Dùng cấu trúc câu tự nhiên, trôi chảy của người Việt.
    3. Giữ nguyên văn phong chia sẻ thực tế trên các diễn đàn (forum/social media).
    4. CHỈ TRẢ VỀ NỘI DUNG ĐÃ DỊCH. TUYỆT ĐỐI KHÔNG thêm bất kỳ từ ngữ nào khác, KHÔNG giải thích, KHÔNG có câu chào như "Dưới đây là bản dịch...".

    Văn bản cần dịch:
    "{text}"
    """

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            result = response.text.strip()
            # Xóa dấu ngoặc kép nếu model tự động thêm vào
            if result.startswith('"') and result.endswith('"'):
                result = result[1:-1]
            return result
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                wait_time = 20
                tqdm.write(f" [!] Quá tải API. Đang nghỉ {wait_time}s rồi thử lại (Lần {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                tqdm.write(f" [!] Lỗi: {e}. Đang thử lại (Lần {attempt + 1}/{max_retries})...")
                time.sleep(6)

    tqdm.write(" [!] Đã thử lại nhiều lần nhưng thất bại, bỏ qua dòng này.")
    return None

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

print(f"\n{'-'*50}\nĐANG XỬ LÝ: {FILE_NAME}\n{'-'*50}")

# Đọc file gốc
try:
    df = pd.read_excel(input_path, engine="openpyxl")
except FileNotFoundError:
    print("❌ Không tìm thấy file input")
    exit()

# Resume nếu có CSV đang dịch dở
if os.path.exists(output_path):
    print("🔁 Phát hiện file CSV đang dịch dở → resume...")
    try:
        df_translated = pd.read_csv(output_path)
    except Exception as e:
        print(f"⚠️ CSV lỗi ({e}) → tạo lại")
        df_translated = df.copy()
else:
    df_translated = df.copy()

# Cột cần dịch 
columns_to_translate = ["Title", "Content"]

# Tạo cột _vi nếu chưa có
for col in columns_to_translate:
    if col in df_translated.columns:
        col_vi = f"{col}_vi"
        if col_vi not in df_translated.columns:
            df_translated[col_vi] = None

pbar = tqdm(total=len(df_translated), desc="Tiến độ", unit="dòng")
processed_count = 0

# =============================
# LOOP
# =============================
for index, row in df_translated.iterrows():
    translated_flag = False

    for col in columns_to_translate:
        col_vi = f"{col}_vi"

        if col in df_translated.columns:
            if pd.isna(row.get(col_vi)) or str(row.get(col_vi)).strip() == "":
                text = row[col]
                result = translate_with_gemini(text)

                if result:
                    df_translated.at[index, col_vi] = result
                    translated_flag = True

                # Tránh rate limit của Gemini API
                time.sleep(3)

    processed_count += 1
    pbar.update(1)

    # Lưu định kỳ
    if translated_flag and processed_count % SAVE_INTERVAL == 0:
        safe_save_csv(df_translated, output_path)

pbar.close()

# Lưu lần cuối
safe_save_csv(df_translated, output_path)

print("\n✅ HOÀN THÀNH!")
print(f"📁 File output: {os.path.abspath(output_path)}")