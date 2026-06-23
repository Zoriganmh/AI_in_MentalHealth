import pandas as pd
import os
import re

# 1. TÊN FILE ĐẦU VÀO / ĐẦU RA
input_file = r"D:\Tai lieu dai hoc\LAB\TramCam\Code\Models\RoBERTa\data\huingface.xlsx" 
output_file = "15k_full_Filtered_Labeled.xlsx"

# 2. BỘ TỪ KHÓA VÀ LABEL
KEYWORDS_DICT = {
    "Rối loạn tăng động giảm chú ý (ADHD)": [
        "rối loạn tăng động giảm chú ý", "chứng tăng động giảm chú ý", 
        "hội chứng tăng động giảm chú ý", "tăng động giảm chú ý", "tăng động"
    ],
    "Rối loạn lo âu (Anxiety)": [
        "rối loạn lo âu", "lo âu", "chứng lo âu", "sự lo lắng", 
        "nỗi lo âu", "trạng thái bồn chồn", "bất an"
    ],
    "Rối loạn lưỡng cực (Bipolar)": [
        "rối loạn lưỡng cực", "bệnh hưng trầm cảm", 
        "chứng rối loạn lưỡng cực", "bệnh lưỡng cực"
    ],
    "Trầm cảm (Depression)": [
        "bệnh trầm cảm", "chứng trầm cảm", "u uất", "sầu uất", 
        "suy nhược tinh thần", "trầm cảm"
    ],
    "Rối loạn căng thẳng sau sang chấn (PTSD)": [
        "rối loạn căng thẳng sau chấn thương", "rối loạn căng thẳng sau sang chấn",
        "rối loạn stress sau sang chấn", "hội chứng chấn thương tâm lý", 
        "rối loạn tâm lý sau sang chấn", "căng thẳng hậu chấn thương"
    ]
}

def get_labels_and_keywords(text):
    """Trích xuất cả Label và Từ khóa cụ thể tìm thấy trong văn bản"""
    if pd.isna(text) or not isinstance(text, str):
        return pd.Series([None, None])
        
    text_lower = text.lower()
    matched_labels = []
    found_keywords = []
    
    for label, keywords in KEYWORDS_DICT.items():
        label_matched = False
        for kw in keywords:
            if kw.lower() in text_lower:
                found_keywords.append(kw)
                label_matched = True
        
        if label_matched and label not in matched_labels:
            matched_labels.append(label)
            
    label_str = ", ".join(matched_labels) if matched_labels else None
    kw_str = ", ".join(found_keywords) if found_keywords else None
    
    return pd.Series([label_str, kw_str])

def clean_excel_string(s):
    if isinstance(s, str):
        return re.sub(r'[\x00-\x1F]+', '', s)
    return s

print(f"Đang đọc file Excel {input_file}...")
try:
    df = pd.read_excel(input_file)
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file '{input_file}'.")
    exit()
except Exception as e:
    print(f"Lỗi khi đọc file: {e}")
    exit()

print("Đang tiến hành lọc và gán nhãn...")

# 3. CHỈ LẤY 2 CỘT ĐẦU TIÊN ĐỂ TÌM KIẾM
df['Text_For_Search'] = df.iloc[:, 0].astype(str) + " " + df.iloc[:, 1].astype(str)

# Áp dụng hàm gán nhãn
df[['Label', 'Keywords_Found']] = df['Text_For_Search'].apply(get_labels_and_keywords)

# 4. LỌC VÀ LƯU DỮ LIỆU
df_filtered = df[df['Label'].notnull()].copy()
df_filtered = df_filtered.drop(columns=['Text_For_Search'])
df_filtered = df_filtered.map(clean_excel_string)
df_filtered.to_excel(output_file, index=False)

# ---------------------------------------------------------
# 5. IN BÁO CÁO THỐNG KÊ SỐ LƯỢNG
# ---------------------------------------------------------
print(f"\n" + "="*40)
print(f"BÁO CÁO KẾT QUẢ LỌC DỮ LIỆU")
print(f"="*40)
print(f"Tổng số dòng dữ liệu ban đầu: {len(df):,}")
print(f"Số dòng chứa từ khóa giữ lại: {len(df_filtered):,}")

print(f"\n--- CHI TIẾT SỐ LƯỢNG THEO TỪNG NHÃN ---")
# Tách các label được ghép chung (ví dụ "Trầm cảm, Lo âu") thành các dòng riêng lẻ để đếm cho chuẩn
all_individual_labels = df_filtered['Label'].str.split(', ').explode()
label_counts = all_individual_labels.value_counts()

for label, count in label_counts.items():
    # Tính phần trăm so với tổng số lượng bài hợp lệ
    percentage = (count / len(df_filtered)) * 100
    print(f"• {label}: {count:,} ({percentage:.1f}%)")

print(f"\n-> File kết quả đã lưu tại: {os.path.abspath(output_file)}")


"""
Stastics
Số dòng chứa từ khóa giữ lại: 1,698

--- CHI TIẾT SỐ LƯỢNG THEO TỪNG NHÃN ---
• Trầm cảm (Depression): 1,119 (65.9%)
• Rối loạn lo âu (Anxiety): 400 (23.6%)
• Rối loạn lưỡng cực (Bipolar): 151 (8.9%)
• Rối loạn tăng động giảm chú ý (ADHD): 89 (5.2%)
• Rối loạn căng thẳng sau sang chấn (PTSD): 27 (1.6%)

"""