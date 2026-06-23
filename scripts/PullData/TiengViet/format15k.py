import pandas as pd

# 1. Đọc dữ liệu từ file 15k (Sử dụng read_excel thay vì read_csv)
# Lưu ý: Sửa lại tên file dưới đây cho khớp chính xác với tên file trên máy bạn
file_path = '15k_full_Filtered_Labeled.xlsx' 
df_15k = pd.read_excel(file_path)

# 2. Dictionary từ khóa của bạn
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

# Ánh xạ từ Key sang class_name (tiếng Anh) và class_id theo chuẩn file test
LABEL_MAPPING = {
    "Rối loạn tăng động giảm chú ý (ADHD)": {"class_name": "adhd", "class_id": 0},
    "Rối loạn lo âu (Anxiety)": {"class_name": "anxiety", "class_id": 1},
    "Rối loạn lưỡng cực (Bipolar)": {"class_name": "bipolar", "class_id": 2},
    "Trầm cảm (Depression)": {"class_name": "depression", "class_id": 3},
    "Rối loạn căng thẳng sau sang chấn (PTSD)": {"class_name": "ptsd", "class_id": 4}
}

# Hàm phân loại dựa trên cột Label (hoặc Keywords_Found)
def assign_class(label_text):
    label_text = str(label_text).lower()
    
    # Duyệt qua các bệnh trong mapping
    for main_key, class_info in LABEL_MAPPING.items():
        if class_info['class_name'] in label_text:
            return class_info['class_name'], class_info['class_id']
        
        for keyword in KEYWORDS_DICT[main_key]:
            if keyword in label_text:
                return class_info['class_name'], class_info['class_id']
                
    return 'none', 5

# 3. Tạo 2 cột class_name và class_id mới
df_15k[['class_name', 'class_id']] = df_15k.apply(
    lambda row: pd.Series(assign_class(row['Label'])), axis=1
)

# 4. Trích xuất và định dạng lại cấu trúc 6 cột chuẩn
df_formatted = pd.DataFrame()
df_formatted['ID'] = df_15k['ID']
df_formatted['title'] = df_15k['Topic'].fillna('')
df_formatted['post'] = df_15k['Question'].fillna('')
df_formatted['class_name'] = df_15k['class_name']
df_formatted['class_id'] = df_15k['class_id']
df_formatted['text'] = df_formatted['title'] + ": " + df_formatted['post']

# 5. Lưu ra file CSV mới
output_filename = 'formatted_15k_dataset.csv'
df_formatted.to_csv(output_filename, index=False)

print(f"Hoàn thành! Đã xử lý {len(df_formatted)} dòng.")
print(f"Đã lưu file chuẩn tại: {output_filename}")