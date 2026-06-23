import pandas as pd

# ==========================================
# 1. Đọc dữ liệu từ file 15k
# ==========================================
file_path = r"D:\Tai lieu dai hoc\LAB\TramCam\Code\Models\RoBERTa\scripts\PullData\TiengViet\15k_full_Filtered_Labeled.xlsx"
df_15k = pd.read_excel(file_path)

# ==========================================
# 2. Dictionary từ khóa và ánh xạ (Mapping)
# ==========================================
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

LABEL_MAPPING = {
    "Rối loạn tăng động giảm chú ý (ADHD)": {"class_name": "adhd", "class_id": 0},
    "Rối loạn lo âu (Anxiety)": {"class_name": "anxiety", "class_id": 1},
    "Rối loạn lưỡng cực (Bipolar)": {"class_name": "bipolar", "class_id": 2},
    "Trầm cảm (Depression)": {"class_name": "depression", "class_id": 3},
    "Rối loạn căng thẳng sau sang chấn (PTSD)": {"class_name": "ptsd", "class_id": 4}
}

def assign_class(label_text):
    label_text = str(label_text).lower()
    for main_key, class_info in LABEL_MAPPING.items():
        if class_info['class_name'] in label_text:
            return class_info['class_name'], class_info['class_id']
        
        for keyword in KEYWORDS_DICT[main_key]:
            if keyword in label_text:
                return class_info['class_name'], class_info['class_id']
                
    return 'none', 5

# ==========================================
# 3. Tạo 2 cột class_name và class_id mới
# ==========================================
df_15k[['class_name', 'class_id']] = df_15k.apply(
    lambda row: pd.Series(assign_class(row.get('Label', ''))), axis=1
)

# ==========================================
# 4. Gộp cột và lọc lại các cột cần thiết (Lọc bỏ số 0)
# ==========================================
para_cols = [col for col in df_15k.columns if 'paragraph' in str(col).lower()]
fact_cols = [col for col in df_15k.columns if 'fact' in str(col).lower()]

def combine_columns(row, cols):
    values = []
    for col in cols:
        if pd.notna(row[col]):
            val = str(row[col]).strip()
            # Bỏ qua nếu ô trống rỗng, hoặc chứa đúng số '0' / '0.0'
            if val != '' and val != '0' and val != '0.0':
                values.append(val)
    # Nối các câu hợp lệ lại bằng dấu cách (hoặc thay bằng '\n' nếu muốn xuống dòng)
    return ' '.join(values)

# Gộp cột
df_15k['Combined_Paragraph'] = df_15k.apply(lambda row: combine_columns(row, para_cols), axis=1)
df_15k['Combined_Fact'] = df_15k.apply(lambda row: combine_columns(row, fact_cols), axis=1)

# Lọc giữ lại đúng các cột yêu cầu
cols_to_keep = ['ID', 'Topic', 'Question', 'class_name', 'class_id', 'Combined_Paragraph', 'Combined_Fact']
final_cols = [col for col in cols_to_keep if col in df_15k.columns]
df_formatted = df_15k[final_cols].copy()

# Xóa bỏ các ký tự xuống dòng (\n) và dấu tab (\t) ẩn trong tất cả các cột chữ
df_formatted = df_formatted.replace(r'\n|\r|\t', ' ', regex=True)

# ==========================================
# 5. Lưu file như bình thường
# ==========================================
output_filename = 'formatted_15k_dataset.csv'
df_formatted.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"Hoàn thành! Đã xử lý {len(df_formatted)} dòng.")