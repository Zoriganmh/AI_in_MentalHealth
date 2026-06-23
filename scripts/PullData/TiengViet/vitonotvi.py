import pandas as pd

# 1. Đọc file val_vi.csv
df_test_vi = pd.read_csv('val_vi.csv')

# 2. Điền các giá trị trống trong cột _vi bằng dữ liệu gốc (tiếng Anh) nếu không có bản dịch
df_test_vi['title_vi'] = df_test_vi['title_vi'].fillna(df_test_vi['title'])
df_test_vi['post_vi'] = df_test_vi['post_vi'].fillna(df_test_vi['post'])
df_test_vi['text_vi'] = df_test_vi['text_vi'].fillna(df_test_vi['text'])

# 3. Thay thế các cột title, post, text bằng nội dung tiếng Việt
df_test_vi['title'] = df_test_vi['title_vi']
df_test_vi['post'] = df_test_vi['post_vi']
df_test_vi['text'] = df_test_vi['text_vi']

# 4. CHỨC NĂNG MỚI: Đổi tên các nhãn trong class_name sang tiếng Việt
class_name_vi_mapping = {
    'adhd': 'rối loạn tăng động giảm chú ý',
    'anxiety': 'rối loạn lo âu',
    'bipolar': 'rối loạn lưỡng cực',
    'depression': 'trầm cảm',
    'ptsd': 'rối loạn căng thẳng sau sang chấn',
    'none': 'không'
}
df_test_vi['class_name'] = df_test_vi['class_name'].map(class_name_vi_mapping).fillna(df_test_vi['class_name'])

# 5. Lọc lại đúng cấu trúc chuẩn của file test.csv gồm 6 cột
columns_to_keep = ['ID', 'title', 'post', 'class_name', 'class_id', 'text']
df_formatted = df_test_vi[columns_to_keep]

# 6. Lưu ra file mới
output_file = 'formatted_val_vi.csv'
df_formatted.to_csv(output_file, index=False)

print("Đã hoàn tất format (bao gồm đổi nhãn class_name sang tiếng Việt) và lưu file:", output_file)