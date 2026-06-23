import pandas as pd

# 1. Đọc file CSV
df = pd.read_csv('formatted_test_vi.csv')

# 2. Tạo từ điển ánh xạ từ tiếng Anh sang tiếng Việt cho class_name
class_name_vi_mapping = {
    'adhd': 'rối loạn tăng động giảm chú ý',
    'anxiety': 'rối loạn lo âu',
    'bipolar': 'rối loạn lưỡng cực',
    'depression': 'trầm cảm',
    'ptsd': 'rối loạn căng thẳng sau sang chấn',
    'none': 'không'
}

# 3. Thay thế các giá trị trong cột class_name
df['class_name'] = df['class_name'].map(class_name_vi_mapping).fillna(df['class_name'])

# 4. Lưu ra file mới
output_file = 'formatted_test_vi_class_vi.csv'
df.to_csv(output_file, index=False)

print(f"Đã lưu file: {output_file}")