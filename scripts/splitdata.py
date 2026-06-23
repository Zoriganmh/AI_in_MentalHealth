import pandas as pd
import numpy as np

# Đường dẫn tới file Excel ban đầu của bạn
input_file = 'dataframe1.xlsx'

# Đọc dữ liệu từ file Excel
print(f"Đang đọc dữ liệu từ {input_file}...")
df = pd.read_excel(input_file)

# Tách dataframe thành 4 phần bằng nhau
# np.array_split sẽ tự động chia đều, nếu số dòng lẻ nó sẽ tự động điều chỉnh cho phù hợp
chunks = np.array_split(df, 4)

# Lưu từng phần ra các file Excel mới
for i, chunk in enumerate(chunks):
    output_file = f'dataframe_1_{i+1}.xlsx'
    # index=False để không lưu thêm cột số thứ tự (index) vào file mới
    chunk.to_excel(output_file, index=False)
    print(f"Đã lưu thành công: {output_file} (gồm {len(chunk)} dòng)")

print("Hoàn tất việc tách file!")