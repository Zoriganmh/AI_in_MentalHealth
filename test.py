import pandas as pd

# Đường dẫn tới file Excel
input_file = r'D:\Tai lieu dai hoc\LAB\TramCam\Code\Models\RoBERTa\data\dataframe1.xlsx'

print(f"Đang đọc dữ liệu từ {input_file}...")

df = pd.read_excel(input_file)

print(f"Số dòng dữ liệu: {len(df)}")