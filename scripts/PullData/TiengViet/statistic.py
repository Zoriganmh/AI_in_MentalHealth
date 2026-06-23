import pandas as pd

# ===== Đọc file CSV =====
df2 = pd.read_csv("formatted_15k_dataset.csv")

# Nhóm cần gom (Giả sử em muốn gom các nhãn liên quan đến tâm lý học)
group_list2 = ["depression", "anxiety"] 

df_group2 = df2[df2["class_name"].isin(group_list2)].copy()

# ===== Thống kê nhóm =====
group_stats2 = pd.DataFrame({
    "class_name": ["Group_Total"],
    "So_bai": [df_group2["title"].count()],
    "Bai_bi_xoa": [(df_group2["post"].isin(["[deleted]", "[removed]"])).sum()],
})

# ===== Thống kê từng nhóm nhãn (class_name) =====
class_stats = (
    df2.groupby("class_name")
    .agg(
        So_bai=("title", "count"),
        Bai_bi_xoa=("post", lambda x: (x.isin(["[deleted]", "[removed]"])).sum()),
    )
    .reset_index()
)

# ===== Tính thêm tỷ lệ =====
tong_bai2 = len(df2)

final_stats2 = pd.concat([class_stats, group_stats2], ignore_index=True)
final_stats2["Tỷ lệ bài bị xóa (%)"] = (final_stats2["Bai_bi_xoa"] / final_stats2["So_bai"] * 100).round(2)
final_stats2["Tỷ lệ tổng (%)"] = (final_stats2["So_bai"] / tong_bai2 * 100).round(2)

# ===== Xuất file Excel =====
final_stats2.to_excel("statistics_15k_dataset.xlsx", index=False)
print("🎉 Đã xuất file thống kê cho formatted_15k_dataset.csv!")