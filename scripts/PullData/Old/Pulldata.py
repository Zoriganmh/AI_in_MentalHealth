import requests
import pandas as pd
import datetime
import os
import time

subreddits = ["Entrepreneur", "law", "sciencefiction", "technology", "mathematics", "music", "travel"]

def safe_get(url, params, retries=5, timeout=15):
    for i in range(retries):
        try:
            res = requests.get(url, params=params, timeout=timeout)
            if res.status_code == 200:
                return res
            else:
                print(f"Lỗi {res.status_code}, thử lại ({i+1}/{retries}) ...")
        except requests.exceptions.RequestException as e:
            print(f"Lỗi kết nối ({i+1}/{retries}): {e}")
        time.sleep(2 + i * 2)
    return None


# Time
start_date = datetime.datetime(2016, 1, 1)
end_date = datetime.datetime(2017, 12, 31)
step = datetime.timedelta(days=3)

url = "https://api.pullpush.io/reddit/search/submission/"

#Lấy dữ liệu
all_posts = []
interval_count = 0

current = start_date
while current < end_date:
    after = int(current.timestamp())
    before = int((current + step).timestamp())
    interval_label = f"{current.strftime('%Y-%m-%d')} → {(current + step).strftime('%Y-%m-%d')}"
    interval_count += 1

    print(f"\n Khoảng {interval_label}")

    for sub in subreddits:

        params = {
            "subreddit": sub,
            "after": after,
            "before": before,
            "size": 100
        }

        res = safe_get(url, params)
        if res:
            data = res.json().get("data", [])
            print(f" Lấy được {len(data)} bài từ r/{sub}")
            for post in data:
                all_posts.append({
                    "Author": post.get("author"),
                    "Posted at": datetime.datetime.fromtimestamp(int(post["created_utc"])).strftime("%Y-%m-%d %H:%M:%S"),
                    "Title": post.get("title"),
                    "Content": post.get("selftext", ""),
                    "Subreddit(Label of category)": sub,
                    "Date Range": interval_label
                })
        else:
            print(f" Không thể lấy dữ liệu r/{sub} trong khoảng {interval_label}")

    current += step


os.makedirs("data", exist_ok=True)
df = pd.DataFrame(all_posts)
output_file = os.path.join("data", "dataTiengViet.xlsx")
import re

def clean_excel_string(s):
    if isinstance(s, str):
        return re.sub(r'[\x00-\x1F]+', '', s)
    return s

df = df.map(clean_excel_string)
df.to_excel(output_file, index=False)

print(f"\n end")
print(f" Tổng số bài: {len(all_posts)}")
print(f" {os.path.abspath(output_file)}")