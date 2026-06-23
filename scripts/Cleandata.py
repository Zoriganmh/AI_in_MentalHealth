import re
import pandas as pd
import emoji
from bs4 import BeautifulSoup


def clean_text(text):
    if pd.isna(text) or str(text).strip() == "":
        return ""

    text = str(text)

    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    text = BeautifulSoup(text, "html.parser").get_text()

    text = emoji.replace_emoji(text, replace='')

    text = re.sub(r'(:\)+|=\)+|:\>|:3|:\(|<3|=\(|:\')', '', text)

    text = re.sub(
        r'(\b\d{1,2})[\/\.\-](\d{1,2})[\/\.\-](\d{2,4}\b)',
        lambda m: f"{int(m.group(1))}-{int(m.group(2))}-{int(m.group(3))}",
        text
    )

    text = re.sub(r'(\b\d{1,2})\.(\d{1,2}\b)', r'\1:\2', text)

    allowed = r"[^0-9A-Za-zÀ-ỹ\s\.\,\?\!\:\;\'\"\(\)\[\]\{\}\-–]"
    text = re.sub(allowed, '', text)

    text = re.sub(r'(?<![A-Za-zÀ-ỹ])-(?![A-Za-zÀ-ỹ])', '', text)

    text = re.sub(r'([?!])\1+', r'\1', text)

    text = re.sub(r'^[\?\!\.\,;:]+(?=\w)', '', text)

    text = re.sub(r'([.,;:])\s*', r'\1 ', text)

    text = re.sub(r'\(\s+', '(', text)   
    text = re.sub(r'\s+\)', ')', text)   

    text = re.sub(r'(\d+\))\s*', r'\1 ', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


def process_excel(input_file="dataframe1.xlsx", output_file="dataframe1_clean.xlsx"):
    df = pd.read_excel(input_file)

    required_cols = ["Title", "Content"]
    for col in required_cols:
        if col not in df.columns:
            raise Exception(f"Không tìm thấy cột: {col}")

    df["Title"] = df["Title"].apply(clean_text)
    df["Content"] = df["Content"].apply(clean_text)

    df.to_excel(output_file, index=False)


if __name__ == "__main__":
    process_excel()
