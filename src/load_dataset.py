import os
import json
import pandas as pd

def match_txt_json(text_dir, label_dir):
    rows = []
    for fname in os.listdir(text_dir):
        if not fname.endswith(".txt"):
            continue

        basename = os.path.splitext(fname)[0]
        txt_path = os.path.join(text_dir, fname)
        json_path = os.path.join(label_dir, f"{basename}.json")

        if not os.path.exists(json_path):
            continue  # 라벨 파일이 없으면 skip

        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        label = data.get("classes", ["정상"])[0]  # ['욕설']처럼 나옴
        rows.append({"text": text, "label": label})

    return pd.DataFrame(rows)

if __name__ == "__main__":
    text_folder = r"C:\Users\MSI\Desktop\SpamGuard-MLOps\data\train\01.원천데이터"    # 너의 VSCode 폴더 구조에 맞게 수정
    label_folder = r"C:\Users\MSI\Desktop\SpamGuard-MLOps\data\train\02.라벨링데이터"  # json 파일 있는 경로

    df = match_txt_json(text_folder, label_folder)
    df.to_csv("../data/merged_train.csv", index=False)
    print(df["label"].value_counts())
    print(df.sample(5))
