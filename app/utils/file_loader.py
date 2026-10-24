import pandas as pd
from typing import List

def load_comments(file_path: str) -> List[str]:
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return [line.strip() for line in lines if line.strip()]

    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)

        if "comment" in df.columns:
            return df["comment"].dropna().astype(str).tolist()
        elif "review" in df.columns:
            return df["review"].dropna().astype(str).tolist()
        else:
            raise ValueError("CSV文件必须包含'comment'列或者'review'列")

    else:
        raise ValueError(f"不支持当前文件格式：{file_path},只能处理csv和txt文件")
