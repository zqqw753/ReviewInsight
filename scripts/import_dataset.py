"""把公开中文评论数据集抽样后上传到 ReviewInsight。

支持两种来源：
1. HuggingFace 数据集（默认 dirtycomputer/online_shopping_10_cats，中文电商评论，带类别与正负标签）
2. 本地 CSV / TXT（适用于天池、京东等已下载的商品评论语料）

说明：seamew/ChnSentiCorp 是脚本型数据集，datasets 4.x 已不支持，因此默认改用
online_shopping_10_cats；如需其他数据集，用 --hf-name 指定 parquet 格式的仓库即可。

示例：
    # 容器内（要关掉 HF 离线标记，镜像默认 HF_HUB_OFFLINE=1）
    docker exec -e HF_HUB_OFFLINE=0 -w /app reviewinsight-api-1 \
        python scripts/import_dataset.py --category 手机 --limit 200 --product-id shopping_phone

    # 本地已下载的天池/京东语料
    docker cp data/raw/jd.csv reviewinsight-api-1:/tmp/jd.csv
    docker exec -w /app reviewinsight-api-1 \
        python scripts/import_dataset.py --source csv --csv /tmp/jd.csv --product-id jd_phone
"""
import argparse
import csv
import json
import os
import random
import sys
import urllib.request
import uuid

COLUMN_CANDIDATES = ("comment", "review", "text", "content", "评价", "评论")


def _pick_column(fieldnames):
    for name in COLUMN_CANDIDATES:
        if name in fieldnames:
            return name
    return fieldnames[0] if fieldnames else None


def load_local(path, limit):
    """读取本地 CSV / TXT，统一成 [{"comment": ..., "label": ...}]。"""
    rows = []
    if path.lower().endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    rows.append({"comment": text, "label": ""})
    else:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            column = _pick_column(reader.fieldnames or [])
            if not column:
                raise ValueError(f"{path} 没有可用的评论文本列，期望之一：{COLUMN_CANDIDATES}")
            for item in reader:
                text = (item.get(column) or "").strip()
                if text:
                    rows.append({"comment": text, "label": item.get("label", "")})
    return rows[:limit] if limit else rows


def load_hf(name, limit):
    """从 HuggingFace 数据集读取（需要 pip install datasets）。"""
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("缺少依赖：请先执行 pip install datasets（HF 数据集模式必须）。")

    try:
        ds = load_dataset(name)
    except ValueError as e:
        if "trust_remote_code" not in str(e):
            raise
        ds = load_dataset(name, trust_remote_code=True)

    rows = []
    # DatasetDict：合并所有 split；Dataset：直接遍历
    splits = ds.values() if isinstance(ds, dict) else [ds]
    for split in splits:
        columns = list(getattr(split, "column_names", []) or [])
        text_column = next((c for c in COLUMN_CANDIDATES if c in columns), None)
        if not text_column:
            continue
        label_column = "label" if "label" in columns else None
        for item in split:
            text = (item.get(text_column) or "").strip()
            if text:
                rows.append({
                    "comment": text,
                    "label": item.get(label_column, "") if label_column else "",
                    "category": item.get("cat", ""),
                })
    return rows[:limit] if limit else rows


def sample_rows(rows, limit, seed, min_len):
    """过滤过短文本、去重后随机抽样，保证结果可复现。"""
    cleaned, seen = [], set()
    for item in rows:
        text = item["comment"]
        if len(text) < min_len or text in seen:
            continue
        seen.add(text)
        cleaned.append(item)

    rng = random.Random(seed)
    rng.shuffle(cleaned)
    return cleaned[:limit]


def save_csv(rows, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["comment", "label", "category"], extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def post_file(api_base, csv_path, product_id):
    """用 multipart/form-data 上传，避免额外依赖 requests。"""
    boundary = uuid.uuid4().hex
    with open(csv_path, "rb") as f:
        file_bytes = f.read()

    body = bytearray()
    body += (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="product_id"\r\n\r\n'
        f"{product_id}\r\n"
    ).encode("utf-8")
    body += (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{os.path.basename(csv_path)}"\r\n'
        f"Content-Type: text/csv\r\n\r\n"
    ).encode("utf-8")
    body += file_bytes
    body += f"\r\n--{boundary}--\r\n".encode("utf-8")

    request = urllib.request.Request(
        api_base.rstrip("/") + "/api/v1/analysis/upload",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="抽样公开中文评论数据并导入 ReviewInsight")
    parser.add_argument("--source", choices=("hf", "csv"), default="hf")
    parser.add_argument("--csv", help="--source csv 时的本地文件路径（.csv/.txt）")
    parser.add_argument(
        "--hf-name", default="dirtycomputer/online_shopping_10_cats",
        help="HuggingFace 数据集名（需为 parquet 格式）",
    )
    parser.add_argument("--category", help="按类别过滤，如 手机 / 计算机 / 衣服")
    parser.add_argument("--limit", type=int, default=200, help="抽样条数")
    parser.add_argument("--min-len", type=int, default=8, help="最短保留长度")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--product-id", default="shopping10")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--out", help="抽样结果 CSV 路径")
    parser.add_argument("--no-upload", action="store_true", help="只生成 CSV，不上传")
    args = parser.parse_args()

    if args.source == "csv":
        if not args.csv:
            sys.exit("--source csv 必须提供 --csv")
        rows = load_local(args.csv, None)
    else:
        rows = load_hf(args.hf_name, None)

    categories = sorted({str(item.get("category", "")) for item in rows if item.get("category")})
    if args.category:
        rows = [item for item in rows if str(item.get("category", "")) == args.category]
        if not rows:
            sys.exit(f"类别 {args.category} 没有数据，可用类别：{categories}")

    sampled = sample_rows(rows, args.limit, args.seed, args.min_len)
    if not sampled:
        sys.exit("没有可用样本，请检查数据源、--category 或 --min-len")

    tag = args.category or args.source
    out_path = args.out or f"data/real/{tag}_{len(sampled)}.csv"
    save_csv(sampled, out_path)

    print(json.dumps({
        "loaded": len(rows),
        "sampled": len(sampled),
        "categories": categories,
        "product_id": args.product_id,
        "csv": out_path,
    }, ensure_ascii=False))

    if args.no_upload:
        return

    result = post_file(args.api, out_path, args.product_id)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
