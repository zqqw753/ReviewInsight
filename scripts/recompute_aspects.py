"""用新的方面抽取规则重算已有评论的 aspects。

默认只做对比（dry-run），加 --apply 才写回 SQLite。向量索引不受影响
（aspects 只存在 SQLite），因此不需要重建 Chroma。

用法：
    docker exec -w /app reviewinsight-api-1 python scripts/recompute_aspects.py
    docker exec -w /app reviewinsight-api-1 python scripts/recompute_aspects.py --apply
"""
import argparse
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.services.stats_service import extract_aspects  # noqa: E402


def _count(aspects_by_row, key):
    stats = {}
    for aspects in aspects_by_row:
        for aspect, polarity in aspects.items():
            item = stats.setdefault(aspect, {"mentions": 0, "positive": 0, "negative": 0})
            item["mentions"] += 1
            if polarity > 0:
                item["positive"] += 1
            else:
                item["negative"] += 1
    return stats.get(key, {"mentions": 0, "positive": 0, "negative": 0})


def main():
    parser = argparse.ArgumentParser(description="按新的关键词规则重算评论方面")
    parser.add_argument("--product-id", help="只处理指定产品")
    parser.add_argument("--apply", action="store_true", help="写回 SQLite（默认 dry-run）")
    parser.add_argument("--samples", type=int, default=3, help="展示多少条变化示例")
    args = parser.parse_args()

    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    sql = "SELECT review_id, product_id, content, aspects FROM reviews"
    params = ()
    if args.product_id:
        sql += " WHERE product_id = ?"
        params = (args.product_id,)
    rows = conn.execute(sql, params).fetchall()

    old_all, new_all, changes = [], [], []
    for row in rows:
        old = json.loads(row["aspects"] or "{}")
        new = extract_aspects(row["content"])
        old_all.append(old)
        new_all.append(new)
        if old != new:
            changes.append((row, old, new))

    print(f"rows={len(rows)} changed={len(changes)} mode={'apply' if args.apply else 'dry-run'}")
    for aspect in ("battery", "heating", "design"):
        before = _count(old_all, aspect)
        after = _count(new_all, aspect)
        print(f"{aspect}: mentions {before['mentions']} -> {after['mentions']}, "
              f"negative {before['negative']} -> {after['negative']}")

    for row, old, new in changes[:max(args.samples, 0)]:
        content = row["content"][:60].replace("\n", " ")
        print(json.dumps({
            "product_id": row["product_id"],
            "content": content,
            "old": old,
            "new": new,
        }, ensure_ascii=False))

    if args.apply and changes:
        conn.executemany(
            "UPDATE reviews SET aspects = ? WHERE review_id = ?",
            [(json.dumps(new, ensure_ascii=False), row["review_id"]) for row, _, new in changes],
        )
        conn.commit()
        print(f"updated={len(changes)}")
    conn.close()

    if args.apply:
        print("提示：统计口径已变化，请清理问答缓存（redis-cli FLUSHDB），否则可能返回旧答案。")


if __name__ == "__main__":
    main()
