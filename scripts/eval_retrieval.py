"""在真实手机评论上对比三种检索：纯向量 / 关键词 / 混合（RRF）。

运行：
    docker exec -w /app reviewinsight-api-1 python scripts/eval_retrieval.py

标注说明：ground truth 由“评论有没有包含该主题的同义词”自动生成，属于弱标注。
- 关键词检索与部分标注同源，指标会偏高，只作参照
- 关键结论是“向量对口语化表达是否漏检、混合能否补回”
"""
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.services.stats_service import ASPECT_KEYWORDS, match_aspect, normalize_text
from app.services.vector_service import search_comments

PRODUCT_ID = "shopping_phone"
TOP_K = 5
RRF_SMOOTHING = 60

# 每个查询列出主题同义词用于弱标注，故意混入词表外的词（没电/充一次），
# 免得标注和生产关键词完全同源
QUERIES = [
    {"query": "电池续航表现怎么样", "gold": ("电池", "续航", "待机", "掉电", "耗电", "没电", "充一次")},
    {"query": "充电速度快不快", "gold": ("充电", "快充", "闪充", "充满", "充一次", "充电器")},
    {"query": "手机发热严重吗", "gold": ("发热", "发烫", "烫", "温度", "温控")},
    {"query": "外观做工和手感", "gold": ("外观", "手感", "设计", "质感", "做工", "颜值")},
    {"query": "屏幕显示效果", "gold": ("屏幕", "显示", "画质", "分辨率", "清晰", "色彩")},
    {"query": "拍照效果", "gold": ("拍照", "相机", "像素", "成像", "照片", "镜头")},
    {"query": "系统运行卡不卡", "gold": ("卡顿", "卡死", "流畅", "运行", "速度", "反应")},
    {"query": "物流发货快不快", "gold": ("物流", "发货", "快递", "配送", "到货", "收到")},
    {"query": "外放声音音质", "gold": ("音质", "声音", "喇叭", "外放", "音乐")},
    {"query": "信号和网络稳定吗", "gold": ("信号", "网络", "断流", "掉线", "wifi", "5g")},
]


def load_reviews(product_id: str) -> list:
    conn = sqlite3.connect(settings.db_path)
    try:
        rows = conn.execute(
            "SELECT content FROM reviews WHERE product_id = ?", (product_id,)
        ).fetchall()
    finally:
        conn.close()
    return [normalize_text(row[0]) for row in rows]


def gold_set(query: dict, reviews: list) -> set:
    """弱标注：包含任一主题词的评论视为相关。"""
    terms = tuple(normalize_text(term) for term in query["gold"])
    return {text for text in reviews if any(term in text for term in terms)}


def keyword_ranking(query: str, reviews: list, top_k: int, product_id: str) -> list:
    """关键词检索用生产词表命中（不用评测标注），按命中词数排序。"""
    aspect = match_aspect(query)
    if aspect is None:
        return []
    terms = tuple(normalize_text(term) for term in ASPECT_KEYWORDS[aspect])
    scored = []
    for text in reviews:
        hits = sum(1 for term in terms if term in text)
        if hits:
            scored.append((hits, text))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [text for _, text in scored[:top_k]]


def reciprocal_rank_fusion(rankings: list, top_k: int, smooth: int = RRF_SMOOTHING) -> list:
    """把多路检索结果按倒数排名融合，同一文本只保留一个名次。"""
    scores = {}
    for ranking in rankings:
        for rank, text in enumerate(ranking, start=1):
            scores[text] = scores.get(text, 0.0) + 1.0 / (smooth + rank)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [text for text, _ in ordered[:top_k]]


def evaluate(ranking: list, gold: set) -> dict:
    hit = any(text in gold for text in ranking)
    reciprocal_rank = 0.0
    for rank, text in enumerate(ranking, start=1):
        if text in gold:
            reciprocal_rank = 1.0 / rank
            break
    return {"hit": hit, "reciprocal_rank": round(reciprocal_rank, 3)}


def run_one(query: dict, reviews: list) -> dict:
    gold = gold_set(query, reviews)
    vector_contents, _ = search_comments(query["query"], k=TOP_K, product_id=PRODUCT_ID)
    vector_ranking = [normalize_text(text) for text in vector_contents]

    keyword = keyword_ranking(query["query"], reviews, TOP_K, PRODUCT_ID)
    hybrid = reciprocal_rank_fusion([vector_ranking, keyword], TOP_K)

    return {
        "query": query["query"],
        "gold_size": len(gold),
        "aspect": match_aspect(query["query"]),
        "vector": evaluate(vector_ranking, gold),
        "keyword": evaluate(keyword, gold),
        "hybrid": evaluate(hybrid, gold),
    }


def main():
    reviews = load_reviews(PRODUCT_ID)
    if not reviews:
        sys.exit(f"product_id={PRODUCT_ID} 没有评论，请先运行 scripts/import_dataset.py")

    results = [run_one(query, reviews) for query in QUERIES]

    summary = {}
    for method in ("vector", "keyword", "hybrid"):
        hit_rate = sum(1 for item in results if item[method]["hit"]) / len(results)
        mrr = sum(item[method]["reciprocal_rank"] for item in results) / len(results)
        summary[method] = {
            f"hit_rate@{TOP_K}": round(hit_rate, 3),
            f"mrr@{TOP_K}": round(mrr, 3),
        }

    print(json.dumps({
        "product_id": PRODUCT_ID,
        "reviews": len(reviews),
        "top_k": TOP_K,
        "summary": summary,
        "details": results,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
