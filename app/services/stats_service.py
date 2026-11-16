import json
import re
import sqlite3
import unicodedata
from typing import Optional
from app.core.config import settings

DB_PATH = settings.db_path

def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def count_aspect(product_id: str, aspect: str) -> dict:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT aspects FROM reviews WHERE product_id = ?", (product_id, ))
    rows = cursor.fetchall()
    conn.close()

    pos = 0
    neg = 0
    mentions = 0
    total_reviews = len(rows)

    for row in rows:
        aspects_data = json.loads(row["aspects"] or "{}")
        if aspect not in aspects_data:
            continue

        mentions += 1
        if aspects_data[aspect] > 0:
            pos += 1
        elif aspects_data[aspect] < 0:
            neg += 1


    return {
        "product_id": product_id,
        "aspect": aspect,
        "mentions": mentions,
        "positive": pos,
        "negative": neg,
        # 负面占比按提及该方面的评论数算，不用总评论数
        "negative_ratio": round(neg / mentions, 2) if mentions else 0.0,
        # 总评论数只用来展示样本规模，不参与占比
        "total_reviews_in_db": total_reviews
    }


def total_reviews(product_id: str = "default") -> int:
    """该产品评论总数，整体评价时用来展示样本规模。"""
    conn = _get_conn()
    row = conn.execute(
        "SELECT COUNT(*) AS total FROM reviews WHERE product_id = ?", (product_id,)
    ).fetchone()
    conn.close()
    return row["total"]


# 词表同时给两个地方用：问题匹配方面、评论抽取方面。
# 手机评论里口语说法很多（待机、掉电、快充、毫安），不收进来就会漏。
# 繁体变体用于混排文本，mah 兼顾规格描述。
ASPECT_KEYWORDS = {
    "battery": (
        "电池", "续航", "充电", "电量", "待机", "掉电", "耗电", "用电",
        "快充", "闪充", "无线充", "充满", "毫安", "mah", "蓄电",
        "電池", "續航", "充電", "電量", "待機", "掉電", "耗電",
    ),
    "heating": ("发热", "发烫", "烫", "温度", "温控", "發熱", "發燙"),
    "design": ("外观", "手感", "设计", "质感", "做工", "颜值", "外觀", "設計"),
}

# 负面词只在命中方面词的同一句里算数，避免整段评论串味。
NEGATIVE_WORDS = (
    "差", "不行", "不好", "慢", "虚标", "短", "不耐用", "垃圾", "故障",
    "坏", "衰减", "不足", "不够", "失望", "退货", "换货", "维修", "后悔",
    "卡顿", "卡死", "断", "翻车", "掉", "耗",
)

# 否定词：“不/没/无/未”紧挨着就判定；“不会/不能/没有”允许中间隔几个字
#（如“不会突然断电”）。不收录“别”，否则“特别慢”会被当成否定。
NEGATION_PREFIXES = ("不", "没", "无", "未")
LONG_NEGATIONS = ("不会", "不能", "没有", "毫无", "并无", "未曾")


def _is_negated(text: str, index: int) -> bool:
    """判断 index 位置的词前面有没有否定词。"""
    prefix = text[max(0, index - 2):index]
    if prefix.endswith(NEGATION_PREFIXES):
        return True
    window = text[max(0, index - 4):index]
    return any(token in window for token in LONG_NEGATIONS)


def normalize_text(text: str) -> str:
    """统一大小写和全角半角，顺手去掉汉字之间的空格。"""
    normalized = unicodedata.normalize("NFKC", text or "").lower()
    return re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", normalized)


def _has_unnegated(text: str, word: str) -> bool:
    """词出现过且前面没有否定词时返回 True。"""
    start = 0
    while True:
        index = text.find(word, start)
        if index == -1:
            return False
        if not _is_negated(text, index):
            return True
        start = index + len(word)


# 有些方面词本身就说负面体验（“烫”“发热”），命中就算负面
NEGATIVE_ASPECT_HINTS = {
    "heating": ("发热", "发烫", "烫", "發熱", "發燙"),
}


def _segment_polarity(segment: str, aspect: str, keywords: tuple) -> Optional[int]:
    """单句判定，被否定的方面词不算提及。"""
    hit = False
    negative = False
    hints = NEGATIVE_ASPECT_HINTS.get(aspect, ())

    for keyword in keywords:
        start = 0
        while True:
            index = segment.find(keyword, start)
            if index == -1:
                break
            # “不烫”“不会突然断电”都不算提及
            if _is_negated(segment, index):
                start = index + len(keyword)
                continue
            hit = True
            if keyword in hints:
                negative = True
            if any(_has_unnegated(segment, word) for word in NEGATIVE_WORDS):
                negative = True
            start = index + len(keyword)

    if not hit:
        return None
    return -1 if negative else 1


def _polarity(segments: list, aspect: str, keywords: tuple) -> Optional[int]:
    """按句判极性，任一句负面就整体判负，长评论不会互相串味。"""
    hit = False
    negative = False
    for segment in segments:
        polarity = _segment_polarity(segment, aspect, keywords)
        if polarity is None:
            continue
        hit = True
        negative = negative or polarity < 0
    if not hit:
        return None
    return -1 if negative else 1


def extract_aspects(content: str) -> dict:
    """从评论里抽取方面和极性，入库时写进 SQLite。

    先归一化文本再按句切分，否定表达不会被当成负面。
    """
    text = normalize_text(content)
    segments = [item for item in re.split(r"[。！？!?；;\n]", text) if item]
    aspects = {}
    for aspect, keywords in ASPECT_KEYWORDS.items():
        polarity = _polarity(segments, aspect, keywords)
        if polarity is not None:
            aspects[aspect] = polarity
    return aspects

# 方面名称，注入 Prompt 时用
ASPECT_LABELS = {
    "battery": "电池",
    "heating": "发热",
    "design": "外观设计",
}


def all_aspect_stats(product_id: str = "default") -> dict:
    """遍历一次拿到该产品所有方面的提及和正负数量。"""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT aspects FROM reviews WHERE product_id = ?", (product_id,))
    rows = cursor.fetchall()
    conn.close()

    stats: dict = {}
    for row in rows:
        data = json.loads(row["aspects"] or "{}")
        for aspect, polarity in data.items():
            item = stats.setdefault(aspect, {"mentions": 0, "positive": 0, "negative": 0})
            item["mentions"] += 1
            if polarity > 0:
                item["positive"] += 1
            elif polarity < 0:
                item["negative"] += 1

    for item in stats.values():
        item["negative_ratio"] = (
            round(item["negative"] / item["mentions"], 2) if item["mentions"] else 0.0
        )
    return stats


def match_aspect(question: str) -> Optional[str]:
    """问题命中具体方面时返回方面名，返回 None 表示要退化到提及最多的方面。"""
    text = normalize_text(question)
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return aspect
    return None


def resolve_aspect(question: str, product_id: str = "default") -> str:
    """先按关键词匹配方面，匹配不到就用提及最多的方面。"""
    matched = match_aspect(question)
    if matched:
        return matched

    stats = all_aspect_stats(product_id)
    if stats:
        return max(stats, key=lambda item: stats[item]["mentions"])
    return "battery"


def data_version(product_id: str = "default") -> str:
    """数据版本是条数加最新入库时间，数据一变缓存就失效。"""
    conn = _get_conn()
    row = conn.execute(
        "SELECT COUNT(*) AS total, COALESCE(MAX(created_at), '') AS latest "
        "FROM reviews WHERE product_id = ?",
        (product_id,),
    ).fetchone()
    conn.close()
    return f"{row['total']}:{row['latest']}"