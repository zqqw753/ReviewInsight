"""回归评测：直接跑工作流，检查答案里的数字和 SQLite 统计是否一致。

在容器内运行：
    docker exec -w /app reviewinsight-api-1 python scripts/eval.py

覆盖缺点 / 优点 / 整体评价三类问题，输出 JSON 指标：
    stats_consistency_rate  答案 stats 与 SQLite 直查结果一致的比例
    prefix_consistency_rate 答案前缀数字与 stats 一致的比例
    forbidden_rate          答案里出现“频次/约X次”这类自行计数的比例
    structured_rate         成功产出结构化条目（缺点 / 优点 / 整体评价）的比例
    avg_latency_seconds     平均单次工作流耗时
"""
import json
import os
import re
import statistics
import sys
import time

# 直接跑脚本时 sys.path 只有 scripts，要把仓库根目录加进来才能 import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.agent_service import agent_app, _overall_stats
from app.services.stats_service import (
    ASPECT_LABELS, count_aspect, resolve_aspect, match_aspect,
)
from app.services.vector_service import index_counts

# 三类问题都放样本，别只测缺点分支
QUESTIONS = [
    "电池有什么缺点？",
    "续航表现怎么样？",
    "发热问题严重吗？",
    "外观有什么缺点？",
    "屏幕有什么缺点？",
    "有什么缺点？",
    "电池有什么优点？",
    "外观设计有哪些优点？",
    "这个手机值得买吗？",
    "整体评价怎么样？",
    "整体表现如何？",
    "概括一下这款产品。",
]

PRODUCT_ID = "default"
FORBIDDEN_PATTERN = re.compile(r"频次|约\s*\d+\s*次|大概\s*\d*\s*次|大约\s*\d*\s*次")
STRUCTURED_FALLBACKS = {
    "缺点": "未能从抽样评论中提取到结构化的缺点条目",
    "优点": "未能从抽样评论中提取到结构化的优点条目",
    "整体评价": "未能从抽样评论中提取到结构化的整体评价",
}


def expected_stats(question: str, question_type: str) -> dict:
    """整体评价分两种：命中具体方面就看该方面，否则汇总全部。"""
    if question_type == "整体评价" and match_aspect(question) is None:
        return _overall_stats(PRODUCT_ID)
    return count_aspect(PRODUCT_ID, resolve_aspect(question, PRODUCT_ID))


def expected_prefix(info: dict, question_type: str) -> str:
    """按 finalize 的拼接口径生成期望前缀，三类问题都参与该指标。"""
    ratio_pct = f"{info['negative_ratio'] * 100:.0f}%"
    label = ASPECT_LABELS.get(info["aspect"], info["aspect"])
    if question_type == "优点":
        return (
            f"根据数据库统计，{label}相关提及 {info['mentions']} 次，"
            f"其中正面 {info['positive']} 次、负面 {info['negative']} 次"
        )
    if question_type == "整体评价" and info.get("breakdown"):
        return (
            f"根据数据库统计（共 {info['total_reviews_in_db']} 条评论，"
            f"各方面提及 {info['mentions']} 次，正面 {info['positive']} 次、"
            f"负面 {info['negative']} 次）："
        )
    if question_type == "整体评价":
        # 单方面的整体评价：中性问句只覆盖问题所指方面
        return (
            f"根据数据库统计，{label}相关提及 {info['mentions']} 次，"
            f"其中正面 {info['positive']} 次、负面 {info['negative']} 次，"
            f"负面占比 {ratio_pct}"
        )
    return (
        f"根据数据库统计，{label}相关提及 {info['mentions']} 次，"
        f"其中负面 {info['negative']} 次，负面占比 {ratio_pct}"
    )


def structured_applicable(info: dict, question_type: str) -> bool:
    """没有可提取内容时，结构化指标不适用。"""
    if question_type == "优点":
        return info["positive"] > 0
    if question_type == "整体评价":
        return info["mentions"] > 0
    return info["negative"] > 0


def run_one(question: str) -> dict:
    state = {
        "question": question,
        "question_type": "",
        "context": "",
        "sources": [],
        "answer": "",
        "product_id": PRODUCT_ID,
        "stats": {},
    }

    started = time.time()
    final_state = agent_app.invoke(state)
    latency = time.time() - started

    answer = final_state["answer"]
    stats = final_state.get("stats") or {}
    question_type = final_state.get("question_type", "")
    expected = expected_stats(question, question_type)

    stats_consistent = all(
        stats.get(key) == expected[key]
        for key in ("mentions", "positive", "negative", "total_reviews_in_db", "negative_ratio")
    )

    prefix_consistent = answer.startswith(expected_prefix(expected, question_type))

    if structured_applicable(expected, question_type):
        structured = STRUCTURED_FALLBACKS[question_type] not in answer
    else:
        structured = None

    return {
        "question": question,
        "question_type": question_type,
        "aspect": expected["aspect"],
        "stats_consistent": stats_consistent,
        "prefix_consistent": prefix_consistent,
        "forbidden_hit": bool(FORBIDDEN_PATTERN.search(answer)),
        "structured": structured,
        "latency_seconds": round(latency, 2),
    }


def main() -> int:
    index = index_counts()
    results = [run_one(question) for question in QUESTIONS]

    total = len(results)
    stats_ok = sum(item["stats_consistent"] for item in results)
    forbidden = sum(item["forbidden_hit"] for item in results)
    latencies = [item["latency_seconds"] for item in results]

    # 只统计适用的题目，不适用的情况不能算失败
    prefix_items = [item for item in results if item["prefix_consistent"] is not None]
    prefix_ok = sum(1 for item in prefix_items if item["prefix_consistent"])
    prefix_rate = prefix_ok / len(prefix_items) if prefix_items else 1.0

    structured_items = [item for item in results if item["structured"] is not None]
    structured_ok = sum(1 for item in structured_items if item["structured"])
    structured_rate = structured_ok / len(structured_items) if structured_items else 1.0

    summary = {
        "total_questions": total,
        "stats_consistency_rate": stats_ok / total,
        "prefix_consistency_rate": prefix_rate,
        "prefix_sample_size": len(prefix_items),
        "forbidden_rate": forbidden / total,
        "structured_rate": structured_rate,
        "structured_sample_size": len(structured_items),
        "avg_latency_seconds": round(statistics.mean(latencies), 2),
        "index": index,
        "details": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    failed = (
        stats_ok != total
        or prefix_ok != len(prefix_items)
        or structured_ok != len(structured_items)
        or forbidden > 0
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
