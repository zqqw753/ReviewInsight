# 答案里的数字只能来自 SQLite，模型只负责缺点名和证据
import json

from langchain_core.messages import AIMessage, ToolMessage

import app.services.agent_service as agent_service

FAKE_STATS = {
    "product_id": "default",
    "aspect": "battery",
    "label": "电池",
    "matched": True,
    "mentions": 12,
    "positive": 3,
    "negative": 9,
    "negative_ratio": 0.75,
    "total_reviews_in_db": 30,
}

STATS_KEYS = (
    "aspect", "label", "matched", "mentions", "positive",
    "negative", "negative_ratio", "total_reviews_in_db",
)


def _stats_tool_message(info=None):
    info = dict(info or FAKE_STATS)
    data = {key: info[key] for key in STATS_KEYS}
    return ToolMessage(
        content=json.dumps(data, ensure_ascii=False),
        tool_call_id="call_1",
        name="aspect_statistics",
    )


def _state(messages):
    return {
        "question": "电池有什么缺点？",
        "question_type": "缺点",
        "context": "",
        "sources": [],
        "answer": "",
        "product_id": "default",
        "stats": {},
        "messages": messages,
    }


def _patch_stats(monkeypatch):
    monkeypatch.setattr(agent_service, "resolve_aspect", lambda question, product_id="default": "battery")
    monkeypatch.setattr(agent_service, "ASPECT_LABELS", {"battery": "电池"})
    monkeypatch.setattr(agent_service, "count_aspect", lambda product_id, aspect: dict(FAKE_STATS))
    monkeypatch.setattr(agent_service, "match_aspect", lambda question: "battery")


def test_answer_numbers_come_from_sqlite(monkeypatch):
    _patch_stats(monkeypatch)
    # 模型即使返回错误数字，也不会进入答案
    messages = [
        _stats_tool_message(),
        AIMessage(content='{"defects": [{"name": "续航衰减", "evidence": "半年后续航下降"}]}'),
    ]

    result = agent_service.finalize(_state(messages))
    answer = result["answer"]

    assert answer.startswith("根据数据库统计，电池相关提及 12 次，其中负面 9 次，负面占比 75%")
    assert "具体缺点：" in answer
    assert "续航衰减" in answer
    assert result["stats"]["mentions"] == 12
    assert result["stats"]["negative"] == 9
    assert result["stats"]["total_reviews_in_db"] == 30


def test_stats_survive_model_failure(monkeypatch):
    _patch_stats(monkeypatch)
    messages = [_stats_tool_message(), AIMessage(content="我拒绝输出 JSON")]

    result = agent_service.finalize(_state(messages))
    assert result["answer"].startswith("根据数据库统计，电池相关提及 12 次")
    assert "未能从抽样评论中提取到结构化的缺点条目" in result["answer"]
    assert result["stats"]["mentions"] == 12


def test_unmatched_aspect_is_declared(monkeypatch):
    # 问题没命中方面时，答案里要写明是按提及最多的方面统计
    _patch_stats(monkeypatch)
    monkeypatch.setattr(agent_service, "match_aspect", lambda question: None)
    messages = [AIMessage(content='{"defects": [{"name": "续航衰减", "evidence": "半年后续航下降"}]}')]

    result = agent_service.finalize(_state(messages))
    assert "按该产品提及最多的方面统计" in result["answer"]
    assert result["stats"]["matched"] is False


def test_aspect_statistics_tool_returns_sqlite_stats(monkeypatch):
    # 统计工具直接读 SQLite，返回里带上代码拼好的统计句
    _patch_stats(monkeypatch)
    tools = agent_service.build_tools("default")
    stats_tool = next(item for item in tools if item.name == "aspect_statistics")

    data = json.loads(stats_tool.invoke({"question": "电池有什么缺点？"}))

    assert data["mentions"] == 12
    assert data["negative"] == 9
    assert data["defect_line"].startswith("根据数据库统计，电池相关提及 12 次")


def test_should_continue_routes_by_tool_calls():
    call = AIMessage(
        content="",
        tool_calls=[{"name": "aspect_statistics", "args": {"question": "电池"}, "id": "call_1"}],
    )
    assert agent_service.should_continue({"messages": [call]}) == "tools"
    assert agent_service.should_continue({"messages": [AIMessage(content="完成")]}) == "finalize"


def test_merit_branch_is_structured(monkeypatch):
    # 优点分支也一样，只接受结构化条目，数字仍由代码拼
    _patch_stats(monkeypatch)
    messages = [
        _stats_tool_message(),
        AIMessage(content='{"merits": [{"name": "续航稳定", "evidence": "一天一充够用"}]}'),
    ]

    state = _state(messages)
    state["question_type"] = "优点"
    result = agent_service.finalize(state)

    assert result["answer"].startswith(
        "根据数据库统计，电池相关提及 12 次，其中正面 3 次、负面 9 次"
    )
    assert "具体优点：" in result["answer"]
    assert "- 续航稳定（原文：一天一充够用）" in result["answer"]


def test_merit_branch_falls_back_when_unparsable(monkeypatch):
    _patch_stats(monkeypatch)
    messages = [_stats_tool_message(), AIMessage(content="优点就是续航好")]

    state = _state(messages)
    state["question_type"] = "优点"
    result = agent_service.finalize(state)

    assert "未能从抽样评论中提取到结构化的优点条目" in result["answer"]


def _patch_overall(monkeypatch):
    monkeypatch.setattr(
        agent_service,
        "all_aspect_stats",
        lambda product_id="default": {
            "battery": {"mentions": 10, "positive": 4, "negative": 6, "negative_ratio": 0.6},
            "design": {"mentions": 5, "positive": 5, "negative": 0, "negative_ratio": 0.0},
        },
    )
    monkeypatch.setattr(agent_service, "total_reviews", lambda product_id="default": 30)


def test_overall_covers_all_aspects(monkeypatch):
    # 整体评价汇总全部方面，不只取提及最多的那一个
    _patch_overall(monkeypatch)
    messages = [
        AIMessage(content=(
            '{"summary": "整体不错，续航是短板", '
            '"merits": [{"name": "外观", "evidence": "手感好"}], '
            '"defects": [{"name": "续航", "evidence": "一天两充"}]}'
        ))
    ]

    state = _state(messages)
    state["question"] = "整体评价怎么样？"
    state["question_type"] = "整体评价"
    result = agent_service.finalize(state)
    answer = result["answer"]

    assert answer.startswith("根据数据库统计（共 30 条评论，各方面提及 15 次，正面 9 次、负面 6 次）：")
    assert "- 电池：提及 10 次，正面 4 次、负面 6 次（负面占比 60%）" in answer
    assert "- 外观设计：提及 5 次，正面 5 次、负面 0 次（负面占比 0%）" in answer
    assert "整体评价：整体不错，续航是短板" in answer
    assert "主要优点：" in answer
    assert "主要问题：" in answer
    assert result["stats"]["mentions"] == 15
    assert result["stats"]["total_reviews_in_db"] == 30
    assert [item["aspect"] for item in result["stats"]["breakdown"]] == ["battery", "design"]


def test_overall_prefers_tool_result(monkeypatch):
    # 优先用整体汇总工具的结果，免得重复查库后数字和工具返回对不上
    _patch_overall(monkeypatch)
    breakdown = [
        {"aspect": "battery", "label": "电池", "mentions": 12, "positive": 6,
         "negative": 6, "negative_ratio": 0.5},
        {"aspect": "heating", "label": "发热", "mentions": 8, "positive": 2,
         "negative": 6, "negative_ratio": 0.75},
    ]
    tool_data = {
        "aspect": "overall", "label": "全部方面", "matched": True,
        "mentions": 20, "positive": 8, "negative": 12,
        "negative_ratio": 0.6, "total_reviews_in_db": 30, "breakdown": breakdown,
    }
    messages = [
        ToolMessage(
            content=json.dumps(tool_data, ensure_ascii=False),
            tool_call_id="call_overall",
            name="overall_statistics",
        ),
        AIMessage(content='{"summary": "优缺点都明显", "merits": [], "defects": []}'),
    ]

    state = _state(messages)
    state["question"] = "整体评价怎么样？"
    state["question_type"] = "整体评价"
    result = agent_service.finalize(state)

    assert result["stats"]["mentions"] == 20
    assert result["stats"]["negative"] == 12
    assert "- 发热：提及 8 次" in result["answer"]


def test_overall_with_aspect_scopes_to_single_aspect(monkeypatch):
    # “续航表现怎么样”算整体评价，但只统计问题点到的方面
    _patch_stats(monkeypatch)
    _patch_overall(monkeypatch)
    messages = [AIMessage(content='{"summary": "续航是短板", "merits": [], "defects": []}')]

    state = _state(messages)
    state["question"] = "续航表现怎么样？"
    state["question_type"] = "整体评价"
    result = agent_service.finalize(state)

    assert result["stats"]["aspect"] == "battery"
    assert result["answer"].startswith(
        "根据数据库统计，电池相关提及 12 次，其中正面 3 次、负面 9 次，负面占比 75%"
    )
    assert "各方面提及" not in result["answer"]


def test_rule_classify_covers_common_phrasings():
    cases = {
        "电池有什么缺点？": "缺点",
        "外观设计有哪些优点？": "优点",
        "这个手机值得买吗？": "整体评价",
        "整体表现如何？": "整体评价",
        "续航表现怎么样？": "整体评价",
        "有什么优缺点？": "整体评价",
    }
    for question, expected in cases.items():
        assert agent_service._rule_classify(question) == expected


def test_rule_classify_returns_none_for_unknown():
    assert agent_service._rule_classify("外壳会不会刮花") is None


def test_resolve_followup_rewrites_context_questions():
    # 追问只用规则改写：换方面就沿用上一轮意图，换意图就沿用上一轮方面
    history = [{"question": "电池有什么缺点？", "question_type": "缺点", "aspect_label": "电池"}]

    assert agent_service._resolve_followup("那发热呢", history) == "发热有什么缺点？"
    assert agent_service._resolve_followup("那优点呢", history) == "电池有什么优点？"
    assert agent_service._resolve_followup("它还有什么缺点", history) == "电池有什么缺点？"
    # 自带完整整体意图、或没有历史时，不改写
    assert agent_service._resolve_followup("整体评价怎么样？", history) is None
    assert agent_service._resolve_followup("电池有什么缺点？", []) is None


def test_classify_question_resolves_followup(monkeypatch):
    def _boom(question):
        raise AssertionError("改写后应命中规则，不应调用模型")

    monkeypatch.setattr(agent_service, "_llm_classify", _boom)
    result = agent_service.classify_question({
        "question": "那发热呢",
        "history": [{"question": "电池有什么缺点？", "question_type": "缺点", "aspect_label": "电池"}],
    })

    assert result["question_type"] == "缺点"
    assert result["resolved_question"] == "发热有什么缺点？"
    assert result["memory_turns"] == 1
    assert result["messages"][1].content == "发热有什么缺点？"


def test_finalize_uses_resolved_question(monkeypatch):
    # 统计按改写后的问题算，不用带指代的原始追问
    monkeypatch.setattr(agent_service, "resolve_aspect", lambda q, product_id="default": "battery")
    monkeypatch.setattr(agent_service, "match_aspect", lambda q: "battery" if "电池" in q else None)
    monkeypatch.setattr(agent_service, "ASPECT_LABELS", {"battery": "电池"})
    monkeypatch.setattr(agent_service, "count_aspect", lambda product_id, aspect: dict(FAKE_STATS))

    messages = [
        _stats_tool_message(),
        AIMessage(content='{"merits": [{"name": "续航稳定", "evidence": "一天一充够用"}]}'),
    ]
    state = _state(messages)
    state["question"] = "那优点呢"
    state["question_type"] = "优点"
    state["resolved_question"] = "电池有什么优点？"

    result = agent_service.finalize(state)

    assert result["resolved_question"] == "电池有什么优点？"
    assert result["stats"]["matched"] is True
    assert result["answer"].startswith("根据数据库统计，电池相关提及 12 次")


def test_classify_question_uses_rule_without_llm(monkeypatch):
    def _boom(question):
        raise AssertionError("命中规则时不应调用模型")

    monkeypatch.setattr(agent_service, "_llm_classify", _boom)
    result = agent_service.classify_question({"question": "电池有什么缺点？"})

    assert result["question_type"] == "缺点"
    assert len(result["messages"]) == 2
    assert "aspect_statistics(question)" in result["messages"][0].content


def test_classify_question_uses_llm_for_unknown(monkeypatch):
    monkeypatch.setattr(agent_service, "_llm_classify", lambda question: "优点")
    result = agent_service.classify_question({"question": "外壳会不会刮花"})

    assert result["question_type"] == "优点"


def test_should_continue_stops_at_tool_round_limit(monkeypatch):
    # 到轮次上限后模型还想调工具也得走 finalize
    monkeypatch.setattr(agent_service.settings, "max_tool_rounds", 1)
    call = AIMessage(
        content="",
        tool_calls=[{"name": "aspect_statistics", "args": {"question": "电池"}, "id": "call_2"}],
    )

    assert agent_service.should_continue({"messages": [call], "tool_rounds": 0}) == "tools"
    assert agent_service.should_continue({"messages": [call], "tool_rounds": 1}) == "finalize"


def test_finalize_marks_guarded_when_cut_off(monkeypatch):
    monkeypatch.setattr(agent_service.settings, "max_tool_rounds", 1)
    _patch_stats(monkeypatch)
    call = AIMessage(
        content="",
        tool_calls=[{"name": "search_review_evidence", "args": {"query": "电池"}, "id": "call_3"}],
    )

    state = _state([call])
    state["tool_rounds"] = 1
    result = agent_service.finalize(state)

    assert result["guarded"] is True
    assert "工具调用已达上限" in result["answer"]


def test_search_tool_clamps_k(monkeypatch):
    # top-k 会被夹到 [1, 20]，空查询直接报错，不会走到检索
    captured = {}

    def fake_search(query, k, product_id="default"):
        captured["k"] = k
        return [], []

    monkeypatch.setattr(agent_service, "search_comments", fake_search)
    tools = {item.name: item for item in agent_service.build_tools("p-001")}

    tools["search_review_evidence"].invoke({"query": "电池", "k": 999})
    assert captured["k"] == 20

    empty = tools["search_review_evidence"].invoke({"query": "  ", "k": 3})
    assert json.loads(empty)["error"] == "query 不能为空"


def test_aspect_tool_rejects_empty_question(monkeypatch):
    _patch_stats(monkeypatch)
    tools = {item.name: item for item in agent_service.build_tools("p-001")}

    result = tools["aspect_statistics"].invoke({"question": " "})
    assert json.loads(result)["error"] == "question 不能为空"


def test_timed_records_node_seconds():
    # 节点包装器把耗时写进 trace，接口 meta 和 /metrics 都用
    wrapped = agent_service._timed("demo", lambda state: {"answer": "x"})
    result = wrapped({})

    assert result["trace"][0]["node"] == "demo"
    assert isinstance(result["trace"][0]["seconds"], float)
