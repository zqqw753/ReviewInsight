import json  # 解析模型返回的结构化结果
import operator
import time
from typing import Annotated, TypedDict, List, Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from app.core.config import settings
from app.services.vector_service import search_comments
# 统计方面不再写死，运行时按问题解析
from app.services.stats_service import (
    count_aspect, resolve_aspect, match_aspect, ASPECT_LABELS,
    all_aspect_stats, total_reviews, normalize_text,
)


class AgentState(TypedDict, total=False):
    question: str
    question_type: str
    context: str
    sources: List[str]
    answer: str
    # 产品标识和统计结果随 state 传递，接口直接取用
    product_id: str
    stats: dict
    # 用 add_messages 累加，节点返回的消息不会覆盖历史
    messages: Annotated[list, add_messages]
    # 已执行的工具轮次，用来限制 Agent 循环次数
    tool_rounds: int
    # 是否因轮次上限被强制收尾，接口 meta 会带上
    guarded: bool
    # 节点耗时和 token 用量，接口 meta 会带上
    trace: Annotated[list, operator.add]
    # 最近几轮问题摘要，用来把追问改写成完整问题
    history: List[dict]
    resolved_question: str
    memory_turns: int


llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=settings.deepseek_api_key,
    temperature=0.1,  # 温度调低，输出更稳定
    timeout=settings.llm_timeout_seconds,  # 避免模型请求一直挂起
)


QUESTION_TYPES = ("缺点", "优点", "整体评价")

# 先用规则判，覆盖不到的再交给模型，避免同义问法分类飘。
# 顺序就是优先级：“有什么优缺点”同时含两类词，必须先判整体评价。
OVERALL_KEYWORDS = (
    "整体", "全面", "总体", "综合", "总结", "概括", "优缺点",
    "值得买", "值得入手", "值不值得", "推荐买", "综合评价",
)
DEFECT_KEYWORDS = (
    "缺点", "短板", "缺陷", "问题", "毛病", "吐槽", "差评", "不足",
    "不满意", "不好", "翻车", "踩雷",
)
MERIT_KEYWORDS = (
    "优点", "亮点", "好处", "好评", "满意", "好在哪里", "哪里好", "优势",
)
# “怎么样/如何”这类问法归为整体评价，统计范围看问题有没有点方面
NEUTRAL_MARKERS = ("怎么样", "如何", "表现", "好吗", "怎样", "行不行", "靠谱吗", "情况")


# 带指代或承接词的问题，需要参考上一轮
CONTINUATION_MARKERS = ("那", "那么", "还有", "它", "这款", "这个手机", "再", "换成", "另外", "刚才")


def _rule_classify(question: str) -> Optional[str]:
    """命中规则返回类型，返回 None 表示要走模型。"""
    text = normalize_text(question)
    if any(word in text for word in OVERALL_KEYWORDS):
        return "整体评价"
    if any(word in text for word in DEFECT_KEYWORDS):
        return "缺点"
    if any(word in text for word in MERIT_KEYWORDS):
        return "优点"
    if any(word in text for word in NEUTRAL_MARKERS):
        return "整体评价"
    return None


def _resolve_followup(question: str, history: list) -> Optional[str]:
    """把“那发热呢 / 那优点呢”改写成能独立统计的完整问题。

    返回 None 表示问题已经完整，不用依赖上文。
    只用规则改写，不让模型参与，免得它顺手改掉统计范围。
    """
    if not history:
        return None
    text = normalize_text(question)
    last = history[-1] if isinstance(history[-1], dict) else {}
    last_label = (last.get("aspect_label") or "").strip()
    last_type = last.get("question_type") or "整体评价"
    matched = match_aspect(question)
    rule_type = _rule_classify(question)

    # 问题本身已经是整体意图，就不要继承上文，否则会被缩成单个方面
    if rule_type == "整体评价" and any(word in text for word in OVERALL_KEYWORDS):
        return None

    # “那发热呢 / 换成外观呢”：方面换了，意图沿用上一轮
    if matched and rule_type is None:
        label = ASPECT_LABELS.get(matched, matched)
        if last_type == "整体评价":
            return f"{label}表现怎么样？"
        return f"{label}有什么{last_type}？"

    # “那优点呢 / 它还有什么缺点”：方面沿用上一轮
    is_follow = any(marker in text for marker in CONTINUATION_MARKERS)
    if is_follow and last_label:
        q_type = rule_type or last_type
        if q_type == "整体评价":
            return f"{last_label}表现怎么样？"
        return f"{last_label}有什么{q_type}？"
    return None


def _llm_classify(question: str) -> str:
    """规则覆盖不到的问法交给模型，用几个示例约束输出。"""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一个分类助手。请判断用户的问题属于以下哪种类型：'缺点'、'优点'、'整体评价'。"
         "只输出这三个词中的一个，不要输出任何标点符号或其他内容。\n"
         "示例：\n"
         "问：这手机玩游戏会不会卡？ → 缺点\n"
         "问：包装和保护壳做得用心吗？ → 优点\n"
         "问：这和同价位的比哪个更合适？ → 整体评价"),
        ("user", "{question}")
    ])
    chain = prompt | llm | StrOutputParser()
    q_type = chain.invoke({"question": question}).strip()
    return q_type if q_type in QUESTION_TYPES else "整体评价"


def classify_question(state: AgentState):
    question = state["question"]
    history = state.get("history") or []
    # 先用上文把追问补全，后面统计都按补全后的问题算
    resolved = _resolve_followup(question, history) or question
    q_type = _rule_classify(resolved)
    source = "规则"
    if q_type is None:
        q_type = _llm_classify(resolved)
        source = "模型"
    if resolved != question:
        print(f"--- 节点1：追问已改写为「{resolved}」 ---")
    print(f"--- 节点1：问题类型为 [{q_type}]（{source}） ---")
    # 在这里初始化对话，System Prompt 要用到刚判好的问题类型
    return {
        "question_type": q_type,
        "resolved_question": resolved,
        "memory_turns": len(history),
        "messages": [
            SystemMessage(content=_system_prompt(q_type, resolved)),
            HumanMessage(content=resolved),
        ],
    }


     # 各分支共用同一份统计，不只有缺点分支才拿真实数字
def _aspect_stats(product_id: str, question: str) -> dict:
    aspect = resolve_aspect(question, product_id)
    stats = count_aspect(product_id, aspect)
    return {
        "aspect": aspect,
        "label": ASPECT_LABELS.get(aspect, aspect),
        # 标记方面是问题直接命中的，还是退化成“提及最多的方面”
        "matched": match_aspect(question) is not None,
        "mentions": stats["mentions"],
        "positive": stats["positive"],
        "negative": stats["negative"],
        "negative_ratio": stats["negative_ratio"],
        "total_reviews_in_db": stats["total_reviews_in_db"],
    }


def _overall_stats(product_id: str) -> dict:
    """整体评价按全部方面汇总，不只取提及最多的那一个。

    提及数是各方面之和，一条评论可能提多个方面，所以会大于评论条数。
    """
    raw = all_aspect_stats(product_id)
    breakdown = [
        {
            "aspect": aspect,
            "label": ASPECT_LABELS.get(aspect, aspect),
            "mentions": item["mentions"],
            "positive": item["positive"],
            "negative": item["negative"],
            "negative_ratio": item["negative_ratio"],
        }
        for aspect, item in sorted(
            raw.items(), key=lambda pair: pair[1]["mentions"], reverse=True
        )
    ]
    mentions = sum(item["mentions"] for item in breakdown)
    positive = sum(item["positive"] for item in breakdown)
    negative = sum(item["negative"] for item in breakdown)
    return {
        "aspect": "overall",
        "label": "全部方面",
        "matched": True,
        "mentions": mentions,
        "positive": positive,
        "negative": negative,
        "negative_ratio": round(negative / mentions, 2) if mentions else 0.0,
        "total_reviews_in_db": total_reviews(product_id),
        "breakdown": breakdown,
    }


def _fallback_note(info: dict) -> str:
    """没命中具体方面时写明统计口径，免得读者以为在答那个方面。"""
    if info.get("matched"):
        return ""
    return "问题未匹配到具体方面，以上按该产品提及最多的方面统计。"


def _stats_line(info: dict, question_type: str) -> str:
    """统计结论由代码拼，模型不参与生成数字。"""
    ratio_pct = f"{info['negative_ratio'] * 100:.0f}%"
    tail = f"（该产品共 {info['total_reviews_in_db']} 条评论）。{_fallback_note(info)}"
    # 负面占比的分母是提及该方面的评论数，和总评论数分开写，免得读者算错
    ratio_tail = (
        f"（负面占比按提及该方面的评论数计算；该产品共 "
        f"{info['total_reviews_in_db']} 条评论）。{_fallback_note(info)}"
    )
    if question_type == "优点":
        return (
            f"根据数据库统计，{info['label']}相关提及 {info['mentions']} 次，"
            f"其中正面 {info['positive']} 次、负面 {info['negative']} 次{tail}"
        )
    if question_type == "整体评价" and info.get("breakdown"):
        # 逐方面列出，数字都来自 SQLite 汇总
        lines = [
            f"根据数据库统计（共 {info['total_reviews_in_db']} 条评论，"
            f"各方面提及 {info['mentions']} 次，正面 {info['positive']} 次、"
            f"负面 {info['negative']} 次）："
        ]
        for item in info["breakdown"]:
            item_pct = f"{item['negative_ratio'] * 100:.0f}%"
            lines.append(
                f"- {item['label']}：提及 {item['mentions']} 次，"
                f"正面 {item['positive']} 次、负面 {item['negative']} 次"
                f"（负面占比 {item_pct}）"
            )
        return "\n".join(lines)
    if question_type == "整体评价":
        return (
            f"根据数据库统计，{info['label']}相关提及 {info['mentions']} 次，"
            f"其中正面 {info['positive']} 次、负面 {info['negative']} 次，"
            f"负面占比 {ratio_pct}{ratio_tail}"
        )
    return (
        f"根据数据库统计，{info['label']}相关提及 {info['mentions']} 次，"
        f"其中负面 {info['negative']} 次，负面占比 {ratio_pct}{ratio_tail}"
    )


# 查统计还是查证据、调几次，都交给模型自己决定
def build_tools(product_id: str):
    @tool
    def aspect_statistics(question: str) -> str:
        """查询该产品在问题所指方面的真实统计（提及数、正面数、负面数、负面占比）。"""
        # 空问题直接报错，否则空串会匹配到不相干的方面
        if not (question or "").strip():
            return json.dumps({"error": "question 不能为空"}, ensure_ascii=False)
        info = _aspect_stats(product_id, question)
        return json.dumps(
            {
                **info,
                "defect_line": _stats_line(info, "缺点"),
                "merit_line": _stats_line(info, "优点"),
            },
            ensure_ascii=False,
        )

    @tool
    def overall_statistics() -> str:
        """查询该产品所有方面的汇总统计（逐方面提及/正面/负面/负面占比），用于整体评价。"""
        info = _overall_stats(product_id)
        return json.dumps(
            {**info, "overall_line": _stats_line(info, "整体评价")},
            ensure_ascii=False,
        )

    @tool
    def search_review_evidence(query: str, k: int = 10) -> str:
        """检索该产品下与 query 最相关的评论文本，作为结论的证据（抽样，不是总量）。"""
        # 校验并夹住 k，模型传空查询或离谱的 top-k 时检索不会出错
        if not (query or "").strip():
            return json.dumps({"error": "query 不能为空"}, ensure_ascii=False)
        try:
            size = int(k)
        except (TypeError, ValueError):
            size = settings.DEFAULT_RETRIEVAL_K
        size = max(1, min(size, 20))
        comments, sources = search_comments(query, size, product_id=product_id)
        return json.dumps({"comments": comments, "sources": sources}, ensure_ascii=False)

    return [aspect_statistics, overall_statistics, search_review_evidence]


FORMAT_INSTRUCTIONS = {
    "缺点": '只输出 JSON，不要输出任何解释性文字或 markdown，格式为 '
            '{"defects": [{"name": "缺点名称", "evidence": "原文片段"}]}；'
            '参考评论中确实没有缺点时也要输出 {"defects": []}，不要改用自然语言说明。',
    "优点": '只输出 JSON，不要输出任何解释性文字或 markdown，格式为 '
            '{"merits": [{"name": "优点名称", "evidence": "原文片段"}]}；'
            '参考评论中确实没有优点时也要输出 {"merits": []}，不要改用自然语言说明。',
    "整体评价": '只输出 JSON，格式为 {"summary": "一段话的整体评价", '
              '"merits": [{"name": "优点名称", "evidence": "原文片段"}], '
              '"defects": [{"name": "缺点名称", "evidence": "原文片段"}]}；'
              '没有对应内容时给空数组。',
}


def _system_prompt(question_type: str, question: str = "") -> str:
    # 整体评价有两种情况：问题点了方面（“续航怎么样”）就只看该方面，
    # 没点方面（“整体评价怎么样”）才汇总全部。
    use_overall_tool = question_type == "整体评价" and match_aspect(question) is None
    if use_overall_tool:
        first_step = "1. 先调用 overall_statistics() 拿到该产品所有方面的汇总统计。"
    else:
        first_step = "1. 先调用 aspect_statistics(question) 拿到问题所指方面的真实统计。"
    return f"""你是跨境电商评论分析 Agent，可以调用工具完成分析。

【可用工具】
- aspect_statistics(question)：从 SQLite 取单个方面的真实统计，是唯一可信的数字来源
- overall_statistics()：从 SQLite 取所有方面的汇总统计，用于整体评价
- search_review_evidence(query)：检索评论文本作为证据，可多次调用、可更换 query

【工作方式】
{first_step}
2. 再调用 search_review_evidence 获取证据。
3. 最后给出结论。
4. 调用 aspect_statistics 时，question 参数必须与当前给出的问题完全一致，不要改写或拆成多个方面。

【硬性约束】
- 禁止输出任何统计数字、次数、频次、占比、条数；这些由系统另行拼接。
- 检索到的评论只是抽样证据，禁止据此推断总体情况。
 
【输出格式】
{FORMAT_INSTRUCTIONS.get(question_type, FORMAT_INSTRUCTIONS["整体评价"])}"""


def agent_node(state: AgentState):
    """模型自己决定调哪些工具、调几次。"""
    product_id = state.get("product_id", "default")
    tools = build_tools(product_id)
    model = llm.bind_tools(tools)

    messages = list(state.get("messages") or [])
    if not messages:
        question = state.get("resolved_question") or state["question"]
        messages = [
            SystemMessage(content=_system_prompt(
                state.get("question_type", "整体评价"), question
            )),
            HumanMessage(content=question),
        ]

    response = model.invoke(messages)
    # 记下本次调用的 token 用量
    usage = getattr(response, "usage_metadata", None) or {}

    print("--- 节点2：Agent 决策完成 ---")
    return {
        "messages": [response],
        "trace": [{
            "node": "llm",
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        }],
    }


def tools_node(state: AgentState):
    """执行模型请求的工具，结果包成 ToolMessage 回填。"""
    product_id = state.get("product_id", "default")
    tools_by_name = {item.name: item for item in build_tools(product_id)}
    last = state["messages"][-1]

    outputs = []
    for call in getattr(last, "tool_calls", []) or []:
        name = call.get("name")
        tool_obj = tools_by_name.get(name)
        if tool_obj is None:
            content = json.dumps({"error": f"unknown tool: {name}"}, ensure_ascii=False)
        else:
            try:
                content = tool_obj.invoke(call.get("args") or {})
            except Exception as e:
                content = json.dumps({"error": str(e)}, ensure_ascii=False)
        outputs.append(ToolMessage(content=content, tool_call_id=call.get("id"), name=name))
        print(f"--- 节点3：工具 {name} 执行完成 ---")
    # 记一轮工具执行，should_continue 用它判断要不要停
    return {"messages": outputs, "tool_rounds": state.get("tool_rounds", 0) + 1}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        # 到轮次上限就收尾，防止模型一直调工具
        if state.get("tool_rounds", 0) >= settings.max_tool_rounds:
            return "finalize"
        return "tools"
    return "finalize"


# 模型只输出条目名和原文证据，数字由代码拼
def _items_from(data: dict, key: str) -> List[dict]:
    items = data.get(key, []) if isinstance(data, dict) else []
    result = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        evidence = str(item.get("evidence", "")).strip()
        if name:
            result.append({"name": name, "evidence": evidence})
    return result[:10]


def _parse_json(raw: str) -> dict:
    """模型偶尔会带解释文字或代码围栏，这里只截最外层的 JSON。"""
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _parse_defects(raw: str) -> List[dict]:
    return _items_from(_parse_json(raw), "defects")


def _parse_merits(raw: str) -> List[dict]:
    return _items_from(_parse_json(raw), "merits")


def _parse_overall(raw: str) -> dict:
    """summary 是正文，merits / defects 和单方面评价用同一套结构。"""
    data = _parse_json(raw)
    if not data:
        return {}
    summary = str(data.get("summary", "")).strip()
    merits = _items_from(data, "merits")
    defects = _items_from(data, "defects")
    if not summary and not merits and not defects:
        return {}
    return {"summary": summary, "merits": merits, "defects": defects}


def _render_items(items: List[dict]) -> str:
    return "\n".join(
        f"- {item['name']}（原文：{item['evidence'] or '无'}）" for item in items
    )


def _collect_tool_results(messages: list) -> tuple:
    """从 ToolMessage 里回收整体汇总统计和来源。

    单方面的统计不取工具结果：模型可能改写参数或换着方面多调几次，
    口径只能由代码按用户原问题定，否则数字会和问题对不上。
    """
    overall = None
    sources: List[str] = []
    for msg in reversed(messages or []):
        if not isinstance(msg, ToolMessage):
            continue
        try:
            data = json.loads(msg.content)
        except (json.JSONDecodeError, TypeError):
            continue
        name = getattr(msg, "name", None)
        if name == "overall_statistics" and overall is None and isinstance(data, dict):
            overall = {
                key: data[key]
                for key in (
                    "aspect", "label", "matched", "mentions", "positive",
                    "negative", "negative_ratio", "total_reviews_in_db", "breakdown",
                )
                if key in data
            }
        elif name == "search_review_evidence" and not sources:
            sources = data.get("sources", []) if isinstance(data, dict) else []
    return overall, sources


def finalize(state: AgentState):
    """数字和结构由代码生成，模型只负责正文。"""
    print("--- 节点4：拼接可信答案 ---")
    product_id = state.get("product_id", "default")
    question_type = state.get("question_type", "整体评价")
    # 统计按改写后的问题算
    question = state.get("resolved_question") or state["question"]
    messages = state.get("messages") or []

    overall, sources = _collect_tool_results(messages)
    # 单方面的口径由代码按问题定，不采用模型改过的工具参数
    if question_type == "整体评价" and match_aspect(question) is None:
        stats = overall or _overall_stats(product_id)
    else:
        stats = _aspect_stats(product_id, question)

    stats_line = _stats_line(stats, question_type)

    body_raw = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            body_raw = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    if question_type == "缺点":
        defects = _parse_defects(body_raw)
        body = _render_items(defects) or "（本次未能从抽样评论中提取到结构化的缺点条目）"
        answer = f"{stats_line}\n\n具体缺点：\n{body}"
    elif question_type == "优点":
        merits = _parse_merits(body_raw)
        body = _render_items(merits) or "（本次未能从抽样评论中提取到结构化的优点条目）"
        answer = f"{stats_line}\n\n具体优点：\n{body}"
    else:
        overall_body = _parse_overall(body_raw)
        if overall_body:
            parts = [stats_line]
            if overall_body["summary"]:
                parts.append(f"整体评价：{overall_body['summary']}")
            if overall_body["merits"]:
                parts.append("主要优点：\n" + _render_items(overall_body["merits"]))
            if overall_body["defects"]:
                parts.append("主要问题：\n" + _render_items(overall_body["defects"]))
            answer = "\n\n".join(parts)
        else:
            answer = f"{stats_line}\n\n（本次未能从抽样评论中提取到结构化的整体评价）"

    # 最后一轮还是工具调用，说明是被轮次上限截断的，答案里要写清楚
    last = messages[-1] if messages else None
    guarded = bool(
        getattr(last, "tool_calls", None)
        and state.get("tool_rounds", 0) >= settings.max_tool_rounds
    )
    if guarded:
        answer = f"{answer}\n\n（工具调用已达上限，本次未能生成完整结论，以上数字仍来自数据库统计）"

    print(f"【后台真实统计】: {stats_line}")
    return {
        "answer": answer,
        "stats": stats,
        "sources": sources,
        "guarded": guarded,
        "resolved_question": question,
    }


def _timed(name: str, func):
    """包一层节点，记录耗时并累加到 state.trace。"""
    def wrapped(state):
        started = time.perf_counter()
        result = func(state) or {}
        result = dict(result)
        result["trace"] = list(result.get("trace") or []) + [
            {"node": name, "seconds": round(time.perf_counter() - started, 3)}
        ]
        return result

    return wrapped


def build_agent():
    workflow = StateGraph(AgentState)

    workflow.add_node("classify_question", _timed("classify_question", classify_question))
    workflow.add_node("agent", _timed("agent", agent_node))
    workflow.add_node("tools", _timed("tools", tools_node))
    workflow.add_node("finalize", _timed("finalize", finalize))

    workflow.add_edge(START, "classify_question")
    workflow.add_edge("classify_question", "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        path_map={"tools": "tools", "finalize": "finalize"},
    )
    workflow.add_edge("tools", "agent")
    workflow.add_edge("finalize", END)

    app = workflow.compile()
    return app


agent_app = build_agent()
