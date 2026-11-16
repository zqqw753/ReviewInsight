import os
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.core import metrics
from app.core.config import settings
from app.services.vector_service import add_comments_to_store, search_comments, index_counts
from app.utils.file_loader import load_comments
from app.services.cache_service import get_cached_answer, set_cached_answer
from app.services.agent_service import agent_app
# 会话历史存在 Redis，追问改写和写回都用这里
from app.services.memory_service import load_history, append_turn
# 数据版本参与缓存 key，上传新数据后旧答案自动失效
from app.services.stats_service import data_version

# 指纹日志路径走配置，容器里挂到 /data，重建后同一文件不会重复入库
PROCESSED_FILES_LOG = settings.processed_files_log

class AnalyzeRequest(BaseModel):
    question: str
    # 按产品查询，统计和检索都用这个 product_id
    product_id: str = "default"
    # 可选会话标识，不传就是普通单轮问答
    session_id: Optional[str] = None

class AnalyzeResponse(BaseModel):
    answer: str
    sources: List[str]
    # 随答案返回本次统计，前端展示和测试都用它
    stats: Optional[dict] = None
    # 节点耗时、token 用量、工具轮次和是否被截断
    meta: Optional[dict] = None

router = APIRouter(prefix="/api/v1/analysis", tags=["选品分析"])


def _build_meta(payload: dict, latency_ms: float, *, cached: bool = False) -> dict:
    """汇总节点耗时、token 和轮次状态，随接口返回。"""
    trace = [item for item in (payload.get("trace") or []) if isinstance(item, dict)]
    return {
        "cached": cached,
        "latency_ms": round(latency_ms, 1),
        "tool_rounds": payload.get("tool_rounds", 0),
        "guarded": bool(payload.get("guarded")),
        # 本次用到的历史轮数，方便判断上下文有没有生效
        "memory_turns": payload.get("memory_turns", 0),
        "input_tokens": sum(item.get("input_tokens", 0) for item in trace),
        "output_tokens": sum(item.get("output_tokens", 0) for item in trace),
        "trace": trace,
    }

def get_file_md5(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()

def load_processed_records() -> dict:
    if os.path.exists(PROCESSED_FILES_LOG):
        with open(PROCESSED_FILES_LOG, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_processed_record(key: str, filename: str, product_id: str) -> None:
    records = load_processed_records()
    records[key] = {
        "filename": filename,
        # 指纹里带产品标识，同一份文件可以入到不同产品
        "product_id": product_id,
        "processed_at": datetime.now().isoformat()
    }
    # 容器里这个目录是 /data，不存在就建
    log_dir = os.path.dirname(PROCESSED_FILES_LOG)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(PROCESSED_FILES_LOG, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

@router.post("/upload")
async def upload_comments(
    file: UploadFile = File(...),
    # 上传时指定产品，统计和检索都按它隔离
    product_id: str = Form("default"),
):
    # 只取文件名本身，防止 ../ 之类的名字把文件写到目录外
    original_name = Path(file.filename or "").name
    if not original_name or not original_name.lower().endswith((".txt", ".csv")):
        raise HTTPException(status_code=400, detail="文件仅支持.txt或.csv格式")

    product_id = (product_id or "default").strip() or "default"

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取文件失败，{str(e)}")

    file_md5 = get_file_md5(content)

    # 去重键是 product_id 加 md5，同一文件能入到多个产品
    record_key = f"{product_id}:{file_md5}"
    records = load_processed_records()
    record = records.get(record_key)
    # 兼容旧日志：旧记录没有 product_id，等价于 default
    if record is None and product_id == "default":
        record = records.get(file_md5)
    if record:
        return{
            "status": "success",
            "message": f"文件{original_name}已经处理过（产品 {product_id}），初次处理时间为：{record['processed_at']}"
        }

    temp_dir = "./temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_filename = f"{timestamp}_{original_name}"
    file_path = os.path.join(temp_dir, safe_filename)

    try:
        with open(file_path, "wb") as buffer:
            buffer.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败：{str(e)}")

    try:
        comments = load_comments(file_path)
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"文件解析失败:{str(e)}")

    if not comments:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail="未提取到有效评论，请检查文件内容")

    try:
        count = add_comments_to_store(comments, original_name, product_id=product_id)
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"向量化入库失败:{str(e)}")

    os.remove(file_path)

    save_processed_record(record_key, original_name, product_id)

    # 返回 SQLite 行数和向量数，方便直接看出索引对不对得上
    try:
        index = index_counts()
    except Exception as e:
        index = {"error": str(e)}

    return{
        "status": "success",
        "message": f"文件{original_name}处理完成（产品 {product_id}），共入库{count}条评论。",
        "index": index
    }

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_comments(request: AnalyzeRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    started = time.perf_counter()

    # 缓存 key 带产品和数据版本，数据变了就不会命中旧答案
    version = data_version(request.product_id)
    # 多轮答案依赖上文，命中单轮缓存会答非所问，所以会话请求跳过缓存
    history = (
        await load_history(request.session_id, request.product_id)
        if request.session_id else []
    )
    cached = None
    if not request.session_id:
        cached = await get_cached_answer(request.question, request.product_id, version)
    if cached:
        latency_ms = (time.perf_counter() - started) * 1000
        metrics.record_request(latency_ms, cached=True)
        # 缓存里也存了 stats，命中时返回的统计和首次一致
        return AnalyzeResponse(
            answer=cached["answer"],
            sources=[],
            stats=cached.get("stats"),
            meta=_build_meta({}, latency_ms, cached=True),
        )

    try:
        initial_state = {
            "question": request.question,
            "question_type": "",
            "context": "",
            "sources": [],
            "answer": "",
            # 产品标识和统计占位一起传给工作流
            "product_id": request.product_id,
            "stats": {},
            "tool_rounds": 0,
            "trace": [],
            # 历史只用来改写追问，统计每轮都重新查库
            "history": history,
            "resolved_question": "",
            "memory_turns": len(history),
        }

        # 工作流是同步阻塞的，放线程池里跑，免得卡住事件循环
        final_state = await run_in_threadpool(agent_app.invoke, initial_state)
        answer = final_state["answer"]
        sources = final_state["sources"]
        # 取出 finalize 写入的统计，随接口返回
        stats = final_state.get("stats") or None
    except Exception as e:
        # 失败请求也计数，/metrics 里能看出错误率
        metrics.record_request((time.perf_counter() - started) * 1000, error=True)
        print(f"Agent运行报错:{e}")
        raise HTTPException(status_code=500, detail=f"Agent工作流执行失败：{str(e)}")

    # 单轮写缓存，多轮写会话历史，下一轮解析追问时用
    if request.session_id:
        await append_turn(
            request.session_id,
            final_state.get("resolved_question") or request.question,
            final_state.get("question_type", ""),
            (stats or {}).get("label", ""),
            request.product_id,
        )
    else:
        await set_cached_answer(request.question, answer, request.product_id, version, stats)

    # 本次耗时、token 和轮次状态一并返回，同时记进进程指标
    latency_ms = (time.perf_counter() - started) * 1000
    meta = _build_meta(final_state, latency_ms)
    metrics.record_request(
        latency_ms,
        guardrail=meta["guarded"],
        input_tokens=meta["input_tokens"],
        output_tokens=meta["output_tokens"],
    )

    return AnalyzeResponse(
        answer=answer,
        sources=sources,
        stats=stats,
        meta=meta,
    )


def _sse(event: str, data: dict) -> str:
    """按 text/event-stream 的格式拼一条事件。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/analyze/stream")
async def analyze_stream(request: AnalyzeRequest):
    """流式问答，每个节点执行完推一帧，最后推答案和 stats。"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    version = data_version(request.product_id)

    async def event_generator():
        started = time.perf_counter()
        # 和同步接口一样跳过缓存，下一轮靠 Redis 历史改写追问
        history = (
            await load_history(request.session_id, request.product_id)
            if request.session_id else []
        )
        cached = None
        if not request.session_id:
            cached = await get_cached_answer(request.question, request.product_id, version)
        if cached:
            metrics.record_request((time.perf_counter() - started) * 1000, cached=True)
            yield _sse("node", {
                "node": "cache",
                "answer": cached["answer"],
                "stats": cached.get("stats"),
                "sources": [],
            })
            yield _sse("done", {"cached": True})
            return

        initial_state = {
            "question": request.question,
            "question_type": "",
            "context": "",
            "sources": [],
            "answer": "",
            "product_id": request.product_id,
            "stats": {},
            "tool_rounds": 0,
            "trace": [],
            "messages": [],
            "history": history,
            "resolved_question": "",
            "memory_turns": len(history),
        }

        recorded = False
        trace = []
        tool_rounds = 0
        memory_turns = len(history)
        question_type = ""
        try:
            # updates 模式下每个节点执行完就推一次，前端能看到走到哪一步
            async for update in agent_app.astream(initial_state, stream_mode="updates"):
                for node, payload in update.items():
                    data = {"node": node}
                    if isinstance(payload, dict):
                        # 流式时 finalize 只给答案，trace 和轮次要在这里从各节点累加
                        trace.extend(payload.get("trace") or [])
                        tool_rounds = max(tool_rounds, payload.get("tool_rounds", 0) or 0)
                        memory_turns = max(
                            memory_turns, payload.get("memory_turns", 0) or 0
                        )
                        for key in (
                            "question_type", "stats", "sources", "answer",
                            "resolved_question",
                        ):
                            if key in payload:
                                data[key] = payload[key]
                    if node == "classify_question":
                        question_type = data.get("question_type", "") or question_type
                    if node == "finalize" and isinstance(payload, dict):
                        # finalize 带上完整 meta，前端可以显示耗时和 token
                        data["meta"] = _build_meta(
                            {
                                **payload,
                                "trace": trace,
                                "tool_rounds": tool_rounds,
                                "memory_turns": memory_turns,
                            },
                            (time.perf_counter() - started) * 1000,
                        )
                    yield _sse("node", data)

                    if node == "finalize" and data.get("answer"):
                        resolved = data.get("resolved_question") or request.question
                        if request.session_id:
                            await append_turn(
                                request.session_id,
                                resolved,
                                question_type or data.get("question_type", ""),
                                (data.get("stats") or {}).get("label", ""),
                                request.product_id,
                            )
                        else:
                            await set_cached_answer(
                                request.question, data["answer"],
                                request.product_id, version, data.get("stats")
                            )
                        meta = data.get("meta") or {}
                        metrics.record_request(
                            meta.get("latency_ms", 0.0),
                            guardrail=meta.get("guarded", False),
                            input_tokens=meta.get("input_tokens", 0),
                            output_tokens=meta.get("output_tokens", 0),
                        )
                        recorded = True
            yield _sse("done", {
                "cached": False,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            })
        except Exception as e:
            if not recorded:
                metrics.record_request((time.perf_counter() - started) * 1000, error=True)
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )