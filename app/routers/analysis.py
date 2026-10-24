import os
import hashlib
import json
from datetime import datetime
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek

from app.core.config import settings
from app.services.vector_service import add_comments_to_store, search_comments
from app.utils.file_loader import load_comments
from app.services.cache_service import get_cached_answer, set_cached_answer
from app.services.agent_service import agent_app

PROCESSED_FILES_LOG = "./processed_files.json"

class AnalyzeRequest(BaseModel):
    question: str

class AnalyzeResponse(BaseModel):
    answer: str
    sources: List[str]

router = APIRouter(prefix="/api/v1/analysis", tags=["选品分析"])

def get_file_md5(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()

def load_processed_records() -> dict:
    if os.path.exists(PROCESSED_FILES_LOG):
        with open(PROCESSED_FILES_LOG, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_processed_record(md5: str, filename: str) -> None:
    records = load_processed_records()
    records[md5] = {
        "filename": filename,
        "processed_at": datetime.now().isoformat()
    }
    with open(PROCESSED_FILES_LOG, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

@router.post("/upload")
async def upload_comments(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith((".txt", ".csv")):
        raise HTTPException(status_code=400, detail="文件仅支持.txt或.csv格式")

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取文件失败，{str(e)}")

    file_md5 = get_file_md5(content)

    records = load_processed_records()
    if file_md5 in records:
        return{
            "status": "success",
            "message": f"文件{file.filename}已经处理过，初次处理时间为：{records[file_md5]['processed_at']}"
        }

    temp_dir = "./temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_filename = f"{timestamp}_{file.filename}"
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
        count = add_comments_to_store(comments, file.filename)
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"向量化入库失败:{str(e)}")

    os.remove(file_path)

    save_processed_record(file_md5, file.filename)

    return{
        "status": "success",
        "message": f"文件{file.filename}处理完成，共入库{count}条评论。"
    }

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_comments(request: AnalyzeRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    cached = await get_cached_answer(request.question)
    if cached:
        return AnalyzeResponse(answer=cached, sources=[])

    try:
        initial_state = {
            "question": request.question,
            "question_type": "",
            "context": "",
            "sources": [],
            "answer": ""
        }

        final_state = agent_app.invoke(initial_state)
        answer = final_state["answer"]
        sources = final_state["sources"]

    except Exception as e:
        print(f"Agent运行报错:{e}")
        raise HTTPException(status_code=500, detail=f"Agent工作流执行失败：{str(e)}")

    await set_cached_answer(request.question, answer)

    return AnalyzeResponse(
        answer=answer,
        sources=sources
    )