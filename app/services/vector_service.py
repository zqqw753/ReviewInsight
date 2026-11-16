import os
import hashlib
import sqlite3
import json
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma
from app.core.embeddings import get_embeddings
from app.core.config import settings
# 抽取和统计共用一套词表和规则，两边口径不会打架
from app.services.stats_service import extract_aspects

PERSIST_DIR = settings.chroma_persist_dir
DB_PATH = settings.db_path
# 集合名只在这里定义，写入和重建索引用同一份
COLLECTION_NAME = "review_insight"

def make_review_id(product_id: str, index: int, content: str) -> str:
    raw = f"{product_id}:{index}:{content.strip()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

def get_vector_store():
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

def upsert_reviews(rows: list[dict]) -> int:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            review_id TEXT PRIMARY KEY,
            product_id TEXT NOT NULL,
            source TEXT,
            rating REAL,
            review_date TEXT,
            content TEXT NOT NULL,
            aspects TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_reviews_product ON reviews(product_id)")


    cursor.executemany(
        """INSERT OR REPLACE INTO reviews
            (review_id, product_id, source, content, aspects)
            VALUES (:review_id, :product_id, :source, :content, :aspects)""",
        rows
    )
    conn.commit()
    conn.close()
    return len(rows)

def add_comments_to_store(comments: List[str], source_name: str, product_id: str = "default") -> int:
    if not comments:
        return 0

    rows = []
    documents = []

    for i, comment in enumerate(comments):
        review_id = make_review_id(product_id, i, comment)
        aspects = extract_aspects(comment)

        rows.append({
            "review_id": review_id,
            "product_id": product_id,
            "source": source_name,
            "content": comment,
            "aspects": json.dumps(aspects, ensure_ascii=False)
        })

        documents.append(
            Document(
                page_content=comment,
                metadata={
                    "review_id": review_id,
                    "product_id": product_id,
                    "source": source_name
                }
            )
        )

    upsert_reviews(rows)

    # 用稳定 review_id 做主键，先删旧的再写，重复上传不会产生重复向量
    ids = [row["review_id"] for row in rows]
    vector_store = get_vector_store()
    vector_store.delete(ids=ids)
    vector_store.add_documents(documents=documents, ids=ids)

    return len(comments)


def count_vectors() -> int:
    """当前集合里的向量数，用来和 SQLite 行数对账。"""
    return get_vector_store()._collection.count()


def index_counts() -> dict:
    """SQLite 评论数和向量数的对照，上传接口和评测脚本都会用。"""
    conn = sqlite3.connect(DB_PATH)
    try:
        reviews = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    finally:
        conn.close()
    return {"reviews": reviews, "vectors": count_vectors()}


def rebuild_vector_index() -> dict:
    """以 SQLite 为准重建整个向量索引，用来清掉历史重复向量。"""
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT review_id, product_id, source, content FROM reviews"
        ).fetchall()
    finally:
        conn.close()

    documents = [
        Document(
            page_content=content,
            metadata={"review_id": review_id, "product_id": product_id, "source": source}
        )
        for review_id, product_id, source, content in rows
    ]

    store = get_vector_store()
    store.delete_collection()

    if documents:
        Chroma.from_documents(
            documents=documents,
            embedding=get_embeddings(),
            persist_directory=PERSIST_DIR,
            collection_name=COLLECTION_NAME,
            ids=[row[0] for row in rows]
        )

    return {"reviews": len(rows), "vectors": count_vectors()}

def search_comments(query: str, k: int = 10, product_id: str = "default") -> Tuple[List[str], List[str]]:
    vector_store = get_vector_store()
    # 按 product_id 过滤，证据和 SQLite 统计是同一批评论
    retriever = vector_store.as_retriever(
        search_kwargs={"k": k, "filter": {"product_id": product_id}}
    )
    docs = retriever.invoke(query)

    contents = [doc.page_content for doc in docs]
    # 用 dict.fromkeys 保序去重，set 的顺序受哈希随机化影响，每次请求可能不一样
    sources = list(dict.fromkeys(doc.metadata.get("source", "未知来源") for doc in docs))

    return contents, sources