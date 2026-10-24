import os
from importlib.metadata import metadata
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma
from app.core.embeddings import get_embeddings
from app.core.config import settings

PERSIST_DIR = settings.chroma_persist_dir

def get_vector_store():
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="review_insight"
    )

def add_comments_to_store(comments: List[str], source_name: str) -> int:
    if not comments:
        return 0

    documents = [
        Document(
            page_content=comment,
            metadata={"source": source_name}
        )
        for comment in comments
    ]
    embeddings = get_embeddings()

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name="review_insight"
    )

    return len(comments)

def search_comments(query: str, k: int = 10) -> Tuple[List[str], List[str]]:
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)

    contents = [doc.page_content for doc in docs]
    sources = list(set([doc.metadata.get("source", "未知来源") for doc in docs]))

    return contents, sources