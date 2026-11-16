import os
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


@lru_cache(maxsize=1)
def get_embeddings():
    """嵌入模型按进程缓存，每次检索重新加载要多花几百毫秒。"""
    return HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
