import os
from langchain_huggingface import HuggingFaceEmbeddings

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        model_kwargs={"device" : "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )