# 重复入库不能产生重复向量，索引条数要和 SQLite 一致
import hashlib

from langchain_core.embeddings import Embeddings

import app.services.vector_service as vector_service


class FakeEmbeddings(Embeddings):
    """固定维度的确定性向量，避免测试下载真实模型。"""

    def _vector(self, text: str):
        digest = hashlib.md5(text.encode("utf-8")).digest()
        return [byte / 255 for byte in digest[:8]]

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


def _patch_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(vector_service, "PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(vector_service, "DB_PATH", str(tmp_path / "reviews.db"))
    monkeypatch.setattr(vector_service, "get_embeddings", lambda: FakeEmbeddings())


def test_reupload_is_idempotent(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    comments = ["电池不耐用，续航衰减", "外观漂亮，手感不错"]

    vector_service.add_comments_to_store(comments, "sample.txt")
    vector_service.add_comments_to_store(comments, "sample.txt")

    counts = vector_service.index_counts()
    assert counts == {"reviews": 2, "vectors": 2}


def test_rebuild_matches_sqlite(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    comments = ["电池不耐用", "发热严重", "外观漂亮"]
    vector_service.add_comments_to_store(comments, "sample.txt")

    result = vector_service.rebuild_vector_index()
    assert result == {"reviews": 3, "vectors": 3}
