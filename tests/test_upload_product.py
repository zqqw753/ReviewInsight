# 上传要把 product_id 传到入库逻辑，去重也按产品隔离
from fastapi.testclient import TestClient

import app.routers.analysis as analysis
from app.main import app


def test_upload_passes_product_id(monkeypatch, tmp_path):
    calls = []

    def fake_add(comments, source_name, product_id="default"):
        calls.append((product_id, len(comments)))
        return len(comments)

    monkeypatch.setattr(analysis, "PROCESSED_FILES_LOG", str(tmp_path / "processed.json"))
    monkeypatch.setattr(analysis, "add_comments_to_store", fake_add)
    monkeypatch.setattr(analysis, "index_counts", lambda: {"reviews": 2, "vectors": 2})

    client = TestClient(app)
    csv_bytes = "comment\n电池很耐用\n外观漂亮\n".encode("utf-8")
    response = client.post(
        "/api/v1/analysis/upload",
        files={"file": ("real_reviews.csv", csv_bytes, "text/csv")},
        data={"product_id": "p-001"},
    )

    assert response.status_code == 200
    assert calls == [("p-001", 2)]
    body = response.json()
    assert body["index"] == {"reviews": 2, "vectors": 2}
    assert "p-001" in body["message"]


def test_same_file_can_belong_to_two_products(monkeypatch, tmp_path):
    calls = []

    def fake_add(comments, source_name, product_id="default"):
        calls.append(product_id)
        return len(comments)

    monkeypatch.setattr(analysis, "PROCESSED_FILES_LOG", str(tmp_path / "processed.json"))
    monkeypatch.setattr(analysis, "add_comments_to_store", fake_add)
    monkeypatch.setattr(analysis, "index_counts", lambda: {"reviews": 2, "vectors": 2})

    client = TestClient(app)
    csv_bytes = "comment\n电池很耐用\n外观漂亮\n".encode("utf-8")
    for product_id in ("p-001", "p-002"):
        response = client.post(
            "/api/v1/analysis/upload",
            files={"file": ("real_reviews.csv", csv_bytes, "text/csv")},
            data={"product_id": product_id},
        )
        assert response.status_code == 200

    # 同一份文件第二次没有被“已处理”短路，说明去重键按产品隔离
    assert calls == ["p-001", "p-002"]
