# 统计服务以 SQLite 为准，这里验证口径和分母
import json
import sqlite3

import app.services.stats_service as stats_service


def _make_db(path):
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE reviews (
            review_id TEXT PRIMARY KEY,
            product_id TEXT NOT NULL,
            source TEXT,
            content TEXT,
            aspects TEXT,
            created_at TEXT
        )"""
    )
    rows = [
        ("r1", "default", "a.txt", "电池不耐用", {"battery": -1}, "2026-10-28 10:00:00"),
        ("r2", "default", "a.txt", "电池还行", {"battery": 1}, "2026-10-28 10:00:00"),
        ("r3", "default", "a.txt", "续航很差", {"battery": -1}, "2026-10-28 10:00:00"),
        ("r4", "default", "a.txt", "外观漂亮", {"design": 1}, "2026-10-28 10:00:00"),
        ("r5", "other", "b.txt", "电池很差", {"battery": -1}, "2026-10-28 11:00:00"),
    ]
    conn.executemany(
        "INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?)",
        [(rid, pid, src, content, json.dumps(aspects), created) for rid, pid, src, content, aspects, created in rows],
    )
    conn.commit()
    conn.close()


def test_count_aspect(monkeypatch, tmp_path):
    db = tmp_path / "reviews.db"
    _make_db(db)
    monkeypatch.setattr(stats_service, "DB_PATH", str(db))

    result = stats_service.count_aspect("default", "battery")
    assert result["mentions"] == 3
    assert result["negative"] == 2
    assert result["positive"] == 1
    assert result["negative_ratio"] == 0.67
    assert result["total_reviews_in_db"] == 4  # 只是展示样本规模


def test_resolve_aspect_keyword_and_fallback(monkeypatch, tmp_path):
    db = tmp_path / "reviews.db"
    _make_db(db)
    monkeypatch.setattr(stats_service, "DB_PATH", str(db))

    assert stats_service.resolve_aspect("续航怎么样？") == "battery"
    assert stats_service.resolve_aspect("外观好看吗") == "design"
    # 没命中关键词时，退化成提及最多的方面
    assert stats_service.resolve_aspect("屏幕怎么样") == "battery"


def test_data_version_changes_with_new_rows(monkeypatch, tmp_path):
    db = tmp_path / "reviews.db"
    _make_db(db)
    monkeypatch.setattr(stats_service, "DB_PATH", str(db))

    before = stats_service.data_version("default")
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO reviews VALUES ('r6', 'default', 'a.txt', '新评论', '{}', '2026-10-28 12:00:00')"
    )
    conn.commit()
    conn.close()
    after = stats_service.data_version("default")

    assert before != after
