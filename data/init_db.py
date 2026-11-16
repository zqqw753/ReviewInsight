# data/init_db.py
import os
import sqlite3

# 默认值和 app/core/config.py 对齐，也可以用 DB_PATH 覆盖；
# 不加载 Settings，免得建张表还要先配 DEEPSEEK_API_KEY。
DB_PATH = os.getenv("DB_PATH", "./data/reviews.db")


def init_db():
    # 路径里确实有目录才创建，兼容 DB_PATH=reviews.db 这种写法
    directory = os.path.dirname(DB_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 评论主表，字段和 vector_service 里建表时保持一致
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS reviews (
        review_id TEXT PRIMARY KEY,
        product_id TEXT NOT NULL,
        source TEXT,
        rating REAL,
        review_date TEXT,
        content TEXT NOT NULL,
        aspects TEXT, -- JSON 格式，例如 {"battery": 1, "charging": -1}
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # 按产品查询走这个索引
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_reviews_product ON reviews(product_id)")

    conn.commit()
    conn.close()
    print(f"数据库初始化完成：{DB_PATH}")


if __name__ == "__main__":
    init_db()
