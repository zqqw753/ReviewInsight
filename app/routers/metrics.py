"""运维指标接口，返回请求量、错误率、缓存命中、token 和平均耗时。"""
from fastapi import APIRouter

from app.core.metrics import snapshot
from app.services.vector_service import index_counts

router = APIRouter(tags=["运维"])


@router.get("/metrics")
def get_metrics():
    data = snapshot()
    # 索引条数先看，入库成功却检索不到时最快能发现
    try:
        data["index"] = index_counts()
    except Exception as e:
        data["index"] = {"error": str(e)}
    return data
