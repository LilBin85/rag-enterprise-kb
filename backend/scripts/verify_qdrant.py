"""阶段 0 连通性验证：确认本机 Conda 环境能连接 Docker 中的 Qdrant。

通过标准：打印 Qdrant 版本与现有 collections 列表；连接失败则抛异常退出。
Qdrant 地址优先读环境变量 QDRANT_URL，默认 http://localhost:6333。
"""

import os

from qdrant_client import QdrantClient

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


def main() -> None:
    client = QdrantClient(url=QDRANT_URL, timeout=5)
    collections = client.get_collections().collections
    print(f"Qdrant 连接成功: {QDRANT_URL}")
    print(f"现有 collections ({len(collections)}): {[c.name for c in collections]}")


if __name__ == "__main__":
    main()
