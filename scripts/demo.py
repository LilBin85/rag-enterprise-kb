"""固定演示脚本(蓝图 12 节):按固定顺序演示权限过滤、版本生命周期、引用与拒答。

前置条件(演示时保持运行):
  1. docker compose up -d
  2. conda run -n rag-kb uvicorn rag_kb.api:app --port 8000

用法: python scripts/demo.py [--skip-tests]

脚本开头自动重置演示 collection,可重复运行,不依赖任何手工数据修改(蓝图阶段 5 完成条件)。
首次运行较慢:API 侧首次摄取会加载 bge-m3 模型(CPU 约 30 秒),PDF 经 Docling 解析。
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import yaml

from rag_kb.config import ROOT_DIR, get_settings
from rag_kb.store import ChunkStore

API_BASE = "http://localhost:8000"


def step(number: int, title: str) -> None:
    print(f"\n{'=' * 12} 步骤 {number}: {title} {'=' * 12}")


def show(resp: httpx.Response) -> dict:
    body = resp.json()
    print(json.dumps(body, ensure_ascii=False, indent=2))
    return body


def main() -> None:
    settings = get_settings()
    client = httpx.Client(base_url=API_BASE, timeout=300)

    step(1, "检查 Qdrant 与 FastAPI 连通")
    try:
        health = client.get("/health").json()
    except httpx.ConnectError:
        sys.exit("API 未启动,请先运行: conda run -n rag-kb uvicorn rag_kb.api:app --port 8000")
    print("health:", health)
    if health["qdrant"] != "ok":
        sys.exit("Qdrant 不可达,请先运行: docker compose up -d")

    step(2, "重置演示 collection(可重复运行的前提)")
    store = ChunkStore(settings.qdrant_url, settings.qdrant_collection)
    if store.client.collection_exists(settings.qdrant_collection):
        store.client.delete_collection(settings.qdrant_collection)
    store.ensure_collection(settings.embedding_dim)
    print(f"collection {settings.qdrant_collection} 已重建")

    step(3, "管理员导入语料(差旅 / 研发 / 休假 v1)")
    corpus = ROOT_DIR / "data" / "corpus"
    manifest = yaml.safe_load((corpus / "manifest.yaml").read_text(encoding="utf-8"))
    initial = [d for d in manifest["documents"] if d["version"] == 1]
    for entry in initial:
        metadata = {k: entry[k] for k in ("document_id", "title", "department", "allowed_roles")}
        with (corpus / entry["file"]).open("rb") as fh:
            resp = client.post(
                "/documents",
                files={"file": (entry["file"], fh)},
                data={"metadata": json.dumps(metadata, ensure_ascii=False)},
            )
        body = show(resp)
        assert resp.status_code == 200, body
    time.sleep(1)
    for entry in initial:
        versions = client.get(f"/documents/{entry['document_id']}/versions").json()
        print(f"{entry['document_id']} → {versions['versions']}")

    step(4, "engineer 查询研发发布流程(期望:回答 + 完整引用)")
    resp = client.post(
        "/questions",
        json={"question": "常规发布窗口是什么时间?灰度放量分几个阶段?", "role": "engineer"},
    )
    show(resp)

    step(5, "employee 询问 HR 休假制度(期望:通用拒答,无受限标识)")
    resp = client.post("/questions", json={"question": "员工年假有多少天?", "role": "employee"})
    body = show(resp)
    assert body["verdict"] == "refusal"
    payload = json.dumps(body, ensure_ascii=False)
    for forbidden in ["leave", "休假", "年假", "v1", "v2"]:
        assert forbidden not in payload, f"拒答响应泄露了受限标识: {forbidden}"
    print("泄露检查通过:响应不含任何受限文档标识")

    step(6, "上传休假制度 v2 后 hr 查询(期望:仅引用 v2,年假 15 天)")
    v2_entry = next(
        d for d in manifest["documents"] if d["document_id"] == "hr-leave-policy" and d["version"] == 2
    )
    with (corpus / v2_entry["file"]).open("rb") as fh:
        show(client.put("/documents/hr-leave-policy", files={"file": (v2_entry["file"], fh)}))
    resp = client.post("/questions", json={"question": "员工年假有多少天?", "role": "hr"})
    body = show(resp)
    assert body["verdict"] == "answer"
    assert all(c["document_version"] == 2 for c in body["citations"]), "引用必须全部来自 v2"
    print("版本检查通过:引用全部来自 v2,v1 未再出现")

    step(7, "删除休假制度后 hr 再次查询(期望:无证据拒答)")
    show(client.delete("/documents/hr-leave-policy", params={"mode": "delete"}))
    resp = client.post("/questions", json={"question": "员工年假有多少天?", "role": "hr"})
    body = show(resp)
    assert body["verdict"] == "refusal"

    step(8, "冻结 pytest 回归(覆盖权限、版本、删除、引用与拒答)")
    if "--skip-tests" in sys.argv:
        print("按参数跳过")
    else:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q"],
            cwd=ROOT_DIR, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        print(proc.stdout.strip().splitlines()[-1])
        if proc.returncode != 0:
            sys.exit("冻结测试失败,见上方输出")

    print("\n演示完成:同一问题的可见证据受角色限制,所有结论有可追溯来源,失效文档不再被使用。")


if __name__ == "__main__":
    main()
