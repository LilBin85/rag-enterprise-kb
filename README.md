# RAG 企业知识库智能体

> 第一次使用？请先看 [使用指南.md](使用指南.md)（新手向全流程：启动、演示、提问、测试、FAQ）。

面向企业制度与技术规范文档的、可审计且受权限约束的检索增强问答原型：系统只在用户有权访问的文档中检索，为回答附上可定位的来源引用；没有足够授权证据时明确拒答。

> 本项目为求职作品集原型，验证范围以冻结测试语料覆盖的场景为准，不宣称生产就绪。

## 技术栈

- FastAPI + Uvicorn：HTTP API 层
- Docling：Markdown / PDF 文档解析
- bge-m3（FlagEmbedding，CPU）：dense 向量检索基线
- Qdrant（Docker Compose）：向量数据库，payload 携带版本、ACL 与来源元数据
- DeepSeek API：经 `ChatDeepSeek`（langchain-deepseek 组件级封装）生成回答，`with_structured_output` 约束拒答结构
- pytest：冻结语料回归测试（默认 mock DeepSeek 调用）

## 快速启动

前置：Docker Desktop 已启动；Conda 环境已创建（`conda env create` 或按蓝图 4.5 节手动安装）。

```powershell
# 1. 启动 Qdrant（项目唯一容器依赖）
docker compose up -d

# 2. 激活环境并验证连通性
conda activate rag-kb
python scripts/verify_qdrant.py

# 3. 配置密钥
Copy-Item .env.example .env   # 然后填入真实 DEEPSEEK_API_KEY

# 4. 启动 API 服务(脚本内含 PYTHONUTF8,修复中文 Windows 下 Docling 的 GBK 编码问题)
.\scripts\start_api.ps1
```

## 环境变量

| 变量 | 必需 | 说明 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | 是 | DeepSeek API 密钥，只存在于本地 `.env`，不提交 Git |
| `DEEPSEEK_BASE_URL` | 否 | 默认 `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | 否 | 默认 `deepseek-v4-flash` |
| `QDRANT_URL` | 否 | 默认 `http://localhost:6333` |
| `HF_ENDPOINT` | 否 | HuggingFace 下载端点；国内网络建议 `https://hf-mirror.com` |

## 固定演示

服务启动后，运行蓝图第 12 节的固定演示脚本（可重复运行，开头自动重置数据）：

```powershell
python scripts/demo.py                # 完整演示(含冻结测试回归)
python scripts/demo.py --skip-tests   # 仅演示 API 行为
```

演示顺序：导入语料 → engineer 查询（回答+引用）→ employee 越权查询（零泄露拒答）→ 上传 v2（仅引用新版）→ 删除后查询（无证据拒答）→ 冻结 pytest 回归。

## 验证与测试

```powershell
python -m pytest tests/ -q            # 冻结测试(57 项,默认脱离真实模型与真实 API)
python scripts/smoke_real_ingest.py   # 手动冒烟:真 bge-m3 + 真 Docling PDF 摄取
python scripts/smoke_answer.py        # 手动冒烟:真 DeepSeek 端到端问答
python scripts/download_bge_m3.py     # 预下载 bge-m3(经 hf-mirror 镜像)
```

## 目录说明

| 路径 | 用途 |
| --- | --- |
| `src/rag_kb/` | 应用代码（摄取、检索、回答、拒答策略） |
| `tests/` | 冻结 pytest 测试 |
| `data/corpus/` | 冻结测试语料（自造模拟文档，需提交） |
| `scripts/` | 环境验证等辅助脚本 |
| `docker-compose.yml` | 仅运行 Qdrant |

## 安全边界

- API 密钥只从环境变量 / 本地 `.env` 读取，不提交 Git。
- 检索文档、用户问题与模型输出均视为不可信数据；文档中的指令性文本不获得任何执行能力。
- 详细设计决策见桌面《RAG企业知识库智能体项目蓝图》。
