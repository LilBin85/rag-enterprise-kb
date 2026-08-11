# RAG 企业知识库智能体

> 第一次使用？请先看 [使用指南.md](使用指南.md)（新手向全流程：启动、演示、提问、测试、FAQ）。

面向企业制度与技术规范文档的、可审计且受权限约束的检索增强问答原型：系统只在用户有权访问的文档中检索，为回答附上可定位的来源引用；没有足够授权证据时明确拒答。

在不改变 `/questions` 契约的前提下，项目另有一个安全收敛的 `POST /policy-comparisons` 工作流：Agent 只能调用服务端绑定的两个只读工具，比较两个明确版本；ACL、生命周期、确定性 diff 和引用均由应用层控制。

> 本项目为求职作品集原型，验证范围以冻结测试语料覆盖的场景为准，不宣称生产就绪。

## 技术栈

- FastAPI + Uvicorn：HTTP API 层
- Docling：Markdown / PDF 文档解析
- bge-m3（FlagEmbedding，CPU）：dense 向量检索基线
- Qdrant（Docker Compose）：向量数据库，payload 携带版本、ACL 与来源元数据
- DeepSeek API：经 `ChatDeepSeek`（langchain-deepseek 组件级封装）生成回答，`with_structured_output` 约束拒答结构
- 受限策略 Agent：使用已锁定的 `langchain==1.3.14` `create_agent`，仅运行服务端绑定的有界只读工具循环
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
Copy-Item .env.example .env   # 然后填入真实 DEEPSEEK_API_KEY 和本地 DEMO_ROLE_TOKENS

# 4. 启动 API 服务(脚本内含 PYTHONUTF8,修复中文 Windows 下 Docling 的 GBK 编码问题)
.\scripts\start_api.ps1
```

## 环境变量

| 变量 | 必需 | 说明 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | 是 | DeepSeek API 密钥，只存在于本地 `.env`，不提交 Git |
| `DEEPSEEK_BASE_URL` | 否 | 默认 `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | 否 | 默认 `deepseek-v4-flash` |
| `DEMO_ROLE_TOKENS` | 策略比较必需 | 本地演示角色令牌 JSON；生产环境应由真实认证上下文替代 |
| `QDRANT_URL` | 否 | 默认 `http://localhost:6333` |
| `HF_ENDPOINT` | 否 | HuggingFace 下载端点；国内网络建议 `https://hf-mirror.com` |
| `AGENT_MAX_STEPS` | 否 | 策略比较 Agent 最多执行步数，默认 `4` |
| `AGENT_MAX_TOOL_CALLS` | 否 | 策略比较 Agent 最多只读工具调用数，默认 `3` |
| `AGENT_TIMEOUT_SECONDS` | 否 | 单次 DeepSeek Agent 模型调用超时，默认 `20` |

## 固定演示

服务启动后，运行蓝图第 12 节的固定演示脚本（可重复运行，开头自动重置数据）：

```powershell
python scripts/demo.py                # 完整演示(含冻结测试回归)
python scripts/demo.py --skip-tests   # 仅演示 API 行为
```

演示顺序：导入语料 → engineer 查询（回答+引用）→ employee 越权查询（零泄露拒答）→ 上传 v2（仅引用新版）→ 删除后查询（无证据拒答）→ 冻结 pytest 回归。

## 验证与测试

```powershell
python -m pytest tests/ -q            # 冻结测试(默认脱离真实模型与真实 API)
python scripts/benchmark_policy_comparisons.py  # 使用 manifest/固定语料 fixture，结果写入 data/local/benchmarks/
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
| `schemas/` | 基准报告 JSON Schema |
| `docker-compose.yml` | 仅运行 Qdrant |

## 安全边界

- API 密钥只从环境变量 / 本地 `.env` 读取，不提交 Git。
- 检索文档、用户问题与模型输出均视为不可信数据；文档中的指令性文本不获得任何执行能力。
- `/policy-comparisons` 只允许读取请求绑定的两个正版本；角色必须通过服务端配置的 `X-Demo-Role-Token` 校验。只比较 `active`/`superseded`，越权、撤回、删除、缺失、非法或模型/工具失败均统一拒答。
- 上传文件名和落盘目录使用的 `document_id` 只允许作为路径单元使用，含路径分隔符的输入会在落盘前拒绝；每次上传使用独立落盘名，失败时只清理本次原件，不覆盖或删除既有版本原件。
- `/health` 在 Qdrant 不可达时返回 HTTP 503，而不是把依赖故障报告为健康。
- Agent 不可选择角色、文档、集合、生命周期状态或 ACL；工具白名单只有授权版本证据读取和确定性比较，无写入、网络、文件、MCP 或任意 Qdrant 工具。
- 运行日志只记录 `run_id`、阶段、角色、文档哈希、计数、耗时和错误码；不记录密钥、提示词、正文或完整模型输出。
- 详细设计决策见桌面《RAG企业知识库智能体项目蓝图》。
