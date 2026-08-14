---
title: RAG 企业知识库智能体项目蓝图
status: MVP + 受限策略比较与策略落地规划工作流——原有问答路径保持兼容，两条 Agent 路径均由聚焦测试验证
project_type: 应届生求职作品集
---

# RAG 企业知识库智能体项目蓝图

## 1. 项目定位

### 1.1 一句话介绍

这是一个面向企业制度与技术规范文档的、**可审计且受权限约束的检索增强问答原型**：系统只在用户有权访问的文档中检索，为回答附上可定位的来源引用；没有足够授权证据时明确拒答。

### 1.2 求职作品集价值

项目的亮点不是“调用大模型聊天”，而是展示完整的工程判断：

- 权限在检索阶段生效，而不是让模型自行判断是否泄露内容。
- 回答可回溯到有效文档版本中的具体位置。
- 文档更新、撤回和删除后，旧内容不会继续被默认检索。
- 无证据、无权限和证据不足均是可验证的拒答场景。
- 冻结测试数据与 `pytest` 回归测试使关键行为可以重复验证。
- 新增的策略比较不是开放式自治 Agent：它只在服务端 ACL 后读取两个明确版本，并把 diff、引用和拒答留在应用层。
- 新增的策略落地规划是加法式 v1：它只动态检索当前角色可见的 `active` 证据，持久化任务目标、角色、状态、澄清和只读计划版本，但不执行 Todo。

### 1.3 不应使用的表述

项目完成前后均不宣称“企业级”“生产就绪”“安全合规”“零幻觉”“支持任意规模知识库”或“高准确率”。可准确描述为：在固定测试语料和已覆盖的场景内，验证了权限过滤、引用、拒答与文档版本生命周期。

## 2. 问题场景与目标用户

### 2.1 场景

模拟一家公司的内部知识库，包含公开制度、研发规范和人力资源制度。员工希望快速查询制度或规范，但不同角色能访问的文档不同。

示例角色：

| 角色 | 可访问内容 | 不可访问内容 |
| --- | --- | --- |
| `employee` | 全员制度、公开技术规范 | 人力资源内部制度、受限研发文档 |
| `engineer` | 全员制度、研发规范 | 人力资源内部制度 |
| `hr` | 全员制度、人力资源制度 | 受限研发文档 |
| `admin` | 所有测试文档 | 无 |

### 2.2 目标用户

- 普通员工：查询公开流程、报销规则、常见技术规范。
- 研发人员：查询研发制度、接口规范、故障处理流程。
- HR 人员：查询人事流程与仅 HR 可见的制度。
- 文档管理员：上传、更新、撤回和删除文档。

### 2.3 MVP 目标

完成一个本地可运行、可演示、可测试的最小闭环：

1. 导入有限数量的 Markdown 与 PDF 测试文档。
2. 为每个文档块保存版本、来源位置和角色权限元数据。
3. 以授权用户身份检索时，只从允许访问的有效文档块中召回结果。
4. 经由 `ChatDeepSeek`（LangChain 组件级）调用 DeepSeek API，基于检索证据生成带结构化约束的回答，并附带引用。
5. 无授权证据、无检索结果或证据不足时，返回标准化拒答。
6. 更新、撤回或删除文档后，默认检索不再返回失效版本。
7. 使用冻结样本通过 `pytest` 验证上述行为。
8. 提供 `POST /policy-comparisons`，在有界只读工具循环中比较两个请求版本。
9. 提供认证的 `/policy-rollout-tasks` 规划工作流，保存可继续的任务和不可变计划版本，不触发外部写入或任务执行。

## 3. MVP 范围与明确边界

### 3.1 MVP 内必须实现的能力

- Windows 本地开发环境。
- FastAPI 作为应用接口层。
- Docling 解析受支持的 Markdown 与 PDF 文档。
- `bge-m3` 的 dense 向量模式作为单一检索基线。
- Qdrant 作为向量数据库，使用 Docker Desktop 与 Docker Compose 运行。
- DeepSeek API 作为唯一的生成模型后端。
- 基于用户角色的检索时 ACL 过滤。
- 文档、版本和文档块级来源引用。
- 无证据、无权限、证据不足时的拒答策略。
- 文档新增、更新、撤回、删除的版本生命周期。
- 基于固定语料和固定断言的 `pytest` 测试集。

### 3.2 MVP 明确不做的能力

以下项目不属于首版交付，不能因“看起来高级”而提前引入：

- 多 Agent 协作、开放式自治工具调用、MCP 市场接入。
- `/questions` 仍采用原有线性问答链；策略比较只使用两个固定只读工具，不扩展为通用 Agent 平台。
- 混合检索、BM25、稀疏向量、RRF 融合、查询改写。
- reranker、ColBERT、多向量检索。
- LangGraph、Langfuse、Ragas。
- PostgreSQL、Redis、MinIO、异步任务队列；策略落地规划 v1 仅使用本地 SQLite 保存最小业务状态。
- vLLM、Ollama、本地大模型与 GPU 依赖。
- SSO、SAML、SCIM、真实企业身份系统与复杂多租户。
- 云部署、Kubernetes、对象存储、监控告警平台。
- OCR、多模态问答、网页爬取、企业网盘同步。

### 3.3 取舍原则

只有当冻结评测能证明当前实现存在明确瓶颈时，才讨论增加新技术。例如，只有 dense 检索在固定的编号、日期或关键词查询中稳定失败，才评估混合检索；只有 Top-K 证据稳定不相关，才评估 reranker。

## 4. 开发环境与软件依赖

### 4.1 已具备条件

- Miniconda 与 Python。
- Git。
- IDE。
- 可用的 DeepSeek API。
- Docker Desktop、WSL 2 后端已安装，Docker Engine 已验证可正常启动。

### 4.2 当前环境验证进度（2026-08-06）

- 已完成 C 盘扩容，目前可用容量为 55.3 GB。
- WSL 2.7.11 已安装成功。获取 Linux 发行版列表时曾发生 GitHub 请求超时，但不影响 Docker Desktop 使用 WSL 2 后端。
- Docker Desktop 的 Engine 已正常启动。
- 已执行 `docker run --rm hello-world`，镜像拉取、容器创建与运行均成功。
- 已可在浏览器打开 Qdrant Dashboard（`http://localhost:6333/dashboard`）。
- 阶段 0 的 `docker compose`、Conda 环境和 FastAPI/Qdrant 连通性已在后续收尾记录中验证；本行仅保留为历史过程，不作为当前状态判断。

**2026-08-07 阶段 0 收尾（已执行）**：

- `docker compose version` → v5.3.1，Compose 命令验证通过。
- 项目骨架建于 `D:\PythonProject\rag-enterprise-kb`；Qdrant 最小 Compose 配置落地（`qdrant/qdrant:v1.19.0`，本地镜像打 tag 锁定），`up`（healthy、`/healthz` 200）与 `down`（容器与网络干净移除、命名数据卷保留）全生命周期验证通过。
- Conda 环境 `rag-kb` 已创建（Python 3.13.14），4.5 节全部依赖安装完成并全量锁定至 `requirements.txt`（torch 为 CPU 版，文件内含 CPU 源声明）；`torch`/`docling`/`FlagEmbedding`/`qdrant_client`/`fastapi`/`langchain_deepseek` import 冒烟通过。
- `scripts/verify_qdrant.py` 已从本机 Conda 环境成功连接容器 Qdrant；FastAPI 封装属阶段 1。
- 用户级 `HF_HOME` 指向 `D:\DevelopmentTools\hf-cache`，避免模型缓存占用 C 盘。

### 4.3 环境验证门槛

在创建项目代码前，必须确认 Docker Desktop 的 Engine 已正常启动，并能运行 Docker Compose。当前 Docker 的 WSL 2 与虚拟化后端问题已不再阻塞；Docker Engine、Docker Compose 和 Qdrant 均已验证。

### 4.4 主机软件

| 软件 | 是否必需 | 用途 |
| --- | --- | --- |
| Windows 10/11 64 位 | 必需 | 本地开发系统 |
| Miniconda / Python 3.13 | 必需 | 隔离 Python 开发环境（实际锁定 3.13.14，见 requirements.txt） |
| Git | 必需 | 代码、固定语料与测试资产版本管理 |
| Docker Desktop + Compose v2 | 必需 | 仅运行 Qdrant 容器 |
| IDE | 必需 | 编写、调试与运行 Python 代码 |
| DeepSeek API | 必需 | 生成带引用的回答 |

不需要安装 Ollama、Node.js、PostgreSQL、Redis、MinIO 或 GPU 驱动作为 MVP 前置条件。

### 4.5 Python 依赖

未来创建独立 Conda 环境后，项目依赖应包括：

| 依赖 | 作用 |
| --- | --- |
| `fastapi` | HTTP API 框架 |
| `uvicorn` | 本地 ASGI 服务运行器 |
| `pydantic-settings` | 环境变量与配置加载 |
| `python-multipart` | 文件上传支持 |
| `docling` | 文档解析与结构化内容提取 |
| `FlagEmbedding` | 加载与调用 `bge-m3` embedding 模型 |
| `torch` | `FlagEmbedding` 推理依赖，首版使用 CPU 方案 |
| `qdrant-client` | 与 Qdrant 服务通信 |
| `langchain` | 仅用于策略比较路径的 `create_agent` 有界工具循环；不直接使用 LangGraph API |
| `langchain-deepseek` | 以 `ChatDeepSeek` 组件级封装 DeepSeek 调用，并用 `with_structured_output` 约束回答与拒答的输出结构（连带 `langchain-core`、`langchain-openai`，版本在初始化时锁定） |
| `httpx` | FastAPI TestClient 依赖与一般 HTTP 请求 |
| `pytest` | 冻结测试与回归验证 |

依赖版本应在项目实际初始化后锁定到项目配置文件中；本规划不伪造未验证的具体版本号。已于 2026-08-07 完成初始化并全量锁定至 `requirements.txt`（torch 为 CPU 版，经 `--extra-index-url` 声明解析）。

## 5. 总体架构

### 5.1 组件边界

```text
                    ┌──────────────────────────────────────┐
                    │          本机 Conda Python 环境        │
                    │                                      │
用户 / 演示客户端 ──>│  FastAPI                              │
                    │    ├─ 身份上下文与 ACL 策略            │
                    │    ├─ Docling 文档解析                 │
                    │    ├─ 切块与元数据构建                 │
│    ├─ bge-m3 dense embedding          │
│    ├─ ChatDeepSeek 生成封装            │
│    ├─ 引用与拒答策略                   │
                    │    └─ pytest 冻结测试                 │
                    └──────────────┬───────────┬───────────┘
                                   HTTP        HTTPS
                                     │           │
                    ┌────────────────▼───┐  ┌────▼─────────────┐
                    │ Qdrant 容器         │  │ DeepSeek API      │
                    │ Docker Desktop 管理 │  │ 远程生成模型服务  │
                    └────────────────────┘  └──────────────────┘
```

Docker Desktop 在 MVP 中**只运行 Qdrant**。FastAPI、Docling、`bge-m3`、测试代码和 DeepSeek API 调用都运行在本机 Conda 环境。DeepSeek 是远程 API，不需要放入 Docker 容器。

### 5.2 查询数据流

```text
请求携带用户角色与问题
        │
        ▼
FastAPI 校验请求并构造授权上下文
        │
        ▼
问题使用 bge-m3 生成 dense 向量
        │
        ▼
Qdrant 在查询中同时应用：status=active + allowed_roles ACL 过滤
        │
        ▼
仅将已授权的有效文档块组成上下文
        │
        ▼
DeepSeek API 基于上下文回答，或由应用层拒答
        │
        ▼
返回 answer / refusal + document_id + version + source_locator + chunk_id
```

### 5.3 受限策略比较数据流

```text
请求携带 document_id、from_version、to_version、role
        │
        ▼
服务端校验正整数、顺序和请求绑定；模型不能修改这些值
        │
        ▼
Qdrant 按 document_id + version 完整读取；应用层校验 status(active/superseded)、allowed_roles 与块集合完整性
        │
        ▼
只把授权版本证据交给两个固定工具：fetch_authorized_version / compare_authorized_versions
        │
        ▼
应用层执行确定性章节 diff，应用层从实际证据构造 Citation
        │
        ▼
结构化 Agent 摘要 + diff + 引用；任一异常、越权或护栏超限均通用拒答
```

该路径使用 `langchain.agents.create_agent`，循环最多 4 步、最多 3 次工具调用，不提供文件、HTTP、网络、写入、删除、上传、MCP、Memory 或任意 Qdrant 工具。真实 `ChatDeepSeek` 仅在运行时懒加载；`/questions` 仍保持原有线性流程与响应契约。

### 5.4 受限策略落地规划数据流（v1）

```text
POST /policy-rollout-tasks
  → X-Demo-Role-Token 解析并绑定 Role，body.role 必须一致
  → SQLite 先创建 planning 任务
  → ChatDeepSeek.with_structured_output(PlannerDecision) 输出 search/clarify/propose/refuse 之一
  → 应用层按决策执行绑定角色的 Retriever，校验 Todo、依赖关系和检索候选，并构造 Citation
  → SQLite 更新为 planned/clarification/failed
GET /policy-rollout-tasks/{task_id}
  → 按 task_id + role 读取，并重新校验引用当前仍为 active 且角色有权访问
POST /policy-rollout-tasks/{task_id}/continue
  → CAS 领取 clarification，再追加下一条不可变计划版本
```

该工作流使用显式结构化 `PlannerDecision`，每次决策只能是 `search`、`clarify`、`propose` 或 `refuse`。应用层控制最多 6 次模型决策、3 次检索和 2 次修复；重复查询反馈会先归一化去重。只读检索通过绑定角色的 `Retriever` 动态应用 `active` 与 `allowed_roles`，CAS 保护澄清领取和计划追加。SQLite 只保存任务目标、绑定角色、状态、澄清和追加式计划版本，不保存证据正文或模型原始输出；Citation 由应用层根据本次检索候选构造。它不执行 Todo，不写外部系统，也没有 Agent graph 的 durable execution、checkpointer 或断点续跑能力。同步规划期间若进程被强制终止，任务可能停留在 `planning` 状态，这是 v1 的窄化崩溃边界。

### 5.5 摄取数据流

```text
管理员上传或更新文档
        │
        ▼
文件类型、大小与角色权限校验
        │
        ▼
Docling 解析文档，提取标题、页码或段落位置
        │
        ▼
按结构化边界切块，计算内容哈希
        │
        ▼
bge-m3 为每个块生成 dense 向量
        │
        ▼
写入 Qdrant：向量 + 版本 + ACL + 来源元数据
        │
        ▼
更新时停用旧版本；删除或撤回时使全部相关块不可检索
```

## 6. 文档与向量元数据设计

### 6.1 Qdrant payload 字段

每一个向量点都必须携带以下 payload。向量本身不能脱离版本、来源与权限单独存在。

| 字段 | 示例 | 用途 |
| --- | --- | --- |
| `document_id` | `hr-leave-policy` | 稳定文档标识，不随版本变化 |
| `document_version` | `2` | 当前文档版本号 |
| `chunk_id` | `hr-leave-policy:v2:chunk-03` | 稳定且可追溯的块标识 |
| `source_file_name` | `员工休假制度.pdf` | 面向用户的来源文件名 |
| `source_locator` | `第 3 页 / 年假规则` | 页码、标题或段落位置 |
| `title` | `员工休假制度` | 文档标题 |
| `department` | `hr` | 文档所属部门 |
| `allowed_roles` | `["hr", "admin"]` | 被授权的角色集合 |
| `status` | `active` | `active`、`superseded`、`withdrawn` 或 `deleted` |
| `content_hash` | `sha256:...` | 用于变更识别和重复内容检查 |
| `created_at` | ISO 8601 时间 | 记录首次写入时间 |
| `updated_at` | ISO 8601 时间 | 记录当前版本更新时间 |

### 6.2 ACL 规则

ACL 是检索约束，不是提示词约束。每次 Qdrant 查询都必须同时包含：

- 当前用户对应的角色条件，匹配 `allowed_roles`。
- `status = active` 条件。
- 必要时的部门或数据范围条件。

任何不满足 ACL 的文档块都不得进入 Top-K 候选，也不得作为 DeepSeek 上下文、引用条目或错误信息的一部分。未授权用户不能通过文档标题、文件名、版本号、引用 ID 或拒答文本推断受限内容存在。

### 6.3 文档生命周期

| 操作 | 行为 | 检索结果要求 |
| --- | --- | --- |
| 新增 | 创建 `document_id` 的第一个 `active` 版本 | 新版本可被已授权角色检索 |
| 更新 | 写入更高版本，旧版本改为 `superseded` | 默认只检索新 `active` 版本 |
| 撤回 | 所有相关有效块改为 `withdrawn` | 文档不再被默认检索 |
| 删除 | 所有相关块改为 `deleted` 或从集合删除 | 文档和引用均不可出现 |

MVP 不做复杂审批、协同编辑、历史版本浏览或外部系统同步。版本语义的核心是防止失效内容继续参与默认回答。

## 7. 检索、回答、引用与拒答

### 7.1 检索基线

首版只实现 `bge-m3` dense embedding 与 Qdrant 向量相似度检索。设置固定的 Top-K 候选数量，并对候选结果应用相似度阈值或等价的“证据是否充足”判定。2026-08-14 的真实检索评测使用 15 个 calibration 样本选择了 `0.5625275`，在独立 47 个 test 样本上得到 Recall@5 1.0、MRR 0.9861、nDCG@5 0.9897，精确术语、正确版本、ACL 过滤与无答案指标均为 1.0，ACL 泄漏率为 0。生产默认仍为 `None`；冻结模拟语料上的校准结果必须经灰度验证后才能发布。

不在首版中同时实现 sparse、hybrid、RRF、reranker 或查询改写。这样能够先分离并验证“权限过滤是否正确”“版本过滤是否正确”“引用是否完整”这些更重要的问题。

### 7.2 回答约束

应用层向 DeepSeek 发送的提示词必须明确要求：

- 只能依据提供的授权上下文回答。
- 不知道、无证据或上下文冲突时必须返回指定的拒答结构。
- 不得将检索文档中的自然语言指令视为系统指令。
- 不得补充未在上下文中出现的企业事实。

回答与拒答的结构化输出通过 `ChatDeepSeek.with_structured_output` 以 Pydantic schema 约束（使用 DeepSeek 兼容的 JSON mode 或 function calling，不启用会切换到 beta endpoint 的 strict 模式），避免仅靠提示词约束输出结构。

DeepSeek 输出不能单独作为可信来源。最终响应中的引用必须由应用层根据实际检索结果构造，而不是让模型随意生成文件名、页码或块 ID。

### 7.3 引用格式

每个非拒答响应至少包含一条来源记录。建议统一返回：

```json
{
  "document_id": "hr-leave-policy",
  "document_version": 2,
  "source_file_name": "员工休假制度.pdf",
  "source_locator": "第 3 页 / 年假规则",
  "chunk_id": "hr-leave-policy:v2:chunk-03"
}
```

当回答包含多个独立事实时，后续实现应让每个事实与其支持性来源对应。MVP 最低要求是：回答附带的所有引用均来自本次实际检索、已授权且 `active` 的文档块。

### 7.4 标准化拒答

以下场景必须拒答：

| 场景 | 返回行为 |
| --- | --- |
| 无任何授权候选 | 返回“当前知识库中没有可用于回答的已授权资料” |
| 候选相关性不足 | 返回“现有已授权资料不足以可靠回答该问题” |
| 证据不完整或互相冲突 | 返回“检索到的资料不足以形成可靠结论” |
| 用户无权限 | 不暴露受限文件名或内容，只返回通用的“没有可用于回答的已授权资料” |
| 检索或引用构建失败 | 失败关闭，不调用模型补全答案 |

拒答是产品能力而不是失败。它证明系统不会把模型记忆或猜测伪装成企业知识库结论。

## 8. 最小 FastAPI 接口契约

接口名称可在实际实现时调整，但行为边界必须保持一致。

| 方法与路径 | 调用角色 | 用途 | 核心结果 |
| --- | --- | --- |
| `POST /documents` | `admin` | 上传新文档 | 返回 `document_id`、版本、处理状态 |
| `PUT /documents/{document_id}` | `admin` | 上传新版本 | 新版本激活，旧版本失效 |
| `DELETE /documents/{document_id}` | `admin` | 删除或撤回文档 | 后续默认检索不得返回相关块 |
| `POST /questions` | 所有模拟角色 | 原有提问并检索 | 返回回答或拒答及引用，契约保持不变 |
| `POST /policy-comparisons` | 所有模拟角色 | 比较两个明确策略版本 | 返回确定性章节 diff、授权引用或通用拒答 |
| `POST /policy-rollout-tasks` | 令牌绑定角色 | 创建策略落地规划任务 | 返回计划、澄清或失败状态；只读、不执行 Todo |
| `GET /policy-rollout-tasks/{task_id}` | 任务绑定角色 | 读取已持久化任务 | 返回任务与计划版本；当前引用失效或失权时不可访问 |
| `POST /policy-rollout-tasks/{task_id}/continue` | 任务绑定角色 | 提交澄清答案并继续规划 | CAS 成功后追加不可变计划版本 |
| `GET /documents/{document_id}/versions` | `admin` | 查看版本状态 | 仅用于调试与演示版本生命周期 |
| `GET /health` | 任意 | 健康检查 | Qdrant 正常返回 200；不可达返回 503 |

`POST /questions` 的请求必须显式传入用于演示的用户身份或角色。首版的角色身份可以是固定测试身份，不把模拟身份认证误称为真实企业认证。

`POST /policy-comparisons` 请求体为 `document_id`、正整数 `from_version`、正整数 `to_version` 和 `role`，同时要求服务端配置的 `X-Demo-Role-Token` 与请求角色匹配。服务端绑定文档、角色、集合、生命周期与 ACL；Agent 只能调用授权版本证据读取和确定性比较两个工具。仅允许 `active`/`superseded`，撤回、删除、缺失、未授权、令牌不匹配、反向/相同版本和模型/工具错误全部失败关闭，拒答不回显文档、版本、文件、块或正文。该比较路径没有持久化 Agent run 或 checkpointer。

策略落地规划的创建请求包含 `goal` 和 `role`，角色由 `X-Demo-Role-Token` 服务端解析，并要求 body 中角色一致；读取和继续请求不从 body 接收角色。规划模型通过 `ChatDeepSeek.with_structured_output(PlannerDecision)` 输出 `search`、`clarify`、`propose` 或 `refuse`，由应用层执行绑定角色的 `Retriever`，并限制最多 6 次模型决策、3 次检索和 2 次修复。计划依赖关系、候选块归属和 Citation 由应用层校验与构造；澄清答案项最多 20 个，每项最多 2000 字符。SQLite 持久化任务业务状态和 append-only 计划版本，但不保存证据正文或模型原始输出；GET 和 continue 都会重新校验已保存引用当前仍为 `active` 且对角色授权。continue 使用 CAS，只有一个请求能领取澄清状态；v1 不执行 Todo、不接外部系统，也不提供 Agent graph 的 durable execution。

## 9. 测试优先的实施阶段

每个阶段都应先写失败测试，再实现最小代码使测试通过；不要等所有功能完成才补测试。

### 阶段 0：环境可用性

目标：确认 Docker Desktop Engine、Docker Compose、Qdrant 容器、Conda Python 和 DeepSeek API 配置都可用。当前 Docker Engine、Docker Compose、Qdrant、Conda 环境和项目测试链路已验证。2026-08-14 已在用户明确授权的冻结 fixture 上完成真实 DeepSeek Planner 4×1 与 11×3：分别 4/4、33/33 通过，安全、质量和成本门均通过；复现仍依赖本地密钥、网络和 Provider 可用性。

完成条件：Docker 能启动 Qdrant；应用能够从本机连接 Qdrant；DeepSeek 密钥只从环境变量读取。

### 阶段 1：项目骨架与固定测试资产

目标：建立最小目录结构、固定模拟文档、用户角色和测试数据。

完成条件：测试资产不包含真实企业敏感数据；每个样本的角色、预期文档与预期结果均明确记录。

### 阶段 2：文档摄取与版本生命周期

目标：实现解析、切块、哈希、版本元数据与状态变更。

优先测试：新增文档后可查询元数据；更新后旧版本状态为 `superseded`；删除后相关块不再有效。

### 阶段 3：带 ACL 的 dense 检索

目标：将向量与 payload 写入 Qdrant，并在查询时强制 ACL 与 `active` 状态过滤。

优先测试：授权角色召回期望 `document_id`；未授权角色候选中不包含受限 `chunk_id`；旧版本和删除文档不被召回。

### 阶段 4：受约束回答、引用与拒答

目标：连接 DeepSeek API，构造证据上下文，返回稳定的响应结构。

优先测试：对 DeepSeek 调用使用 mock 或固定响应；引用只能来自实际候选；无证据时返回拒答；拒答不泄露受限元数据。

### 阶段 5：冻结回归与演示准备

目标：将核心测试在干净环境中重复运行，并准备固定演示脚本。

完成条件：所有冻结测试通过；演示不依赖临时手工修改数据；README 中列出环境变量但不包含真实密钥。

## 10. 冻结 `pytest` 评测设计

### 10.1 固定语料

使用少量自建或公开可分发的模拟文档，例如：

- `全员差旅报销制度.md`：允许 `employee`、`engineer`、`hr`、`admin`。
- `研发发布流程.md`：允许 `engineer`、`admin`。
- `员工休假制度-v1.pdf`：允许 `hr`、`admin`。
- `员工休假制度-v2.pdf`：更新后替代 v1。

### 10.2 必测用例

| 用例 | 输入 | 期望断言 |
| --- | --- | --- |
| 授权检索 | `engineer` 查询发布流程 | 返回研发流程的有效块与有效引用 |
| 越权隔离 | `employee` 查询 HR 休假制度 | 候选和最终响应均不含 HR 文档任何标识 |
| 版本替换 | 激活 v2 后查询休假规则 | 只引用 v2，不引用 v1 |
| 删除隔离 | 删除文档后重复查询 | 返回拒答，不返回已删除块 |
| 无证据拒答 | 询问语料不存在的问题 | 返回标准化拒答 |
| 证据不足拒答 | 只检索到不完整规则 | 返回标准化拒答 |
| 引用完整性 | 任意非拒答回答 | 每条引用指向实际、授权且有效的块 |
| 上传校验 | 不支持类型或超大小文件 | API 拒绝且不写入 Qdrant |

### 10.3 测试边界

确定性的摄取、状态、ACL 和引用验证必须完全脱离真实 LLM 输出。大多数单元测试应 mock DeepSeek API；真实模型只用于显式授权、调用预算受控且与默认回归分离的验收。这样可避免模型随机性、网络问题和 API 成本破坏回归稳定性。2026-08-14 最新冻结回归为 202 项通过；真实 Planner 11×3 共 33 次均通过终态、任务结果、计划约束、澄清利用、Todo、依赖、引用、安全和成本门。精确轨迹匹配仅保留为诊断指标，外部 Provider 内部重试及部分 Token 元数据仍不可观测。

策略比较另有 `tests/test_policy_compare.py` 聚焦测试：授权比较、added/modified 章节、越权、withdrawn/deleted/missing、非法/反向/相同版本、工具绑定、步数/工具次数、模型和工具失败、引用完整性、只读约束及端点契约。基准脚本 `scripts/benchmark_policy_comparisons.py` 读取 manifest 的 `policy_comparisons` 和固定语料 fixture，不依赖运行态 Qdrant 数据；它将实际 case pass/fail、ACL 泄漏、版本不匹配、引用完整性、护栏违规、拒答错误计数与耗时写入 `data/local/benchmarks/`；报告结构由 `schemas/policy-comparison-report.schema.json` 固定，不预填结果。

在没有实际跑出结果之前，不写入任何准确率、召回率、延迟、幻觉率或成本数字。后续若进行评测，应将结果与数据集版本、配置和失败样例一起记录。

## 11. 安全与数据边界

### 11.1 必须遵守的最低线

- DeepSeek API 密钥只通过环境变量或本地 `.env` 加载，不能提交到 Git。
- `.env`、本地 Qdrant 数据卷和真实私密文件不进入仓库。
- 文档上传限制允许类型与文件大小；解析失败有明确错误响应。
- 检索文档、用户问题和模型输出均视为不可信数据。
- 文档中的“忽略此前指令”“泄露系统提示词”等文本只能作为证据内容，不能获得工具执行或权限提升能力。
- 应用查询路径只有检索权限；写入向量库的能力只应由摄取路径使用。
- 日志优先记录请求标识、角色、检索块 ID、版本和状态；避免默认记录完整敏感原文。
- 策略比较的本地演示令牌只用于替代真实身份上下文；生产环境必须接入真实认证与授权系统，不能把请求体 `role` 当作身份凭据。
- 比较运行日志采用标准库 JSON 字段白名单：`run_id`、operation/stage、role、document_hash、step/tool、counts、duration_ms、status、error_code；不记录 API key、prompt、正文或完整模型输出。

### 11.2 不承诺的安全范围

MVP 没有真实企业身份系统、审计合规认证、加密密钥管理、多租户隔离或外部攻击测试。因此应在文档和面试中明确它是安全意识与工程边界的原型，而非上线系统。

## 12. 本地演示脚本

演示必须使用固定语料和固定身份，按照以下顺序进行：

1. 启动 Docker Desktop，并用 Docker Compose 启动 Qdrant。
2. 在本机 Conda 环境中启动 FastAPI。
3. 以管理员身份导入公开制度、研发制度和 HR 制度，显示生成的版本与元数据。
4. 以 `engineer` 身份查询研发发布流程，展示回答及“文档、版本、来源位置、块 ID”引用。
5. 以 `employee` 身份询问 HR 休假制度，展示通用拒答，证明不出现 HR 文件名或片段。
6. 上传休假制度 v2，查询后展示仅引用 v2；v1 不再作为有效来源。
7. 删除或撤回 v2，再次查询，展示无证据拒答。
8. 运行冻结 `pytest` 测试，说明它覆盖权限、版本、删除、引用和拒答。

演示重点不是回答写得多漂亮，而是让观察者看到：同一问题的可见证据受角色限制，所有成功结论都有可追溯来源，失效文档不会被继续使用。

## 13. 面试叙事与关键取舍

### 13.1 推荐介绍方式

“我实现的是一个面向内部制度和技术文档的 RAG 问答原型。它的重点不是让模型尽可能多说，而是在检索阶段按角色过滤可见文档，并要求每个回答附带有效版本的来源。对于没有授权证据的问题，系统明确拒答。文档更新和删除会改变有效版本，冻结测试保证这些行为不会回归。”

求职表述应保持诚实：可以说“实现了一个有界的策略版本比较 Agent 工作流”，具体强调服务端 ACL、固定只读工具、确定性 diff、应用层引用、结构化观测和 manifest 基准；不能表述为通用自治 Agent、生产级审计平台或真实企业身份系统。

### 13.2 可解释的技术取舍

| 选择 | 原因 |
| --- | --- |
| 先做 dense 检索 | 先验证权限、版本、引用和拒答等核心正确性，降低问题定位难度 |
| Qdrant 放入 Docker | 数据库是独立服务，容器化启动可复现；Python 应用仍在本地 Conda 调试 |
| 使用 DeepSeek API | 已有可用服务，避免本地模型下载、GPU 与推理吞吐成为项目阻塞项 |
| 生成层采用 ChatDeepSeek 组件级封装 | 获得 `with_structured_output` 的结构化拒答与统一消息抽象；检索层评估后保持原生 `qdrant-client`，因为 ACL filter 是第一公民，框架在此无增量收益 |
| ACL 在检索前执行 | 防止未授权文本进入候选、上下文、引用或模型提示词 |
| 使用 mock 做多数测试 | 将确定性工程行为与随机的模型生成行为分开验证 |
| 只在策略比较中采用有界 Agent | 原有 `/questions` 保持线性和稳定；比较路径的工具、版本、ACL、步数和引用均由应用层固定，避免开放自治攻击面 |

### 13.3 可能被问到的问题

| 问题 | 回答要点 |
| --- | --- |
| 为什么不直接让模型判断权限？ | 模型不是访问控制系统；权限必须约束检索候选，避免敏感内容先进入上下文。 |
| 为什么要引用？ | 引用让回答可审查、可定位；它也能帮助发现过期或不支持结论的内容。 |
| 为什么需要拒答？ | 检索不到或证据不足时继续回答会把猜测伪装为企业事实。 |
| 为什么不把整个项目改成 `create_agent`、LangGraph 或多 Agent？ | 原有 `/questions` 是稳定的线性 RAG，不需要开放式工具循环；新增的 `/policy-comparisons` 单独使用 `create_agent`，但只允许两个服务端绑定的只读工具，且没有持久 Agent run 或 checkpointer。`/policy-rollout-tasks` 另有持久化业务任务和计划版本，但不持久化 Agent 图执行；多 Agent 和 Memory 仍未引入，ACL、diff、引用和拒答仍由应用层控制。 |
| 为什么不做混合检索和 reranker？ | 它们是质量优化手段，不是 ACL、版本、引用与拒答的前提；应由冻结评测证明需要后再加入。 |
| 如何避免旧文档继续被引用？ | 向量 payload 有版本与状态；更新后旧版本标为失效，检索始终限定 `status=active`。 |

## 14. 二元验收清单

以下所有项目均满足，才将 MVP 标记为完成：

- [x] Docker Desktop Engine 可正常运行，并已成功运行 `hello-world` 验证容器。
- [x] Docker Compose v2 命令可正常运行（2026-08-07 验证：v5.3.1）。
- [x] Qdrant 作为 Docker 容器运行，且 Dashboard 可通过本机地址访问。
- [x] 后续 FastAPI 应用能通过本机地址访问 Qdrant(阶段 2 起 API 集成测试连真实容器验证)。
- [x] FastAPI、Docling、embedding 与测试都在独立 Conda 环境运行(`rag-kb`,Python 3.13.14)。
- [x] 固定语料中每个文档块都有文档 ID、版本、来源位置、角色 ACL、状态和内容哈希。
- [x] 检索请求在 Qdrant 中同时过滤用户角色与 `status=active`(2026-08-07 阶段 3 集成测试验证)。
- [x] 未授权用户的候选、回答、引用和错误信息均不含受限文档内容或标识(泄露专项测试验证)。
- [x] 每个非拒答回答都带有来自实际检索有效块的引用(模型只能选择,应用层构造)。
- [x] 无证据、无权限和证据不足时均返回标准化拒答。
- [x] 新版本激活后默认检索不返回被替代版本。
- [x] 删除或撤回文档后默认检索不返回相关内容。
- [x] 冻结 `pytest` 测试覆盖授权、越权、版本更新、删除、引用、拒答及策略版本比较安全边界。
- [x] 测试默认不依赖真实 DeepSeek API 返回值(全部经 FakeAnswerModel)。
- [x] 2026-08-14 最新冻结回归 270 项通过；真实 Planner 4×1 与 11×3、真实 `bge-m3 + Qdrant` 独立检索测试均通过各自验收门，结果只适用于冻结模拟语料。
- [x] API 密钥、真实私密文档和本地向量数据均未提交到 Git。
- [x] 演示可按固定步骤重复完成（`scripts/demo.py`，开头自动重置 collection，2026-08-07 全流程实测通过），且没有临时手工修改数据。

## 15. 后续路线图与升级触发条件

只有在 MVP 验收清单全部通过后，才按明确需求升级：

| 观察到的问题或需求 | 可评估的后续能力 |
| --- | --- |
| 编号、术语和关键词查询在冻结集上表现不佳 | sparse 或 hybrid 检索 |
| 候选 Top-K 经常相关性不足 | reranker |
| 需要观察真实调用链、延迟与用户反馈 | Langfuse 或等价追踪方案 |
| 需要多轮追问澄清、会话状态持久化或人工审批节点 | LangGraph 状态图（从最小图开始，引入 checkpoint 前先评估其攻击面） |
| 需要切换多家 LLM 或 embedding 提供方，或复用标准 Runnable 生态 | 扩大 LangChain 组件级接入（评估检索层 `langchain-qdrant`；ACL filter 仍由应用层手写构造，注意其 payload 嵌套于 `metadata` 键下，filter key 须带 `metadata.` 前缀） |
| 需要系统化评估生成和检索质量 | Ragas 与人工复核结合 |
| 文档数量、上传并发或任务耗时增长 | PostgreSQL、Redis、MinIO 与异步摄取 |
| 需要真实组织身份与跨系统权限 | SSO、细粒度授权与审计设计 |
| 需要本地模型、隐私隔离或高吞吐 | GPU、vLLM 或经过批准的本地推理服务 |

在每次升级前，应新增对应的冻结测试与失败样例，而不是仅通过一次主观演示判断“效果更好”。

## 16. 下一次讨论的起点

MVP 全部阶段已完成（2026-08-07），并新增一个安全收敛的策略版本比较 Agent 工作流。固定演示脚本 `scripts/demo.py` 全流程实测通过（8 步含泄露与版本断言）；启动脚本 `scripts/start_api.ps1` 固化 `PYTHONUTF8=1`（中文 Windows 下 Docling 读取模型文件的 GBK 编码问题，2026-08-07 实测排障）。策略比较的基准结果只由命令实际运行后写入被忽略的 `data/local/benchmarks/`，不在文档中预填数字。后续方向按第 15 节触发器评估：补充编号/术语类冻结用例检验 dense 检索盲区、真实演示录像与面试讲稿；不把该工作流表述为通用自治 Agent 或生产级身份系统。
### 5.5 已批准计划的确定性本地草稿（v2）

v2 在 append-only 计划版本之外增加独立审批与执行账本。审批摘要由服务端对任务目标、持久化澄清、完整计划版本 JSON 和当前授权引用的内容哈希作确定性 SHA-256；执行前必须再次校验摘要、角色、引用状态与来源元数据。唯一动作 `create_local_draft_work_item` 只用 SQLite 幂等键和确定性文件名创建本地 JSON 草稿，并可在写入后进程中断时对账恢复；不覆盖既有文件，状态不确定即失败关闭为 UNKNOWN。

该“执行”不执行 Todo，也不包含通知、部署、账户变更、支付、外部 API、worker、队列或多智能体。SQLite 不保存证据正文、提示词、令牌或模型原始输出；草稿仅保存任务、版本、Todo、引用、审批摘要和创建时间。
