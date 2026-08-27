# M1 批次 3 遗留缺陷修复方案

> 创建日期：2026-08-27
> 状态：F1-F9 已实施；2026-08-27 验收复核发现 F3/F4 残缺，二次修复方案已确认待实施（见文末"二次修复"节）
> 依据：`M1_FIX_PLAN.md`（§3.5-3.7、§5.7-5.8、§9.8-9.9）
> 范围：只修复批次 3 评审发现的 P1/P2 缺陷，不新增 schema、不修改 0012、不进入批次 4/M2

## 背景

批次 3（正文生命周期闭环）核心结构已完成并通过 `425 passed, 1 skipped` 全量验证，但静态评审发现 3 个 P1 与若干 P2 缺陷。其中 P1 会破坏“到期即不可读”“最终绑定身份门”“孤儿最终收敛”三项验收不变量，因此**尚不满足“完成并验收通过”**，需先按本方案修复。

## 修复项总览

| ID | 问题 | 定级 | 修复策略 | 新增依赖/迁移 |
|---|---|---|---|---|
| F1 | `finish_task()` 结果绑定无同事务身份复核 | P1 | 锁内调用 `current_task_auth()` | 无 |
| F2 | `TaskResultService.read()` 到期/GC TOCTOU | P1 | 返回前锁行二次复核 | 无 |
| F3 | 孤儿扫描 cursor/重试仅内存，重启后推进丢失 | P1 | 枚举期排除已引用对象 + 有界扫描 | 无 |
| F4 | `iter_ref_entries` 全量收集排序 | P2 | 有界堆 + 扫描预算 + exhausted 标志 | 无 |
| F5 | `EvidenceLedgerConflict` 一律 409 | P2 | 授权类/完整性类分流 | 无 |
| F6 | grace/TTL 校验与工具超时契约脱节 | P2 | 以 `ToolDefinition` 上限为事实源 | 无 |
| F7 | `create_task` 默认 deadline 用进程时钟 | P2 | 事务内改用数据库时钟 | 无 |
| F8 | Worker 退出时 capability 注销过晚 | P2 | 调整 close 顺序 | 无 |
| F9 | PostgreSQL retention 验收矩阵缺项 | P2 | 补 3 组隔离 PG 用例 | 无 |

全部修复不新增 schema、不修改 0012、不进入批次 4/M2。

---

## F1：最终结果绑定加同事务身份门（P1）

### 根因

`task_worker.py:414` 写结果对象后，`finish_task()`（`task_repository.py:691-742`）只复核租约与 deadline；`_settle()` 在对象写入前 resolve 的身份（`task_worker.py:387`）与最终提交之间存在身份变更窗口，违反计划 §3.7 #2。

### 方案

`finish_task()` 内、持有任务行锁时，对 `is_result_submission` 路径追加：

```python
# task_repository.py finish_task() 内，deadline 转换之后、写行之前：
if is_result_submission:  # deadline 已转 failed 的不再需要身份门
    try:
        current_task_auth(
            work.session,
            task=row,
            auth_context_ref=lease.state.auth_context_ref,
            request_id=trace_id,
        )
    except TaskExecutionContextError as exc:
        raise TaskRepositoryError("task_auth_context_changed") from exc
```

### 关键约定

- 导入 `task_execution.py` 的 `current_task_auth`；已确认无循环依赖（`task_execution` 不导入 `task_repository`）。
- **判定顺序**：租约校验 → deadline 转换 → 身份门。deadline 已过期时直接收敛 `failed/task_deadline_expired`，不必再验身份（更可诊断，且让既有 deadline 测试不依赖主体指纹）。
- **锁序**：task row → principal → group_memberships，与工具执行门完全一致；`DatabaseTaskAuthContextPort.resolve` 不持任务锁，无反向锁序，无死锁面。
- **Worker 侧无需改**：`task_auth_context_changed` 走现有 `_fail_safely`（FAILED 收敛）+ `_delete_unbound_task_result` 补偿，与批次 1“身份失效且持租约→失败关闭”语义一致。

### 测试

- 新增 `test_task_worker_recovery.py`：用 `PreemptingStore` 式 hook 在 `save_task_result` 后改 `PrincipalRow.roles` → 断言 outcome `failed/task_auth_context_changed`、任务无 `result_ref`、对象已补偿删除。
- **破坏性审计**：`test_task_repository.py:38` fixture 的 `PrincipalRow(attributes={})` 与 `auth()` 指纹不匹配——`test_finished_research_task_requires_immutable_result...`、`test_finished_research_task_uses_injected_sqlite_clock...` 等直接提交 SUCCEEDED 的用例需把 fixture 改为持久化 `auth().attributes`（对齐 `test_tool_lease_fencing.py` fixture 做法）。`test_postgres_task_repository.py` 同步审计。

---

## F2：结果读取返回前二次复核（P1）

### 根因

`task_result_service.py:45-77` 只在读对象前检查一次；读取、citation 复核到返回之间，到期/GC 可并发写入，已读正文仍会返回，违反 §3.5 #3 与 §9.8 #3。

### 方案

镜像 `tool_execution_service.py:341-377` 的双重门，在 `read()` 返回前追加锁内终检：

```python
# citations 校验通过后：
with self._database.sessions() as session:
    row = session.scalar(
        select(AgentTaskRow).where(
            AgentTaskRow.tenant_id == auth.tenant_id,
            AgentTaskRow.task_id == state.task_id,
            AgentTaskRow.user_id == auth.user_id,
        ).with_for_update()
    )
    now = self._result_now()
    if row is None or TaskStatus(row.status) not in {SUCCEEDED, REFUSED}:
        raise TaskResultError("task_result_unavailable")
    if row.result_purged_at is not None or (
        (expires := _aware(row.content_expires_at)) is not None and expires <= now
    ):
        raise TaskResultError("task_result_expired")
    if row.result_ref != state.result_ref or row.result_hash != state.result_hash:
        raise TaskResultError("task_result_binding_invalid")
```

### 关键约定

- `_result_now()` 已区分 PostgreSQL `clock_timestamp()` / SQLite 注入时钟，直接复用。
- 复核失败映射：expired → 410；binding_invalid → 500；unavailable → 409，与现有 router 映射（`tasks.py:119-124`）零改动兼容。
- API 读路径短锁单行，持有时间是一次比较，无重入风险。

### 测试

`test_task_agent.py`（真实 SQLite + 伪造 ResultStore）：
- `read_task_result` 内先 UPDATE `content_expires_at` 为过去 / 写 `result_purged_at` 再返回正文 → 断言 `task_result_expired`（对象已读但不可交付）。
- 同法篡改 `result_ref` → 断言 `task_result_binding_invalid`。

---

## F3 + F4：孤儿扫描重启推进 + 有界枚举（P1/P2，合并实施）

> 2026-08-27 复核：按本节初版实施的代码存在推进性残缺（首窗全引用时游标停滞），以文末"二次修复：F3/F4 残缺"节为准。

### 根因

- cursor/重试队列仅在内存（`task_retention_service.py:76-88`），重启后从最旧对象重扫；最旧页全是持久引用对象时（已终态、retention 未到期），进度依赖进程存活时长。
- `local.py:147-189` 无条件遍历整个命名空间再排序切片，`limit` 不约束遍历成本。

### 方案（不改 schema，两个文件联动）

#### 1. `LocalObjectStore.iter_ref_entries()` 签名扩展

```python
def iter_ref_entries(
    self, prefix, *, older_than, limit,
    after=None,
    exclude: frozenset[str] = frozenset(),
    max_scanned: int = 10_000,
) -> tuple[tuple[float, str], ...], bool:  # (entries, exhausted)
```

- 收集期跳过 `exclude` 中的 ref；维护容量为 `limit` 的最小堆（`(mtime, key)`），只保留最小者，内存 O(limit)。
- 计数已检查条目，达 `max_scanned` 即停止并返回 `exhausted=False`；自然扫尽返回 `exhausted=True`。
- `iter_refs()` 保持旧返回类型，内部丢弃 exhausted。

#### 2. `purge_orphaned_result_objects()` 改为“先查库、再枚举”

```python
with self._database.sessions() as session:
    exclude = self._referenced_refs(session) | self._running_tool_refs(session)
# 枚举时传 exclude；task-results 的 running 任务保护仍走扫描后校验
# （其文件名为随机 uuid，无法从 DB 反推，但 running 受 deadline 约束，是暂时态）
```

- `_referenced_refs`：`SELECT result_ref FROM tool_calls/agent_tasks WHERE result_ref IS NOT NULL`（全量字符串集，M1 量级可接受；升级路径见下方权衡）。
- cursor 重置条件收紧为 `exhausted and not scanned`；预算耗尽且零结果时保持 cursor 原值，下轮续扫。
- 重启后即使 cursor 归零，已引用对象在枚举期被跳过，quota 直接落在真孤儿上——**推进不再依赖进程存活时长**；重试队列随重启丢失无害（失败对象未被引用，下轮全 pass 必然重扫到）。

### 测试

- 枚举单测：exclude 命中的对象不占 `limit`；`max_scanned` 不足时 `exhausted=False` 且 cursor 不重置。
- 重启模拟：新建 service（cursor 全空）+ 大量已引用对象 + 1 个真孤儿 → 首轮 `limit=1` 即删真孤儿。
- 既有 8 个 orphan 测试回归（部分断言需适配新返回形状）。

### 权衡点 A（已确认）

放弃“持久化 cursor（DB 表或对象根目录下的 JSON 文件）”方案——引入新存储面与 0012 冻结冲突，而“枚举期排除”以更小 diff 达到同等收敛保证。若单命名空间对象量到 10⁵+ 且引用集巨大，再改为持久化游标（届时建议随批次 4 之后的迁移做表）。

---

## F5：EvidenceLedgerConflict 分类（P2）

### 根因

`task_result_service.py:73-74` 把所有冲突归 `task_result_evidence_unavailable` → 409；`duplicate_evidence_id`、`evidence_ledger_binding_invalid` 等完整性损坏被误报为客户端冲突，违反 §5.7“完整性损坏按服务端错误处理”。

### 方案

在 `task_result_service.py` 增加码集分流：

```python
_EVIDENCE_AUTH_CODES = frozenset({"evidence_authorization_stale"})
# except EvidenceLedgerConflict as exc:
#   code = exc.args[0] if exc.args else ""
#   授权失效 → task_result_evidence_unavailable (409)
#   其余（duplicate/binding/immutable/insert_not_visible）→ task_result_binding_invalid (500)
```

### 测试

`test_v1_tasks_api.py` 补两断言：duplicate → 500；authorization_stale → 409。

---

## F6：保留窗口校验绑定工具契约（P2）

### 根因

`config.py:103-109` 用 `DEFAULT_TOOL_TIMEOUT_SECONDS`(30s) 比较，而 `ToolDefinition.timeout_seconds` 上限 300s（`domain/tools.py:38`）；工具超时策略变更后校验静默失效。

### 方案

- `domain/tools.py` 新增 `TOOL_TIMEOUT_CEILING_SECONDS = 300`，`Field(le=...)` 改引该常量（单一事实源）。
- `config.py` 校验改为：`orphan_grace > CEILING + 60`（绑定窗口余量）、`tool_result_ttl > CEILING`。当前默认 3600/604800 均通过。
- 同步更新 `test_retention_settings_require_safe_object_windows` 的边界值（30/31 不再是边界）。

### 权衡点 B（已确认）

grace 下限从 31s 提高到 361s，会收紧演示环境想快速触发孤儿清理的配置空间（测试直接注入 `orphan_grace_seconds=60` 不受影响，只影响 Settings 层）。可接受则按 361 执行；若演示环境需要更小值，可把余量常量单独暴露为配置项。

---

## F7：create_task 默认 deadline 改用数据库时钟（P2）

### 根因

`task_repository.py:119-122` 在事务外用 `self._now()`（API 进程时钟）生成并校验默认 deadline，与后续全部 `clock_timestamp()` 判定不同源；主机钟偏或创建事务阻塞时“创建即过期/超时仍可执行”。

### 方案

把 deadline 计算与校验移入 `UnitOfWork` 内：

```python
with UnitOfWork(self._database) as work:
    now = self._lease_now(work.session)
    if deadline is None:
        deadline = now + timedelta(seconds=configured.max_runtime_seconds)
    if deadline <= now:
        raise TaskRepositoryError("task_deadline_expired")
    ...
```

SQLite 注入时钟行为不变（现有单测全兼容）；PostgreSQL 改为 `clock_timestamp()` 派生。`test_postgres_task_repository.py` 补一条：service `_now` 置于 2100 年时 deadline 仍取库时钟。

---

## F8：Worker 退出顺序（P2）

`worker.py:438-444` finally 改为先 `task_worker.close()`（注销 capability + 关工具/checkpointer 连接）再 `task_retention.close()`，消除“主循环已停但 capability 仍有效、API 却返回 202”的窗口。retention 持独立连接，先关 worker 无影响；`test_worker_runtime.py` 只断言 close 次数不断言顺序，无需改动。

---

## F9：PostgreSQL retention 验收补齐（P2）

`test_postgres_retention.py` 沿用现有 fixture 模式补三组：

1. 删除失败（monkeypatch `delete_exact` 抛 OSError）→ ref/marker 保留；恢复后重试成功清 ref。
2. 无结果 failed 终态 → `result_purged_at` 只写一次，二次扫描不命中。
3. orphan：grace 内不删、已引用不删、running 且 deadline 未到不删、grace 后删（验证 `clock_timestamp()` 语义）。

---

## 实施顺序与验证门禁

```
第 1 步  F1 + F2（写边界与读边界，互相独立）
第 2 步  F3 + F4（local.py + retention 联动，一次改完）
第 3 步  F5 + F6 + F7 + F8（小修集合）
第 4 步  F9（PG 用例）+ TASK_PROGRESS.md 批次3表述改写（按最终事实收敛，不再声称"已完成并验证"超出实际的结论）
```

每步完成后立即跑受影响 SQLite 定向；全部完成后：

1. 新临时 PostgreSQL 容器 → 迁移 0012 → `init_agent_checkpointer.py` → 定向 PG 套件。
2. 全量测试。
3. `git diff --check`。
4. 删除临时容器。
5. 更新 `TASK_PROGRESS.md` 批次 3 表述为修复后的实际状态。

**明确不在本轮做、仅记录为已知限制**：retention 与任务执行共线程序（长 Agent invoke 会推迟 GC 周期）——这是 `M1_FIX_PLAN` §5.8 “在现有周期恢复循环中调用” 的既定设计，如需独立线程化应作为单独变更评审。

## 风险与回滚

- 所有改动均为行为收紧（多一道校验、少一类误报），无数据迁移。
- F1 是唯一可能破坏既有测试的项（fixture 主体指纹），已在方案内列出审计清单。
- 每步独立可回滚；不提交、不推送，遵守批次实施纪律。

---

## 二次修复：F3/F4 残缺（2026-08-27 验收复核发现）

> 状态：方案已确认，待实施
> 决策记录：① F3 验收语义按"有界轮次内收敛"签署，不要求字面"首轮即删"；② F4 采用机会式空目录清理；③ 本节写入目标文档，不额外加强 worker close 顺序断言

### 复核发现

1. **P1/F3**：`iter_ref_entries()` 在判断 `exclude` 与 `after` 之前消耗 `max_scanned`（`local.py` 初版 `:176` vs `:195-196`），且 retention 仅在命中候选时推进 cursor（`task_retention_service.py` 初版 `:185-186`）。首窗对象全部已引用时，每轮零结果、`exhausted=False`、cursor 停滞，后续孤儿永久不可达。
   - 最小复现（已证实）：11 个旧对象、`max_scanned=10` —— 首轮返回 10 个，次轮 0 个且 `exhausted=False`，第 11 个不可达；把首窗设为引用对象时，排在后面的孤儿同样不可达。违反本文件"重启后 quota 直接落到真孤儿"的推进不变量。
2. **P2/F4**：`max_scanned` 只统计安全普通文件；空任务目录、无效条目不计数，`delete_exact()` 又不删除父目录。正常 GC 持续积累空目录后，每轮 opendir 数随历史任务数无界增长，阻塞 worker 恢复周期（`worker.py:345-364`）。
3. **测试缺口**：现有用例仅覆盖 8 个引用对象与单预算项，未覆盖"引用数 ≥ max_scanned"边界，故全量通过未能暴露上述问题。

### 为什么不能用"排除项不计预算"修复

已引用对象存活数天不消失；若 exclude 命中不消耗预算，每轮都需全量遍历它们，重新引入 F4 的无界遍历。唯一同时保证"每轮有界"与"跨轮推进"的方式是按物理位置续扫。

### 修复设计（两文件联动，无 schema 变更）

#### 1. `local.py`：`iter_ref_entries()` 改为规范序 + 位置游标

```python
def iter_ref_entries(
    self, prefix, *, older_than, limit,
    after: str | None = None,          # 位置游标：上一轮最后访问的完整 ref
    exclude: frozenset[str] = frozenset(),
    max_scanned: int = 10_000,
) -> tuple[tuple[tuple[float, str], ...], bool, str | None]:  # (entries, exhausted, resume)
```

- **规范序遍历**：任务目录与目录内文件均按名称排序后遍历（仅排序名字，不 stat）。"跳过 ref ≤ after" 因此精确，不依赖 readdir 顺序，跨轮确定性推进。
- **resume 游标** = 本轮最后访问的文件 ref（无论是否命中 cutoff/exclude），截断时返回；自然扫尽返回 `exhausted=True, resume=None`。孤儿被删导致游标对象消失无碍——续扫靠名字比较，不靠对象存在。
- **qualify 过滤不变**：`mtime < cutoff and ref not in exclude` 才入容量为 `limit` 的最小堆（窗口内 oldest-first 启发式保留）。
- **预算只约束 stat 成本**：被排除/未到 cutoff 的条目照样消耗 `max_scanned`（这是有界性来源），但不再阻碍推进。
- `iter_refs()` 旧包装不变（丢弃 resume）。
- ponytail 标注：root 目录名单排序 O(D log D) 是每轮剩余成本；D 到 10⁵+ 且目录不收缩时升级为持久化游标表（原权衡点 A 不变）。

#### 2. `local.py`：机会式清理过期空目录（关 F4 积累面）

- `delete_exact()` 删除文件后 best-effort `os.rmdir(parent)`（`ENOTEMPTY`/`OSError` 静默忽略）。
- 扫描循环遇到"目录 mtime < cutoff 且为空"时同样 best-effort rmdir。
- 并发安全：`put_exclusive()` 临时文件 open 遇 `FileNotFoundError` 时重试一次（重跑 `mkdir(parents, exist_ok=True)`），消除 rmdir 与新写入的竞态窗口。

#### 3. `task_retention_service.py`：cursor 语义切换为位置游标

`_orphan_cursors` 值类型 `tuple[float, str] | None` → `str | None`（opaque marker）：

```python
scanned, exhausted, resume = self._objects.iter_ref_entries(...)
if resume is not None:      # 截断：无论本轮是否有结果，位置前进
    self._orphan_cursors[prefix] = resume
elif exhausted:             # 扫尽：重置开启新 epoch，兼容 mtime 变化
    self._orphan_cursors[prefix] = None
```

- 删除 `scanned[-1]` 作 sort-watermark 的旧逻辑；`exhausted=True` 一律重置。
- 重试队列、overflow 回卷、前缀轮转全部不动。

### 已确认的验收语义（决策 ①）

F3 按"有界轮次内收敛"签署：引用前缀数 R、预算 B 时，孤儿在 ≤ ceil(R/B)+1 轮内被删除，推进不依赖进程存活时长；不要求字面"首轮 `limit=1` 即删真孤儿"。

### 测试计划

新增（`tests/test_task_retention_service.py` + 枚举单测）：

1. **越窗可达**：N 个旧对象、`max_scanned < N`，连续调用数轮后全部枚举到，最终 `exhausted=True`。
2. **重启收敛（验收核心）**：全新 service（cursor 空）+ 引用对象数 > `max_scanned` + 1 个真孤儿排后 → 有限轮数内孤儿被删，引用对象全程保留。
3. **空目录收敛**：大量空目录 + 过期 mtime → 扫描中被 rmdir，后续轮次不再遍历它们。
4. **写入竞态**：`put_exclusive` 遇目录被并发 rmdir 后重试成功（ENOENT 路径）。

适配：8 个既有 orphan 测试与 `test_orphan_scanner_keeps_cursor_when_scan_budget_exhausted`（断言从"cursor 不变"改为"cursor 前进到 resume"）、三元组返回形状、`after` 参数类型。

### 实施顺序与验证门禁

1. `local.py` → `task_retention_service.py` → 测试（一次联动改完）。
2. 定向 SQLite：`test_task_retention_service.py` 全量 + `test_task_agent.py`、`test_task_worker_recovery.py`。
3. 新临时 PostgreSQL 容器 → 迁移 0012 → `init_agent_checkpointer.py` → 全量 `pytest -q` → 删除容器。
4. `git diff --check`。
5. `TASK_PROGRESS.md` 批次 3 表述按最终事实收敛。

### 风险与回滚

改动局部于孤儿扫描器两文件 + 测试；无 schema、无迁移、不触碰 F1/F2/F5-F9 已验证代码。实施前将两文件当前内容备份至临时目录以便手工回滚；每步独立可回滚，不提交、不推送。
