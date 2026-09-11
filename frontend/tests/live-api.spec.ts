import { expect, test, type APIRequestContext, type APIResponse, type Page } from "@playwright/test";
import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import type { Approval, ApprovalDecision, ConversationMessages, DocumentUpload, Identity, StreamAccepted, Task, TaskResult, TurnResponse } from "../src/api/types";

const admin = process.env.RAG_KB_E2E_ADMIN_TOKEN!;
const approver = process.env.RAG_KB_E2E_APPROVER_TOKEN!;
const api = process.env.RAG_KB_E2E_API_URL!;
const runId = process.env.RAG_KB_E2E_RUN_ID!;
const output = process.env.RAG_KB_E2E_OUTPUT_DIR!;
const hash = (value: Buffer) => createHash("sha256").update(value).digest("hex");

// 仅断言状态码及白名单业务字段，认证失败不输出原始请求错误。
async function get<T>(request: APIRequestContext, route: string, token = admin): Promise<T> {
  let response;
  try { response = await request.get(`${api}${route}`, { headers: { Authorization: `Bearer ${token}` }, timeout: 30_000 }); }
  catch { throw new Error(`读取失败：${route}`); }
  expect(response.status(), route).toBe(200);
  return response.json() as Promise<T>;
}
async function requestStatus(request: APIRequestContext, method: "GET" | "POST" | "DELETE", route: string, token = admin, data?: Record<string, unknown>): Promise<APIResponse> {
  try {
    return await request.fetch(`${api}${route}`, { method, headers: { Authorization: `Bearer ${token}` }, data, timeout: 30_000 });
  } catch {
    throw new Error(`请求失败：${method} ${route}`);
  }
}
async function connect(page: Page, token: string): Promise<Identity> {
  await page.goto("/");
  await expect(page.getByLabel("演示令牌")).toBeVisible();
  try { await page.getByLabel("演示令牌").fill(token); }
  catch { throw new Error("无法填写测试身份"); }
  const identity = page.waitForResponse((r) => new URL(r.url()).pathname === "/v1/me");
  await page.getByRole("button", { name: "连接身份" }).click();
  const response = await identity;
  expect(response.status()).toBe(200);
  const current = await response.json() as Identity;
  await expect(page.getByRole("button", { name: "切换", exact: true })).toBeVisible();
  const persisted = await page.evaluate((expected) => ({
    sessionMatches: sessionStorage.getItem("rag-kb.demo-token") === expected,
    localContainsToken: Object.values(localStorage).some((value) => value.includes(expected)),
  }), token);
  expect(persisted).toEqual({ sessionMatches: true, localContainsToken: false });
  return current;
}
async function closeTaskDrawer(page: Page) {
  const drawer = page.getByRole("complementary", { name: "任务详情" });
  if (await drawer.isVisible()) {
    await drawer.getByRole("button", { name: "关闭任务详情" }).click();
    await expect(drawer).toHaveCount(0);
  }
}
async function submit(page: Page, mode: string, content: string) {
  await closeTaskDrawer(page);
  await page.getByRole("button", { name: "新建", exact: true }).click();
  await page.getByRole("combobox", { name: "模式", exact: true }).selectOption(mode);
  await page.getByRole("textbox", { name: "问题", exact: true }).fill(content);
  const response = page.waitForResponse((r) => r.request().method() === "POST" && /\/v1\/conversations\/[^/]+\/messages$/.test(new URL(r.url()).pathname), { timeout: 180_000 });
  await page.getByRole("button", { name: "发送", exact: true }).click();
  return response;
}
async function taskUntil(request: APIRequestContext, taskId: string, target: string, allowWaitingForApproval = false): Promise<Task> {
  const deadline = Date.now() + 660_000;
  while (Date.now() < deadline) {
    const task = await get<Task>(request, `/v1/tasks/${taskId}`);
    if (task.status === target && (target !== "succeeded" || task.result_available)) return task;
    if (["failed", "refused", "cancelled", "waiting_for_user"].includes(task.status)) throw new Error(`任务 ${taskId} 未达到 ${target}，实际状态：${task.status}`);
    if ((task.status === "succeeded" && target !== "succeeded") || (task.status === "waiting_for_approval" && target === "succeeded" && !allowWaitingForApproval)) throw new Error(`任务 ${taskId} 未达到 ${target}，实际状态：${task.status}`);
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
  throw new Error(`任务 ${taskId} 等待 ${target} 超时`);
}

function parseSse(body: string): { event: string; data: unknown }[] {
  return body.split(/\r?\n\r?\n/).flatMap((block) => {
    const event = block.match(/^event:\s*(.+)$/m)?.[1];
    const data = block.split(/\r?\n/).filter((line) => line.startsWith("data:")).map((line) => line.slice(5).trimStart()).join("\n");
    return event ? [{ event, data: data ? JSON.parse(data) : null }] : [];
  });
}

test("真实上传、流式问答、研究、独立审批与产物权限", async ({ page, browser, request }, testInfo) => {
  const documentId = `ui-live-${runId}`;
  const policy = `北辰${runId}专项假制度`;
  const evidence: Record<string, unknown> = { run_id: runId, started_at: new Date().toISOString(), status: "running" };
  const conversationIds = new Set<string>();
  const taskIds = new Set<string>();
  const artifactContentRoutes: string[] = [];
  let uploaded: DocumentUpload | undefined;
  let adminIdentity: Identity | undefined;
  let testFailed = false;
  const approvalContext = await browser.newContext({ baseURL: process.env.RAG_KB_E2E_BASE_URL });
  try {
    await test.step("真实文档标识与摄取", async () => {
      adminIdentity = await connect(page, admin);
      await page.getByRole("link", { name: "文档", exact: true }).click();
      await page.getByRole("button", { name: "上传文档", exact: true }).click();
      const uploadDialog = page.getByRole("dialog", { name: "上传文档" });
      await uploadDialog.locator('input[type="file"]').setInputFiles({ name: `${policy}.md`, mimeType: "text/markdown", buffer: Buffer.from(`# ${policy}\n\n本文件是隔离验收合成材料。${policy}规定正式员工每年享有 17 天带薪专项假。申请必须提前 3 个工作日提交，直属主管审批后生效。\n`) });
      await uploadDialog.getByRole("textbox", { name: /^文档标识/ }).fill(documentId);
      await uploadDialog.getByRole("textbox", { name: "标题", exact: true }).fill(policy);
      await uploadDialog.getByRole("textbox", { name: "部门", exact: true }).fill("qa");
      const response = page.waitForResponse((r) => r.request().method() === "POST" && new URL(r.url()).pathname === "/v1/documents");
      await page.getByRole("button", { name: "提交上传" }).click();
      const accepted = await response;
      expect(accepted.status()).toBe(202);
      uploaded = await accepted.json() as DocumentUpload;
      evidence.upload = uploaded;
      expect(uploaded.document_id).toBe(documentId);
      await expect(page).toHaveURL(new RegExp(`document=${documentId}`));
      const detail = page.locator(".document-detail-panel");
      await expect(detail.getByRole("heading", { name: policy })).toBeVisible();
      const version = detail.locator(".version-row").filter({ has: page.getByText(`V${uploaded.version}`, { exact: true }) });
      await expect(version.getByText("可用于问答", { exact: true })).toBeVisible({ timeout: 300_000 });
      await expect(version.getByText("已完成", { exact: true })).toBeVisible();
    });
    await test.step("流式 done 与正式持久消息一致", async () => {
      const response = await submit(page, "answer", `只根据${policy}，每年有多少天专项假，申请须提前几个工作日？请引用该文件。`);
      expect(response.status()).toBe(200);
      expect(response.headers()["content-type"]).toContain("text/event-stream");
      const events = parseSse(await response.text());
      const accepted = events.filter((event) => event.event === "turn.accepted");
      const done = events.filter((event) => event.event === "done");
      expect(accepted).toHaveLength(1);
      expect(done).toHaveLength(1);
      expect(events.some((event) => event.event === "error")).toBe(false);
      expect(events.findIndex((event) => event.event === "turn.accepted")).toBeLessThan(events.findIndex((event) => event.event === "done"));
      const acceptedPayload = accepted[0].data as StreamAccepted;
      const payload = done[0].data as TurnResponse;
      expect(acceptedPayload.conversation_id).toBe(payload.conversation_id);
      expect(acceptedPayload.turn_id).toBe(payload.turn_id);
      expect(acceptedPayload.user_message.message_id).toBe(payload.user_message.message_id);
      conversationIds.add(payload.conversation_id);
      expect(payload.assistant_message).not.toBeNull();
      const message = payload.assistant_message!;
      expect(message.content).toContain("17");
      expect(message.citations.some((citation) => citation.document_id === uploaded!.document_id)).toBe(true);
      const persisted = await get<ConversationMessages>(request, `/v1/conversations/${payload.conversation_id}/messages`);
      expect(persisted.items.find((item) => item.message_id === message.message_id)).toEqual(message);
      await expect(page.locator(".preview-message")).toHaveCount(0);
      await expect(page.locator(".message-bubble.from-assistant .markdown-content").last()).toContainText("17");
      await expect(page.locator(".source-item").filter({ hasText: `${policy}.md` }).first()).toBeVisible();
      evidence.answer = { conversation_id: payload.conversation_id, turn_id: payload.turn_id, message_id: message.message_id, citations: message.citations, stream_events: events.map((event) => event.event) };
    });
    await test.step("浏览器触发真实研究并查看结果事件", async () => {
      const response = await submit(page, "research", `请研究文档 ${documentId}（${policy}）的当前版本，只核对专项假天数、申请提前的工作日数和审批要求。本任务不涉及历史版本，不调用 list_document_versions 或 compare_document_versions。请先用 search_knowledge 检索该文档，再用 fetch_evidence 复核检索结果中的原文证据；三项均有原文证据后，立即输出有引用的制度摘要并停止。无需澄清，不生成产物。`);
      expect(response.status()).toBe(202);
      const accepted = await response.json() as TurnResponse;
      expect(accepted.task?.task_id).toBeTruthy();
      const taskId = accepted.task!.task_id;
      conversationIds.add(accepted.conversation_id);
      taskIds.add(taskId);
      await taskUntil(request, taskId, "succeeded");
      const result = await get<TaskResult>(request, `/v1/tasks/${taskId}/result`);
      expect(result.answer).toContain("17");
      expect(result.citations.some((citation) => citation.document_id === uploaded!.document_id)).toBe(true);
      const drawer = page.getByRole("complementary", { name: "任务详情" });
      await expect(drawer.locator(".status-succeeded")).toBeVisible();
      await expect(drawer.locator(".result-block")).toContainText("17");
      await expect(drawer.locator(".event-list li").first()).toBeVisible();
      evidence.research = { task_id: taskId, conversation_id: accepted.conversation_id, citations: result.citations };
      await drawer.getByRole("button", { name: "关闭任务详情" }).click();
    });
    await test.step("独立身份审批、真实产物、下载与权限", async () => {
      const response = await submit(page, "controlled_action", `请依据${policy}生成且只生成一份 Markdown 制度摘要报告，包含17天专项假和提前3个工作日申请要求，引用原文。报告名称为${policy}报告，请发起审批后执行，不要请求补充信息。`);
      expect(response.status()).toBe(202);
      const accepted = await response.json() as TurnResponse;
      const taskId = accepted.task!.task_id;
      conversationIds.add(accepted.conversation_id);
      taskIds.add(taskId);
      const waiting = await taskUntil(request, taskId, "waiting_for_approval");
      expect(waiting.result_available).toBe(false);
      const drawer = page.getByRole("complementary", { name: "任务详情" });
      await expect(drawer.locator(".status-waiting_for_approval")).toBeVisible();
      await expect(drawer.locator(".artifact-button")).toHaveCount(0);
      const unavailable = await requestStatus(request, "GET", `/v1/tasks/${taskId}/result`);
      expect(unavailable.status()).toBe(409);
      const pending = await get<Approval[]>(request, "/v1/approvals?status=pending", approver);
      const approvals = pending.filter((approval) => approval.task_id === taskId);
      expect(approvals).toHaveLength(1);
      const approval = approvals[0];
      expect(approval.tool_name).toBe("create_report_artifact");
      const selfDecision = await requestStatus(request, "POST", `/v1/approvals/${approval.approval_id}/decision`, admin, {
        decision: "approve",
        expected_task_state_version: waiting.state_version,
      });
      expect(selfDecision.status()).toBe(403);
      const reviewer = await approvalContext.newPage();
      const approverIdentity = await connect(reviewer, approver);
      expect(adminIdentity?.user_id).toBeTruthy();
      expect(approverIdentity.user_id).not.toBe(adminIdentity!.user_id);
      const listResponse = reviewer.waitForResponse((r) => new URL(r.url()).pathname === "/v1/approvals" && new URL(r.url()).searchParams.get("status") === "pending");
      await reviewer.getByRole("link", { name: "审批", exact: true }).click();
      const uiList = await (await listResponse).json() as Approval[];
      const index = uiList.findIndex((item) => item.approval_id === approval.approval_id);
      expect(index).toBeGreaterThanOrEqual(0);
      await reviewer.locator(".approval-row").nth(index).click();
      await expect(reviewer.locator(".approval-detail-panel")).toContainText(taskId);
      await reviewer.getByRole("button", { name: "批准", exact: true }).click();
      const decision = reviewer.waitForResponse((r) => new URL(r.url()).pathname === `/v1/approvals/${approval.approval_id}/decision` && r.request().method() === "POST");
      await reviewer.getByRole("button", { name: "确认批准", exact: true }).click();
      const decided = await decision;
      expect(decided.status()).toBe(202);
      const approved = await decided.json() as ApprovalDecision;
      expect(approved.resume_command?.command_id).toBeTruthy();
      await taskUntil(request, taskId, "succeeded", true);
      const finalApproval = await get<Approval>(request, `/v1/approvals/${approval.approval_id}`, approver);
      expect(finalApproval.status).toBe("executed");
      expect(finalApproval.approved_by).toBe(approverIdentity.user_id);
      const result = await get<TaskResult>(request, `/v1/tasks/${taskId}/result`);
      expect(result.artifacts).toHaveLength(1);
      const artifact = result.artifacts[0];
      expect(artifact.artifact_type).toBe("report");
      expect(artifact.status).toBe("ready");
      expect(artifact.content_hash).toMatch(/^sha256:[0-9a-f]{64}$/);
      const route = `/v1/tasks/${taskId}/artifacts/${artifact.artifact_id}/content`;
      artifactContentRoutes.push(route);
      await expect(drawer.locator(".status-succeeded")).toBeVisible();
      await drawer.locator(".artifact-button").click();
      await expect(drawer.locator(".artifact-preview pre")).toContainText("17");
      const downloadPromise = page.waitForEvent("download");
      await drawer.getByRole("button", { name: "下载原始内容" }).click();
      const download = await downloadPromise;
      const downloaded = await readFile((await download.path())!);
      let raw;
      try { raw = await request.get(`${api}${route}`, { headers: { Authorization: `Bearer ${admin}` } }); }
      catch { throw new Error("读取产物失败"); }
      expect(raw.status()).toBe(200);
      expect(downloaded.equals(await raw.body())).toBe(true);
      expect(`sha256:${hash(downloaded)}`).toBe(artifact.content_hash);
      let denied;
      try { denied = await request.get(`${api}${route}`, { headers: { Authorization: `Bearer ${approver}` } }); }
      catch { throw new Error("产物权限检查失败"); }
      expect([403, 404]).toContain(denied.status());
      evidence.controlled_action = { task_id: taskId, approval_id: approval.approval_id, resume_command_id: approved.resume_command!.command_id, artifact_id: artifact.artifact_id, content_hash: artifact.content_hash, denied_status: denied.status() };
      await page.screenshot({ path: testInfo.outputPath("completed-artifact.png"), fullPage: true });
      await drawer.getByRole("button", { name: "关闭任务详情" }).click();
      await page.getByRole("button", { name: "切换", exact: true }).click();
      await connect(page, approver);
      await expect(page.locator(".artifact-preview")).toHaveCount(0);
      await expect(page.locator(".message-bubble")).toHaveCount(0);
      await page.getByRole("link", { name: "审批", exact: true }).click();
      await expect(page.getByRole("heading", { name: "审批", exact: true })).toBeVisible();
      evidence.identity_switch = true;
    });
    evidence.status = "passed";
  } catch (error) {
    testFailed = true;
    evidence.status = "failed";
    throw error;
  } finally {
    const cleanupErrors: string[] = [];
    const taskCleanup: Record<string, unknown> = {};
    const conversationCleanup: Record<string, unknown> = {};
    const artifactCleanup: Record<string, unknown> = {};
    const collectCleanup = async (label: string, action: () => Promise<void>) => {
      try {
        await action();
      } catch (error) {
        cleanupErrors.push(`${label}:${error instanceof Error ? error.message : "unknown"}`);
      }
    };
    const activeTaskStatuses = new Set(["planning", "running", "waiting_for_user", "waiting_for_approval"]);
    for (const taskId of taskIds) {
      await collectCleanup(`task:${taskId}`, async () => {
        const response = await requestStatus(request, "GET", `/v1/tasks/${taskId}`);
        if (response.status() === 404) {
          taskCleanup[taskId] = { status: "not_found" };
          return;
        }
        expect(response.status()).toBe(200);
        const task = await response.json() as Task;
        taskCleanup[taskId] = { status: task.status, state_version: task.state_version };
        if (activeTaskStatuses.has(task.status)) {
          const cancelled = await requestStatus(request, "POST", `/v1/tasks/${taskId}/cancel`, admin, { expected_task_state_version: task.state_version });
          expect(cancelled.status()).toBe(200);
          const terminal = await cancelled.json() as Task;
          expect(["cancelled", "succeeded", "failed", "refused"]).toContain(terminal.status);
          taskCleanup[taskId] = { status: terminal.status, state_version: terminal.state_version, cancelled: true };
        }
      });
    }
    await collectCleanup("approval_context", () => approvalContext.close());
    const requestedConversationDeletes: string[] = [];
    for (const conversationId of conversationIds) {
      await collectCleanup(`conversation:${conversationId}`, async () => {
        const deleted = await requestStatus(request, "DELETE", `/v1/conversations/${conversationId}`);
        expect(deleted.status(), "会话删除必须由本次 ID 确认，404 不是成功").toBe(202);
        const receipt = await deleted.json() as { conversation_id: string };
        expect(receipt.conversation_id).toBe(conversationId);
        conversationCleanup[conversationId] = { delete_status: deleted.status() };
        requestedConversationDeletes.push(conversationId);
      });
    }
    await Promise.all(requestedConversationDeletes.map((conversationId) => collectCleanup(`conversation_purge:${conversationId}`, async () => {
      const deadline = Date.now() + 120_000;
      while (Date.now() < deadline) {
        const response = await requestStatus(request, "GET", `/v1/conversations/${conversationId}`);
        if (response.status() === 404) {
          conversationCleanup[conversationId] = { delete_status: 202, verified_deleted: true };
          return;
        }
        expect(response.status()).toBe(200);
        await new Promise((resolve) => setTimeout(resolve, 1500));
      }
      throw new Error("会话正文未在限定时间内清理");
    })));
    for (const route of artifactContentRoutes) {
      await collectCleanup(`artifact:${route}`, async () => {
        const expired = await requestStatus(request, "GET", route);
        expect(expired.status()).toBe(410);
        artifactCleanup[route] = { status: expired.status(), verified_expired: true };
      });
    }
    try {
      if (uploaded) {
        const document = uploaded;
        await collectCleanup(`document:${document.document_id}`, async () => {
          const deleted = await requestStatus(request, "DELETE", `/v1/documents/${document.document_id}`);
          expect(deleted.status(), "必须准确清理真实上传文档，404 不是成功").toBe(200);
          const receipt = await deleted.json() as { document_id: string };
          expect(receipt.document_id).toBe(document.document_id);
          const missing = await requestStatus(request, "GET", `/v1/documents/${document.document_id}`);
          expect(missing.status()).toBe(404);
          evidence.document_cleanup = { document_id: receipt.document_id, delete_status: deleted.status(), verified_deleted: true };
        });
      }
    } finally {
      evidence.cleanup = { tasks: taskCleanup, conversations: conversationCleanup, artifacts: artifactCleanup, errors: cleanupErrors };
      evidence.finished_at = new Date().toISOString();
      if (cleanupErrors.length) evidence.status = "failed";
      await writeFile(path.join(output, "browser-evidence.json"), JSON.stringify(evidence, null, 2), "utf8");
    }
    if (cleanupErrors.length && !testFailed) throw new Error(`cleanup_failed:${cleanupErrors.join(",")}`);
  }
});
