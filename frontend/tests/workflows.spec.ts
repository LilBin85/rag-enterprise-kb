import { expect, test, type Page } from "@playwright/test";

const admin = {
  tenant_id: "local-demo",
  user_id: "demo-admin",
  roles: ["admin"],
  permissions: ["knowledge:read", "document:read", "document:write", "task:read", "task:approve"],
};

const approver = {
  tenant_id: "local-demo",
  user_id: "demo-approver",
  roles: ["approver"],
  permissions: ["task:approve"],
};

const approval = {
  approval_id: "approval-1",
  task_id: "task-1",
  task_state_version: 3,
  summary: "将年假制度更新为已审核版本。",
  plan_revision: 1,
  plan_digest: "hidden",
  evidence_hash: "hidden",
  canonical_arguments_hash: "hidden",
  checkpoint_binding_hash: "hidden",
  tool_name: "更新年假制度",
  tool_version: "v1",
  expires_at: "2027-01-02T03:04:05Z",
  status: "pending",
  approved_by: null,
  approved_at: null,
  rejected_by: null,
  rejected_at: null,
};

async function mockApi(page: Page) {
  let conversationVisible = true;
  await page.route("**/v1/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const json = (body: unknown, status = 200) => route.fulfill({
      status,
      contentType: "application/json",
      body: JSON.stringify(body),
    });
    if (url.pathname === "/v1/me") {
      return json(request.headers().authorization === "Bearer approver-token" ? approver : admin);
    }
    if (url.pathname === "/v1/conversations") return json({
      items: conversationVisible ? [{
        conversation_id: "conversation-1",
        title: "需要删除的会话",
        state_version: 1,
        status: "active",
        created_at: "2027-01-02T03:04:05Z",
        updated_at: "2027-01-02T03:04:05Z",
        active_turn: null,
      }] : [],
      next_cursor: null,
    });
    if (url.pathname === "/v1/conversations/conversation-1" && request.method() === "GET") {
      return json({
        conversation_id: "conversation-1",
        title: "需要删除的会话",
        state_version: 1,
        status: "active",
        created_at: "2027-01-02T03:04:05Z",
        updated_at: "2027-01-02T03:04:05Z",
        active_turn: null,
      });
    }
    if (url.pathname === "/v1/conversations/conversation-1" && request.method() === "DELETE") {
      conversationVisible = false;
      return json({ conversation_id: "conversation-1", status: "deleting" }, 202);
    }
    if (url.pathname === "/v1/documents") {
      return json({
        items: [{ document_id: "annual-leave", title: "员工年假制度", department: "hr", status: "active" }],
        next_cursor: null,
      });
    }
    if (url.pathname === "/v1/documents/annual-leave") {
      return json({
        document_id: "annual-leave",
        title: "员工年假制度",
        department: "hr",
        status: "active",
        versions: [{ version_id: "version-1", version: 1, lifecycle_status: "active" }],
      });
    }
    if (url.pathname === "/v1/documents/annual-leave/ingestion-jobs") {
      return json([{ job_id: "job-1", version_id: "version-1", status: "succeeded", attempts: 1, error_code: null }]);
    }
    if (url.pathname === "/v1/approvals") return json([approval]);
    if (url.pathname === "/v1/approvals/approval-1") return json(approval);
    return json({ detail: "not found" }, 404);
  });
}

async function connect(page: Page, token: string) {
  await page.goto("/");
  await page.getByLabel("演示令牌").fill(token);
  await page.getByRole("button", { name: "连接身份" }).click();
  await expect(page.getByText(token === "approver-token" ? "demo-approver" : "demo-admin")).toBeVisible();
}

test("shows document versions and ingestion status for an authorized identity", async ({ page }) => {
  await mockApi(page);
  await connect(page, "admin-token");

  await page.getByRole("link", { name: "文档" }).click();
  await expect(page.getByRole("heading", { name: "文档" })).toBeVisible();
  await page.getByRole("button", { name: /员工年假制度/ }).click();
  const version = page.locator(".version-row");
  await expect(version.getByText("可用于问答", { exact: true })).toBeVisible();
  await expect(version.getByText("已完成", { exact: true })).toBeVisible();
});

test("removes an accepted conversation deletion from the rail immediately", async ({ page }) => {
  await mockApi(page);
  await connect(page, "admin-token");

  await expect(page.getByText("需要删除的会话", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "删除会话" }).click();
  await expect(page.getByRole("heading", { name: "删除“需要删除的会话”？" })).toBeVisible();
  await page.getByRole("button", { name: "确认删除" }).click();

  await expect(page.getByText("需要删除的会话", { exact: true })).toHaveCount(0);
  await expect(page.getByText("会话已移除", { exact: true })).toBeVisible();
});

test("shows an approver the decision confirmation step", async ({ page }) => {
  await mockApi(page);
  await connect(page, "approver-token");

  await page.getByRole("link", { name: "审批" }).click();
  await page.getByRole("button", { name: /更新年假制度/ }).click();
  await expect(page.locator(".approval-detail-panel").getByText("将年假制度更新为已审核版本。", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "批准" }).click();
  await expect(page.getByRole("heading", { name: "批准此操作？" })).toBeVisible();
  await expect(page.getByText("此决定只会提交恢复命令，不能代表任务已经执行完成。")).toBeVisible();
});
