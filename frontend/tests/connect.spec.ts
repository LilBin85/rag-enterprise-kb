import { expect, test } from "@playwright/test";

test("opens the token connection screen without a saved session", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "连接知识工作台" })).toBeVisible();
  await expect(page.getByLabel("演示令牌")).toBeVisible();
});
