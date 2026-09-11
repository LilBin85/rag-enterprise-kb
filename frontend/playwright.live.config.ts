import { defineConfig, devices } from "@playwright/test";
import path from "node:path";

for (const key of ["RAG_KB_E2E_BASE_URL", "RAG_KB_E2E_API_URL", "RAG_KB_E2E_RUN_ID", "RAG_KB_E2E_ADMIN_TOKEN", "RAG_KB_E2E_APPROVER_TOKEN", "RAG_KB_E2E_OUTPUT_DIR"]) {
  if (!process.env[key]) throw new Error(`真实联调缺少配置：${key}`);
}
for (const key of ["RAG_KB_E2E_BASE_URL", "RAG_KB_E2E_API_URL"]) {
  const url = new URL(process.env[key]!);
  if (url.protocol !== "http:" || url.hostname !== "127.0.0.1" || url.username || url.password) {
    throw new Error(`真实联调地址必须是无凭据的本机回环 HTTP 地址：${key}`);
  }
}
const output = process.env.RAG_KB_E2E_OUTPUT_DIR!;
const executablePath = process.env.PLAYWRIGHT_EXECUTABLE_PATH || undefined;
export default defineConfig({
  testDir: "./tests",
  testMatch: "**/live-api.spec.ts",
  workers: 1,
  retries: 0,
  timeout: 2_400_000,
  globalTimeout: 2_700_000,
  expect: { timeout: 15_000 },
  outputDir: path.join(output, "browser-results"),
  reporter: [["list"], ["json", { outputFile: path.join(output, "browser-report.json") }]],
  use: {
    ...devices["Desktop Chrome"],
    baseURL: process.env.RAG_KB_E2E_BASE_URL,
    channel: executablePath ? undefined : process.env.PLAYWRIGHT_CHANNEL || undefined,
    launchOptions: executablePath ? { executablePath } : undefined,
    trace: "off",
    video: "off",
    screenshot: "off",
    actionTimeout: 30_000,
    navigationTimeout: 30_000,
  },
});
