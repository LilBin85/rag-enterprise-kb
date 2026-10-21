import { describe, expect, it } from "vitest";

import { normalizeMarkdownTables } from "./markdown-content";

describe("normalizeMarkdownTables", () => {
  it("repairs a two-column table emitted on one line", () => {
    const source = "| 维度 | 内容 | |---|---| | **责任人** | 人力资源 | | **审批条件** | 双重审批 |";

    expect(normalizeMarkdownTables(source)).toBe(
      "| 维度 | 内容 |\n|---|---|\n| **责任人** | 人力资源 |\n| **审批条件** | 双重审批 |",
    );
  });
});
