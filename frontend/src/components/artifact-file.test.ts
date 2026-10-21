import { describe, expect, it } from "vitest";

import type { ArtifactSummary } from "../api/types";
import { artifactDisplayName, artifactDownloadName } from "./artifact-file";

function artifact(overrides: Partial<ArtifactSummary> = {}): ArtifactSummary {
  return {
    artifact_id: "cf892401dbe0478dbc7d0f7a8f3f4a71",
    artifact_type: "report",
    content_hash: `sha256:${"a".repeat(64)}`,
    status: "ready",
    ...overrides,
  };
}

describe("产物文件名", () => {
  it("使用业务标题和服务端扩展名", () => {
    expect(artifactDownloadName(artifact({
      display_name: "新员工入职核心流程验收报告",
      file_extension: "md",
    }))).toBe("新员工入职核心流程验收报告.md");
  });

  it("清理非法字符、结尾字符和 Windows 保留名", () => {
    expect(artifactDownloadName(artifact({ display_name: "季度:报告?. ", file_extension: "json" })))
      .toBe("季度_报告_.json");
    expect(artifactDownloadName(artifact({ display_name: "CON", file_extension: "md" })))
      .toBe("_CON.md");
  });

  it("兼容旧接口并限制基础名称长度", () => {
    expect(artifactDisplayName(artifact())).toBe("报告-cf892401db");
    expect(artifactDownloadName(artifact({ display_name: "文".repeat(130), file_extension: "md" })))
      .toBe(`${"文".repeat(120)}.md`);
  });
});
