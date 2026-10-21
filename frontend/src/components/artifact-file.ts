import type { ArtifactSummary } from "../api/types";

const WINDOWS_RESERVED_NAME = /^(con|prn|aux|nul|com[1-9]|lpt[1-9])$/i;

export function artifactTypeLabel(type: string) {
  return type === "work_item_draft" ? "工作项草稿" : "报告";
}

export function artifactDisplayName(artifact: ArtifactSummary) {
  return artifact.display_name?.trim()
    || `${artifactTypeLabel(artifact.artifact_type)}-${artifact.artifact_id.slice(0, 10)}`;
}

export function artifactDownloadName(artifact: ArtifactSummary) {
  const fallback = artifactDisplayName({ ...artifact, display_name: undefined });
  let base = artifactDisplayName(artifact)
    .normalize("NFC")
    .replace(/[<>:"/\\|?*\u0000-\u001F]/g, "_")
    .replace(/[ .]+$/g, "")
    .trim();
  base = Array.from(base).slice(0, 120).join("").replace(/[ .]+$/g, "");
  if (!base) base = fallback;
  if (WINDOWS_RESERVED_NAME.test(base)) base = `_${base}`;
  const extension = artifact.file_extension === "md" ? "md" : "json";
  return `${base}.${extension}`;
}
