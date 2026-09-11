import { useEffect, useRef, useState, type FormEvent } from "react";
import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate, useSearchParams } from "react-router-dom";

import {
  ApiError,
  getDocument,
  getDocumentIngestionJobs,
  listDocuments,
  newIdempotencyKey,
  uploadDocument,
} from "../api/client";
import type { DocumentDetail, DocumentUpload, IngestionJob } from "../api/types";
import { useAuth } from "../app/auth";
import { useToast } from "../components/feedback";

const ACTIVE_INGESTION_STATUSES = new Set(["awaiting_object", "pending", "running"]);
const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;

function statusLabel(status: string) {
  const labels: Record<string, string> = {
    awaiting_object: "等待文件",
    pending: "排队中",
    indexing: "处理中",
    running: "处理中",
    active: "可用于问答",
    succeeded: "已完成",
    failed: "失败",
    cancelled: "已取消",
    superseded: "已替代",
    withdrawn: "已撤回",
  };
  return labels[status] ?? status;
}

export function documentIdFromFileName(fileName: string): string {
  const base = fileName.replace(/\.[^.]+$/, "").toLowerCase();
  const slug = base.normalize("NFKD").replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
  return slug || `document-${crypto.randomUUID().slice(0, 8)}`;
}

function uploadError(cause: unknown) {
  if (!(cause instanceof ApiError)) return "上传未完成，请检查网络后重试。";
  if (cause.recovery) {
    return `文档已有处理记录，可查看任务 ${cause.recovery.ingestion_job_id.slice(0, 12)} 的状态。`;
  }
  if (cause.status === 413) return "文件或 metadata 超出服务器允许的大小。";
  if (cause.status === 403) return "当前身份无权上传文档。";
  return `服务器拒绝了上传：${cause.code}`;
}

type UploadTarget = Pick<DocumentDetail, "document_id" | "title" | "department"> | null;

function UploadDialog({
  target,
  onClose,
  onAccepted,
}: {
  target: UploadTarget;
  onClose: () => void;
  onAccepted: (upload: DocumentUpload) => void;
}) {
  const { token } = useAuth();
  const [file, setFile] = useState<File | null>(null);
  const [documentId, setDocumentId] = useState(target?.document_id ?? "");
  const [title, setTitle] = useState(target?.title ?? "");
  const [department, setDepartment] = useState(target?.department ?? "");
  const [error, setError] = useState<string | null>(null);
  const upload = useMutation({
    mutationFn: () => uploadDocument(
      token!,
      {
        documentId,
        title,
        department,
        file: file!,
        versionOf: target?.document_id,
      },
      newIdempotencyKey(),
    ),
    onSuccess: onAccepted,
    onError: (cause) => setError(uploadError(cause)),
  });

  const selectFile = (next: File | null) => {
    setError(null);
    setFile(next);
    if (!next) return;
    const lower = next.name.toLowerCase();
    if (!lower.endsWith(".md") && !lower.endsWith(".pdf")) {
      setError("仅支持 Markdown 或 PDF 文件。");
      return;
    }
    if (next.size > MAX_UPLOAD_BYTES) {
      setError("文件不能超过 10 MiB。");
      return;
    }
    if (!target) {
      setDocumentId((current) => current || documentIdFromFileName(next.name));
      setTitle((current) => current || next.name.replace(/\.[^.]+$/, ""));
    }
  };

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!file || error || !documentId.trim() || !title.trim() || !department.trim()) {
      if (!error) setError("请填写文档标识、标题、部门并选择文件。");
      return;
    }
    upload.mutate();
  };

  return (
    <div className="modal-layer" role="presentation">
      <button className="modal-scrim" type="button" aria-label="关闭上传窗口" onClick={onClose} />
      <section className="modal-dialog" role="dialog" aria-modal="true" aria-labelledby="upload-title">
        <div className="drawer-heading">
          <div><p className="eyebrow">{target ? "新版本" : "摄取新文档"}</p><h2 id="upload-title">{target ? `上传 ${target.title} 的新版本` : "上传文档"}</h2></div>
          <button className="icon-button" type="button" onClick={onClose}>关闭</button>
        </div>
        <form className="stack-form compact-form" onSubmit={submit}>
          <label><span>文件</span><input type="file" accept=".md,.pdf,application/pdf,text/markdown" onChange={(event) => selectFile(event.target.files?.[0] ?? null)} required /><small>支持 Markdown、PDF，最大 10 MiB；服务器会再次校验。</small></label>
          <label><span>文档标识</span><input value={documentId} onChange={(event) => setDocumentId(event.target.value)} disabled={Boolean(target)} maxLength={200} required /><small>{target ? "新版本沿用原文档标识。" : "根据文件名自动生成，可在上传前调整。"}</small></label>
          <label><span>标题</span><input value={title} onChange={(event) => setTitle(event.target.value)} disabled={Boolean(target)} maxLength={500} required /></label>
          <label><span>部门</span><input value={department} onChange={(event) => setDepartment(event.target.value)} disabled={Boolean(target)} maxLength={200} required /></label>
          {error && <p className="form-error" role="alert">{error}</p>}
          <div className="modal-actions"><button className="text-button" type="button" onClick={onClose}>取消</button><button className="primary-button" type="submit" disabled={upload.isPending || Boolean(error)}>{upload.isPending ? "正在接收" : "提交上传"}</button></div>
        </form>
      </section>
    </div>
  );
}

function IngestionState({ detail, jobs }: { detail: DocumentDetail; jobs: IngestionJob[] }) {
  return (
    <section className="document-detail-section">
      <p className="section-label">版本与摄取</p>
      <div className="version-list">
        {detail.versions.map((version) => {
          const job = jobs.find((item) => item.version_id === version.version_id);
          return (
            <div className="version-row" key={version.version_id}>
              <div><strong>V{version.version}</strong><span>{statusLabel(version.lifecycle_status)}</span></div>
              <div className="version-job">
                {job ? <><span>{statusLabel(job.status)}</span>{job.error_code && <small>{job.error_code}</small>}</> : <span>尚无摄取状态</span>}
              </div>
            </div>
          );
        })}
      </div>
      {jobs.some((job) => ACTIVE_INGESTION_STATUSES.has(job.status)) && <p className="polling-note">摄取进行中，正在自动更新状态。</p>}
    </section>
  );
}

export function DocumentsPage() {
  const navigate = useNavigate();
  const { token, session, hasPermission } = useAuth();
  const toast = useToast();
  const queryClient = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();
  const [uploadTarget, setUploadTarget] = useState<UploadTarget | undefined>(undefined);
  const selectedDocumentId = searchParams.get("document");
  const previousJobs = useRef<string | null>(null);
  const documents = useInfiniteQuery({
    queryKey: ["documents", session],
    initialPageParam: null as string | null,
    queryFn: ({ pageParam, signal }) => listDocuments(token!, pageParam, signal),
    getNextPageParam: (page) => page.next_cursor,
    enabled: Boolean(token && hasPermission("document:read")),
  });
  const detail = useQuery({
    queryKey: ["document", session, selectedDocumentId],
    queryFn: ({ signal }) => getDocument(token!, selectedDocumentId!, signal),
    enabled: Boolean(token && selectedDocumentId && hasPermission("document:read")),
  });
  const jobs = useQuery({
    queryKey: ["document-jobs", session, selectedDocumentId],
    queryFn: ({ signal }) => getDocumentIngestionJobs(token!, selectedDocumentId!, signal),
    enabled: Boolean(token && selectedDocumentId && hasPermission("document:read")),
    refetchInterval: (query) => query.state.data?.some((item) => ACTIVE_INGESTION_STATUSES.has(item.status)) ? 3000 : false,
  });

  const jobSignature = jobs.data?.map((item) => `${item.job_id}:${item.status}:${item.error_code ?? ""}`).join("|") ?? null;
  useEffect(() => {
    if (!selectedDocumentId || !jobSignature || previousJobs.current === jobSignature) return;
    previousJobs.current = jobSignature;
    void queryClient.invalidateQueries({ queryKey: ["document", session, selectedDocumentId] });
    if (!jobs.data?.some((item) => ACTIVE_INGESTION_STATUSES.has(item.status))) {
      void queryClient.invalidateQueries({ queryKey: ["documents", session] });
    }
  }, [jobSignature, jobs.data, queryClient, selectedDocumentId, session]);

  if (!hasPermission("document:read")) {
    return <section className="restricted-page"><p className="eyebrow">受限入口</p><h1>当前身份无文档读取权限</h1><p>请切换到具有文档权限的演示身份后再访问此页面。</p></section>;
  }

  const rows = documents.data?.pages.flatMap((page) => page.items) ?? [];
  const select = (documentId: string) => setSearchParams({ document: documentId });
  const accepted = (upload: DocumentUpload) => {
    setUploadTarget(undefined);
    select(upload.document_id);
    void queryClient.invalidateQueries({ queryKey: ["documents", session] });
    toast({ title: "已接收，处理中", detail: `摄取任务 ${upload.ingestion_job_id.slice(0, 12)} 正在等待服务器处理。` });
  };

  return (
    <section className="documents-page">
      <header className="page-header">
        <div><p className="eyebrow">资料库</p><h1>文档</h1><p>只有状态为“可用于问答”的版本会作为可检索资料生效。</p></div>
        {hasPermission("document:write") && <button className="primary-button" type="button" onClick={() => setUploadTarget(null)}>上传文档</button>}
      </header>
      <div className="documents-layout">
        <section className="document-list-panel" aria-label="文档列表">
          {documents.isLoading && <><div className="skeleton h-14 w-full" /><div className="skeleton h-14 w-full" /></>}
          {documents.isError && <p className="empty-copy">无法读取文档列表，请刷新后重试。</p>}
          {!documents.isLoading && !rows.length && <p className="empty-copy">尚无当前身份可读取的文档。</p>}
          {rows.map((document) => <button className={`document-row ${selectedDocumentId === document.document_id ? "selected" : ""}`} type="button" key={document.document_id} onClick={() => select(document.document_id)}><strong>{document.title}</strong><span>{document.department}</span><small>{statusLabel(document.status)}</small></button>)}
          {documents.hasNextPage && <button className="text-button load-more" type="button" disabled={documents.isFetchingNextPage} onClick={() => documents.fetchNextPage()}>{documents.isFetchingNextPage ? "正在加载" : "加载更多"}</button>}
        </section>
        <section className="document-detail-panel" aria-live="polite">
          {!selectedDocumentId && <div className="detail-empty"><p className="eyebrow">选择一份资料</p><h2>查看版本和摄取状态</h2><p>上传完成不代表新版本已经可用于问答。请选择左侧文档查看当前进度。</p></div>}
          {selectedDocumentId && detail.isLoading && <><div className="skeleton h-8 w-2/3" /><div className="skeleton h-36 w-full" /></>}
          {selectedDocumentId && detail.isError && <div className="detail-empty"><p className="form-error">无法读取该文档，可能已被删除或无权访问。</p></div>}
          {detail.data && <>
            <div className="detail-heading">
              <div><p className="eyebrow">{detail.data.department}</p><h2>{detail.data.title}</h2><p>{statusLabel(detail.data.status)}</p></div>
              <div className="detail-actions">
                {detail.data.status === "active" && hasPermission("knowledge:read") && (
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={() => navigate(`/chat?scopeDocument=${encodeURIComponent(detail.data.document_id)}&scopeTitle=${encodeURIComponent(detail.data.title)}`)}
                  >
                    仅在此文档中提问
                  </button>
                )}
                {hasPermission("document:write") && <button className="secondary-button" type="button" onClick={() => setUploadTarget(detail.data)}>上传新版本</button>}
              </div>
            </div>
            {jobs.isLoading ? <div className="skeleton h-28 w-full" /> : <IngestionState detail={detail.data} jobs={jobs.data ?? []} />}
          </>}
        </section>
      </div>
      {uploadTarget !== undefined && <UploadDialog target={uploadTarget} onClose={() => setUploadTarget(undefined)} onAccepted={accepted} />}
    </section>
  );
}
