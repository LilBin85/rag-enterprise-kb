import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import ReactMarkdown from "react-markdown";

import {
  ApiError,
  cancelTask,
  getArtifactContent,
  getTask,
  getTaskEvents,
  getTaskResult,
} from "../api/client";
import type { ArtifactSummary, Citation, Task } from "../api/types";
import { useAuth } from "../app/auth";
import { useToast } from "./feedback";

const activeStatuses = new Set([
  "planning",
  "running",
  "waiting_for_user",
  "waiting_for_approval",
]);

const taskLabels: Record<string, string> = {
  planning: "规划中",
  running: "运行中",
  waiting_for_user: "等待补充",
  waiting_for_approval: "等待审批",
  succeeded: "完成",
  refused: "已拒答",
  failed: "失败",
  cancelled: "已取消",
  unknown: "状态待确认",
};

function TaskBadge({ task }: { task: Task }) {
  const label = task.status === "succeeded" && !task.result_available
    ? "结果待确认"
    : taskLabels[task.status] ?? task.status;
  return <span className={`status-badge status-${task.status}`}>{label}</span>;
}

export function CitationList({ citations }: { citations: Citation[] }) {
  if (!citations.length) return null;
  return (
    <section className="source-list" aria-label="来源">
      <p className="section-label">来源</p>
      {citations.map((citation) => (
        <Link
          className="source-item"
          key={`${citation.document_id}-${citation.chunk_id}-${citation.source_locator}`}
          to={`/documents?document=${encodeURIComponent(citation.document_id)}`}
        >
          <span>{citation.document_title || citation.source_file_name}</span>
          <small>
            {citation.document_title ? `${citation.source_file_name} · ` : ""}
            V{citation.document_version} · {citation.document_id.slice(0, 8)} · {citation.source_locator}
          </small>
        </Link>
      ))}
    </section>
  );
}

export function TaskCard({ taskId, onOpen }: { taskId: string; onOpen: () => void }) {
  const { token, session } = useAuth();
  const task = useQuery({
    queryKey: ["task", session, taskId],
    queryFn: ({ signal }) => getTask(token!, taskId, signal),
    enabled: Boolean(token),
    refetchInterval: (query) => activeStatuses.has(query.state.data?.status ?? "") ? 3000 : false,
  });
  if (!task.data) return null;
  return (
    <button className="task-card" type="button" onClick={onOpen}>
      <span className="section-label">任务</span>
      <strong>{task.data.task_mode === "controlled_action" ? "受控操作" : "深入研究"}</strong>
      <TaskBadge task={task.data} />
      {task.data.interrupt_summary && <span className="task-summary">{task.data.interrupt_summary}</span>}
    </button>
  );
}

export function TaskDrawer({ taskId, onClose }: { taskId: string; onClose: () => void }) {
  const { token, session, hasPermission } = useAuth();
  const toast = useToast();
  const queryClient = useQueryClient();
  const [selectedArtifact, setSelectedArtifact] = useState<ArtifactSummary | null>(null);
  const task = useQuery({
    queryKey: ["task", session, taskId],
    queryFn: ({ signal }) => getTask(token!, taskId, signal),
    enabled: Boolean(token),
    refetchInterval: (query) => activeStatuses.has(query.state.data?.status ?? "") ? 3000 : false,
  });
  const events = useQuery({
    queryKey: ["task-events", session, taskId],
    queryFn: ({ signal }) => getTaskEvents(token!, taskId, signal),
    enabled: Boolean(token),
    refetchInterval: task.data && activeStatuses.has(task.data.status) ? 3000 : false,
  });
  const result = useQuery({
    queryKey: ["task-result", session, taskId],
    queryFn: ({ signal }) => getTaskResult(token!, taskId, signal),
    enabled: Boolean(token && task.data?.result_available),
  });
  const artifact = useQuery({
    queryKey: ["artifact", session, taskId, selectedArtifact?.artifact_id],
    queryFn: async () => {
      const blob = await getArtifactContent(token!, taskId, selectedArtifact!.artifact_id);
      return { blob, text: await blob.text() };
    },
    enabled: Boolean(token && selectedArtifact),
  });
  const cancel = useMutation({
    mutationFn: () => cancelTask(token!, taskId, task.data!.state_version),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["task", session, taskId] });
      toast({ title: "已提交取消请求", detail: "请等待服务器确认最终状态。" });
    },
    onError: (cause) => {
      void queryClient.invalidateQueries({ queryKey: ["task", session, taskId] });
      toast({
        title: "取消未完成",
        detail: cause instanceof ApiError && cause.status === 409 ? "任务状态已变化，请重新确认。" : "请刷新后重试。",
      });
    },
  });

  useEffect(() => setSelectedArtifact(null), [taskId]);

  if (task.isLoading) {
    return <aside className="drawer" aria-busy="true"><div className="skeleton h-6 w-24" /><div className="skeleton h-24 w-full" /></aside>;
  }
  if (!task.data) {
    return <aside className="drawer"><p className="empty-copy">无法读取任务详情。</p><button className="text-button" onClick={onClose}>关闭</button></aside>;
  }
  const canCancel = hasPermission("task:read") && activeStatuses.has(task.data.status);
  const artifacts = result.data?.artifacts ?? [];
  const downloadArtifact = () => {
    if (!artifact.data || !selectedArtifact) return;
    const url = URL.createObjectURL(artifact.data.blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = selectedArtifact.artifact_id;
    document.body.append(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  };
  return (
    <aside className="drawer" aria-label="任务详情">
      <div className="drawer-heading">
        <div>
          <p className="eyebrow">执行详情</p>
          <h2>任务状态</h2>
        </div>
        <button className="icon-button" type="button" onClick={onClose} aria-label="关闭任务详情">关闭</button>
      </div>
      <TaskBadge task={task.data} />
      <p className="task-goal">{task.data.goal}</p>
      {task.data.interrupt_summary && (
        <div className="notice-block">
          <strong>{taskLabels[task.data.status] ?? "需要处理"}</strong>
          <p>{task.data.interrupt_summary}</p>
        </div>
      )}
      {canCancel && (
        <button
          className="danger-button"
          type="button"
          disabled={cancel.isPending}
          onClick={() => window.confirm("确认取消此任务？") && cancel.mutate()}
        >
          取消任务
        </button>
      )}
      <section className="drawer-section">
        <p className="section-label">执行阶段</p>
        {events.data?.length ? (
          <ol className="event-list">
            {events.data.slice(-5).reverse().map((event) => (
              <li key={event.sequence}>
                <span>{event.stage}</span>
                <small>{event.status}{event.error_code ? ` · ${event.error_code}` : ""}</small>
              </li>
            ))}
          </ol>
        ) : <p className="empty-copy">暂无可展示的阶段事件。</p>}
      </section>
      {result.data && (
        <section className="drawer-section result-block">
          <p className="section-label">最终结果</p>
          <ReactMarkdown skipHtml>{result.data.answer || "任务未提供可展示的正文。"}</ReactMarkdown>
          <CitationList citations={result.data.citations} />
        </section>
      )}
      {task.data.status === "succeeded" && !task.data.result_available && (
        <p className="empty-copy">任务已完成，结果仍在服务器确认中。</p>
      )}
      {artifacts.length > 0 && (
        <section className="drawer-section">
          <p className="section-label">产物</p>
          <div className="artifact-list">
            {artifacts.map((item) => (
              <button key={item.artifact_id} className="artifact-button" type="button" onClick={() => setSelectedArtifact(item)}>
                <span>{item.artifact_type}</span><small>{item.artifact_id.slice(0, 10)}</small>
              </button>
            ))}
          </div>
          {selectedArtifact && (
            <div className="artifact-preview">
              <div className="artifact-preview-heading"><strong>{selectedArtifact.artifact_type}</strong><button className="text-button" onClick={() => setSelectedArtifact(null)}>收起</button></div>
              {artifact.isLoading && <div className="skeleton h-20 w-full" />}
              {artifact.isError && <p className="empty-copy">产物正文不可用、已隐藏或已过期。</p>}
              {artifact.data && <><pre>{artifact.data.text}</pre><button className="secondary-button artifact-download" type="button" onClick={downloadArtifact}>下载原始内容</button></>}
            </div>
          )}
        </section>
      )}
    </aside>
  );
}
