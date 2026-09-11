import { useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { ApiError, decideApproval, getApproval, listApprovals } from "../api/client";
import type { Approval, ApprovalStatus } from "../api/types";
import { useAuth } from "../app/auth";
import { useToast } from "../components/feedback";

const approvalTabs: Array<{ value: ApprovalStatus; label: string }> = [
  { value: "pending", label: "待审批" },
  { value: "approved", label: "已批准" },
  { value: "rejected", label: "已拒绝" },
  { value: "executed", label: "已执行" },
  { value: "expired", label: "已过期" },
];

const statusLabel: Record<ApprovalStatus, string> = {
  pending: "待审批",
  approved: "已批准",
  rejected: "已拒绝",
  executed: "已执行",
  expired: "已过期",
};

function formatTime(value: string) {
  return new Intl.DateTimeFormat("zh-CN", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

function ApprovalBadge({ status }: { status: ApprovalStatus }) {
  return <span className={`status-badge approval-${status}`}>{statusLabel[status]}</span>;
}

function DecisionDialog({
  approval,
  decision,
  pending,
  onClose,
  onSubmit,
}: {
  approval: Approval;
  decision: "approve" | "reject";
  pending: boolean;
  onClose: () => void;
  onSubmit: (reason: string) => void;
}) {
  const [reason, setReason] = useState("");
  const approving = decision === "approve";
  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    onSubmit(reason.trim());
  };
  return (
    <div className="modal-layer" role="presentation">
      <button className="modal-scrim" type="button" aria-label="关闭确认窗口" onClick={onClose} />
      <section className="modal-dialog decision-dialog" role="dialog" aria-modal="true" aria-labelledby="decision-title">
        <p className="eyebrow">需要确认</p>
        <h2 id="decision-title">{approving ? "批准此操作？" : "拒绝此操作？"}</h2>
        <p className="decision-summary">{approval.summary || "服务端未提供可展示的操作摘要。"}</p>
        <p className="decision-note">此决定只会提交恢复命令，不能代表任务已经执行完成。</p>
        <form className="stack-form compact-form" onSubmit={submit}>
          <label><span>理由（可选）</span><textarea value={reason} onChange={(event) => setReason(event.target.value)} maxLength={200} rows={3} placeholder="最多 200 字" /><small>{reason.length}/200</small></label>
          <div className="modal-actions"><button className="text-button" type="button" onClick={onClose}>取消</button><button className={approving ? "primary-button" : "danger-button compact-danger"} type="submit" disabled={pending}>{pending ? "正在提交" : approving ? "确认批准" : "确认拒绝"}</button></div>
        </form>
      </section>
    </div>
  );
}

export function ApprovalsPage() {
  const { token, session, hasPermission } = useAuth();
  const queryClient = useQueryClient();
  const toast = useToast();
  const [status, setStatus] = useState<ApprovalStatus>("pending");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [decision, setDecision] = useState<"approve" | "reject" | null>(null);
  const approvals = useQuery({
    queryKey: ["approvals", session, status],
    queryFn: ({ signal }) => listApprovals(token!, status, signal),
    enabled: Boolean(token && hasPermission("task:approve")),
    refetchInterval: status === "pending" ? 5000 : false,
  });
  const detail = useQuery({
    queryKey: ["approval", session, selectedId],
    queryFn: ({ signal }) => getApproval(token!, selectedId!, signal),
    enabled: Boolean(token && selectedId && hasPermission("task:approve")),
  });
  const decide = useMutation({
    mutationFn: (values: { decision: "approve" | "reject"; reason: string }) => decideApproval(
      token!,
      selectedId!,
      {
        expected_task_state_version: detail.data!.task_state_version,
        decision: values.decision,
        ...(values.reason ? { reason: values.reason } : {}),
      },
    ),
    onSuccess: (result) => {
      if (selectedId) queryClient.setQueryData(["approval", session, selectedId], result);
      void queryClient.invalidateQueries({ queryKey: ["approvals", session] });
      setDecision(null);
      toast({
        title: result.status === "approved" ? "已接受批准决定" : "已接受拒绝决定",
        detail: "服务器已保存恢复命令，任务仍需等待 Worker 处理。",
      });
    },
    onError: async (cause) => {
      if (cause instanceof ApiError && cause.status === 409 && selectedId) {
        try {
          await queryClient.fetchQuery({
            queryKey: ["approval", session, selectedId],
            queryFn: () => getApproval(token!, selectedId),
          });
        } catch {
          // 详情刷新失败时也不重试原决定，避免覆盖服务器的最新状态。
        }
        setDecision(null);
        toast({ title: "审批状态已变化", detail: "已重新读取详情；请确认最新状态后再决定。" });
        return;
      }
      toast({ title: "未能提交审批决定", detail: cause instanceof ApiError && cause.status === 403 ? "服务器拒绝了该决定。" : "请稍后重试。" });
    },
  });

  if (!hasPermission("task:approve")) {
    return <section className="restricted-page"><p className="eyebrow">受限入口</p><h1>当前身份无审批权限</h1><p>审批请切换到独立 approver 演示身份。发起任务的身份不能自行审批。</p></section>;
  }

  const selected = detail.data;
  return (
    <section className="approvals-page">
      <header className="page-header">
        <div><p className="eyebrow">受控操作</p><h1>审批</h1><p>只显示当前审批身份有权处理的操作。批准不等于任务已经执行完成。</p></div>
      </header>
      <div className="approval-tabs" role="tablist" aria-label="审批状态">
        {approvalTabs.map((tab) => <button className={status === tab.value ? "active" : ""} type="button" role="tab" aria-selected={status === tab.value} key={tab.value} onClick={() => { setStatus(tab.value); setSelectedId(null); }}>{tab.label}</button>)}
      </div>
      <div className="approvals-layout">
        <section className="approval-list-panel" aria-label="审批列表">
          {approvals.isLoading && <><div className="skeleton h-24 w-full" /><div className="skeleton h-24 w-full" /></>}
          {approvals.isError && <p className="empty-copy">无法读取审批列表，请刷新后重试。</p>}
          {!approvals.isLoading && !approvals.data?.length && <p className="empty-copy">此状态下没有可展示的审批项。</p>}
          {approvals.data?.map((approval) => <button className={`approval-row ${selectedId === approval.approval_id ? "selected" : ""}`} type="button" key={approval.approval_id} onClick={() => setSelectedId(approval.approval_id)}><div><strong>{approval.tool_name}</strong><span>{approval.summary || "无可展示的操作摘要"}</span></div><div><ApprovalBadge status={approval.status} /><small>截止 {formatTime(approval.expires_at)}</small></div></button>)}
        </section>
        <section className="approval-detail-panel" aria-live="polite">
          {!selectedId && <div className="detail-empty"><p className="eyebrow">选择一项操作</p><h2>查看安全摘要后再决定</h2><p>界面不会显示原始参数、检查点或内部哈希；服务端仍会校验所有绑定。</p></div>}
          {selectedId && detail.isLoading && <><div className="skeleton h-8 w-2/3" /><div className="skeleton h-36 w-full" /></>}
          {selectedId && detail.isError && <div className="detail-empty"><p className="form-error">无法读取审批详情，请返回列表重新选择。</p></div>}
          {selected && <>
            <div className="detail-heading"><div><p className="eyebrow">操作工具</p><h2>{selected.tool_name}</h2><p>版本 {selected.tool_version}</p></div><ApprovalBadge status={selected.status} /></div>
            <section className="approval-detail-section"><p className="section-label">操作摘要</p><p>{selected.summary || "服务端未提供可展示的操作摘要。"}</p></section>
            <section className="approval-detail-section"><dl className="approval-facts"><div><dt>任务</dt><dd>{selected.task_id}</dd></div><div><dt>有效期至</dt><dd>{formatTime(selected.expires_at)}</dd></div><div><dt>当前状态</dt><dd>{statusLabel[selected.status]}</dd></div></dl></section>
            {selected.status === "pending" && <div className="approval-actions"><button className="secondary-button" type="button" onClick={() => setDecision("approve")}>批准</button><button className="danger-button compact-danger" type="button" onClick={() => setDecision("reject")}>拒绝</button></div>}
            {selected.status !== "pending" && <p className="empty-copy">该审批已处理；如需确认任务执行结果，请在关联会话的任务详情中查看。</p>}
          </>}
        </section>
      </div>
      {selected && decision && <DecisionDialog approval={selected} decision={decision} pending={decide.isPending} onClose={() => setDecision(null)} onSubmit={(reason) => decide.mutate({ decision, reason })} />}
    </section>
  );
}
