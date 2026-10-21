import {
  useEffect,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
} from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import {
  ApiError,
  createConversation,
  getConversation,
  getMessages,
  getTask,
  newIdempotencyKey,
  submitMessage,
} from "../api/client";
import type { ConversationMessage, MessageMode, TurnResponse } from "../api/types";
import { useAuth } from "../app/auth";
import { CitationList, TaskCard, TaskDrawer } from "../components/task-drawer";
import { useToast } from "../components/feedback";
import { MarkdownContent } from "../components/markdown-content";

type DraftState = {
  conversationId: string;
  idempotencyKey: string;
  content: string;
  stage: string;
  preview: string;
  userMessageId?: string;
  interrupted?: boolean;
  interruptedAt?: number;
  scopeDocumentId?: string;
};

type SubmissionError = {
  code: string;
  content: string;
  scopeDocumentId?: string;
};

const modeLabels: Record<MessageMode, string> = {
  auto: "自动",
  answer: "知识问答",
  research: "深入研究",
  controlled_action: "受控操作",
};

const modeHints: Record<MessageMode, string> = {
  auto: "自动选择问答、研究或受控操作",
  answer: "只读问答，不创建持久化产物",
  research: "只读研究，不创建持久化产物",
  controlled_action: "可研究并申请写入，执行前需要审批",
};

const submissionErrorMessages: Record<string, string> = {
  controlled_action_explicit_mode_required: "此请求包含写入操作，请切换到受控操作。",
  controlled_action_unavailable: "当前环境未启用受控操作。",
  conversation_has_active_turn: "当前会话已有任务正在执行。",
};

function submissionErrorMessage(code: string) {
  return submissionErrorMessages[code] ?? "服务器未接受本次请求，请稍后重试。";
}

function titleFor(content: string) {
  return content.replace(/\s+/g, " ").slice(0, 30);
}

function isComposing(event: KeyboardEvent<HTMLTextAreaElement>) {
  return event.nativeEvent.isComposing || event.keyCode === 229;
}

function MessageBubble({ message, onOpenTask }: { message: ConversationMessage; onOpenTask: (taskId: string) => void }) {
  const isUser = message.role === "user";
  return (
    <article className={`message-bubble ${isUser ? "from-user" : "from-assistant"}`}>
      <div className="message-meta"><span>{isUser ? "你" : "知识工作台"}</span><time>{new Date(message.created_at).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })}</time></div>
      {message.visibility === "redacted" ? (
        <p className="redacted-copy">该消息因 {message.redaction_reason ?? "当前授权或资料状态"} 无法展示。</p>
      ) : isUser ? (
        <p className="user-message-content">{message.content}</p>
      ) : (
        <MarkdownContent>{message.content ?? ""}</MarkdownContent>
      )}
      {!isUser && <CitationList citations={message.citations} />}
      {message.task_id && (
        <button className="message-task-link" type="button" onClick={() => onOpenTask(message.task_id!)}>查看关联任务</button>
      )}
    </article>
  );
}

function DraftBubble({ draft, knownMessageIds }: { draft: DraftState; knownMessageIds: Set<string> }) {
  return (
    <>
      {(!draft.userMessageId || !knownMessageIds.has(draft.userMessageId)) && (
        <article className="message-bubble from-user pending-user"><div className="message-meta"><span>你</span><small>已提交</small></div><p className="user-message-content">{draft.content}</p></article>
      )}
      {!draft.interrupted && (
        <article className="message-bubble from-assistant preview-message">
          <div className="message-meta"><span>知识工作台</span><small>{draft.stage === "validating" ? "正在校验来源" : "生成中，尚未校验"}</small></div>
          {draft.preview ? <MarkdownContent>{draft.preview}</MarkdownContent> : <div className="skeleton h-5 w-32" />}
        </article>
      )}
      {draft.interrupted && (
        <article className="stream-notice" role="status">
          <strong>连接已中断</strong>
          <p>正在从服务器恢复会话状态；不会以新请求重新生成。</p>
        </article>
      )}
    </>
  );
}

export function ChatPage() {
  const { conversationId } = useParams();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const queryClient = useQueryClient();
  const toast = useToast();
  const { token, session } = useAuth();
  const [inputs, setInputs] = useState<Record<string, string>>({});
  const [mode, setMode] = useState<MessageMode>("auto");
  const [drafts, setDrafts] = useState<Record<string, DraftState>>({});
  const [submissionErrors, setSubmissionErrors] = useState<Record<string, SubmissionError>>({});
  const [routeNotices, setRouteNotices] = useState<Record<string, string>>({});
  const [drawerTaskId, setDrawerTaskId] = useState<string | null>(null);
  const submitting = useRef(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showLatest, setShowLatest] = useState(false);
  const controllers = useRef(new Set<AbortController>());
  const scrollArea = useRef<HTMLDivElement>(null);
  const inputArea = useRef<HTMLTextAreaElement>(null);
  const followLatest = useRef(true);
  const pendingPreviews = useRef(new Map<string, { idempotencyKey: string; preview: string }>());
  const previewFrame = useRef<number | null>(null);
  const sessionRef = useRef(session);
  sessionRef.current = session;
  const scopeDocumentId = searchParams.get("scopeDocument") || undefined;
  const scopeTitle = searchParams.get("scopeTitle") || scopeDocumentId;
  const inputKey = conversationId ?? "new";
  const content = inputs[inputKey] ?? "";
  const setContent = (value: string) => setInputs((current) => ({ ...current, [inputKey]: value }));

  const conversation = useQuery({
    queryKey: ["conversation", session, conversationId],
    queryFn: ({ signal }) => getConversation(token!, conversationId!, signal),
    enabled: Boolean(token && conversationId),
    refetchInterval: (query) => query.state.data?.active_turn ? 3000 : false,
  });
  const messages = useQuery({
    queryKey: ["messages", session, conversationId],
    queryFn: ({ signal }) => getMessages(token!, conversationId!, signal),
    enabled: Boolean(token && conversationId),
    refetchInterval: conversationId && (drafts[conversationId]?.interrupted || conversation.data?.active_turn) ? 3000 : false,
  });

  const draft = conversationId ? drafts[conversationId] : undefined;
  const submissionError = conversationId ? submissionErrors[conversationId] : undefined;
  const routeNotice = conversationId ? routeNotices[conversationId] : undefined;
  const activeTurn = conversation.data?.active_turn;
  const latestTaskId = messages.data?.items.slice().reverse().find((message) => message.task_id)?.task_id;
  const latestTask = useQuery({
    queryKey: ["task", session, latestTaskId],
    queryFn: ({ signal }) => getTask(token!, latestTaskId!, signal),
    enabled: Boolean(token && latestTaskId && !activeTurn),
  });

  useEffect(() => () => {
    controllers.current.forEach((item) => item.abort());
    controllers.current.clear();
    if (previewFrame.current !== null) cancelAnimationFrame(previewFrame.current);
  }, [session]);

  useEffect(() => setDrawerTaskId(null), [conversationId]);

  useEffect(() => {
    const element = inputArea.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, 200)}px`;
  }, [content]);

  useEffect(() => {
    if (!followLatest.current) return;
    scrollArea.current?.scrollTo({ top: scrollArea.current.scrollHeight, behavior: "smooth" });
  }, [draft?.preview, messages.data?.items.length]);

  useEffect(() => {
    if (!conversationId || !draft?.interrupted || !draft.interruptedAt || conversation.isFetching || messages.isFetching) return;
    const refreshedAfterInterruption = conversation.dataUpdatedAt >= draft.interruptedAt && messages.dataUpdatedAt >= draft.interruptedAt;
    if (refreshedAfterInterruption && !conversation.data?.active_turn) {
      setDrafts((current) => {
        const { [conversationId]: _finished, ...remaining } = current;
        return remaining;
      });
    }
  }, [conversation.data?.active_turn, conversation.dataUpdatedAt, conversation.isFetching, conversationId, draft?.interrupted, draft?.interruptedAt, messages.dataUpdatedAt, messages.isFetching]);

  const refreshConversation = (id: string) => {
    void queryClient.invalidateQueries({ queryKey: ["conversation", session, id] });
    void queryClient.invalidateQueries({ queryKey: ["messages", session, id] });
    void queryClient.invalidateQueries({ queryKey: ["conversations", session] });
  };

  const handleDone = (response: TurnResponse, id: string) => {
    pendingPreviews.current.delete(id);
    setDrafts((current) => {
      const { [id]: _completed, ...remaining } = current;
      return remaining;
    });
    setSubmissionErrors((current) => {
      const { [id]: _completed, ...remaining } = current;
      return remaining;
    });
    if (response.requested_mode === "auto" && response.selected_mode === "controlled_action") {
      setRouteNotices((current) => ({
        ...current,
        [id]: "已识别为研究并创建产物；研究会先执行，写入将在审批后进行。",
      }));
    }
    refreshConversation(id);
  };

  const submit = async (
    retry?: DraftState,
    override?: { content: string; mode: MessageMode; scopeDocumentId?: string },
  ) => {
    const input = (override?.content ?? retry?.content ?? content).trim();
    if (!input || !token || submitting.current || (draft && !retry)) return;
    const activeTurn = conversation.data?.active_turn;
    let requestController: AbortController | null = null;
    const clarification = activeTurn?.status === "waiting_for_user";
    if (activeTurn && !clarification && !retry) {
      toast({ title: "当前轮次仍在执行", detail: "请等待、查看任务详情或在另一会话继续。" });
      return;
    }
    submitting.current = true;
    setIsSubmitting(true);
    let id = conversationId;
    try {
      if (!id) {
        const created = await createConversation(token, titleFor(input));
        id = created.conversation_id;
        const query = searchParams.toString();
        navigate(`/chat/${id}${query ? `?${query}` : ""}`, { replace: true });
        void queryClient.invalidateQueries({ queryKey: ["conversations", session] });
      }
      if (!id) return;
      const targetConversationId = id;
      const idempotencyKey = retry?.idempotencyKey ?? newIdempotencyKey();
      const requestedMode = override?.mode ?? mode;
      const currentSession = session;
      const nextDraft: DraftState = {
        conversationId: targetConversationId,
        idempotencyKey,
        content: input,
        stage: "retrieving",
        preview: "",
        scopeDocumentId: override?.scopeDocumentId ?? retry?.scopeDocumentId ?? scopeDocumentId,
      };
      setDrafts((current) => ({ ...current, [targetConversationId]: nextDraft }));
      setSubmissionErrors((current) => {
        const { [targetConversationId]: _previous, ...remaining } = current;
        return remaining;
      });
      if (!retry) setInputs((current) => ({ ...current, [inputKey]: "", [targetConversationId]: "" }));
      requestController = new AbortController();
      controllers.current.add(requestController);
      const reply = await submitMessage(
        token,
        targetConversationId,
        clarification
          ? {
              kind: "clarification",
              content: input,
              turn_id: activeTurn!.turn_id,
              task_id: activeTurn!.task_id!,
              expected_task_state_version: activeTurn!.state_version,
            }
          : {
              kind: "new",
              content: input,
              mode: requestedMode,
              ...(nextDraft.scopeDocumentId
                ? { scope_document_id: nextDraft.scopeDocumentId }
                : {}),
            },
        idempotencyKey,
        (event) => {
          if (sessionRef.current !== currentSession) return;
          if (event.type === "turn.accepted") {
            setDrafts((current) => current[targetConversationId]?.idempotencyKey === idempotencyKey ? {
              ...current,
              [targetConversationId]: { ...current[targetConversationId], userMessageId: event.data.user_message.message_id },
            } : current);
          } else if (event.type === "status") {
            setDrafts((current) => current[targetConversationId]?.idempotencyKey === idempotencyKey ? {
              ...current,
              [targetConversationId]: { ...current[targetConversationId], stage: event.data.stage },
            } : current);
          } else if (event.type === "answer.preview") {
            pendingPreviews.current.set(targetConversationId, { idempotencyKey, preview: event.data.answer });
            if (previewFrame.current === null) {
              previewFrame.current = requestAnimationFrame(() => {
                previewFrame.current = null;
                const updates = new Map(pendingPreviews.current);
                pendingPreviews.current.clear();
                setDrafts((current) => {
                  let next = current;
                  updates.forEach((update, targetConversationId) => {
                    const item = current[targetConversationId];
                    if (item?.idempotencyKey !== update.idempotencyKey) return;
                    if (next === current) next = { ...current };
                    next[targetConversationId] = { ...item, preview: update.preview };
                  });
                  return next;
                });
              });
            }
          } else if (event.type === "done") {
            handleDone(event.data, targetConversationId);
          } else if (event.type === "error") {
            pendingPreviews.current.delete(targetConversationId);
            setDrafts((current) => {
              if (current[targetConversationId]?.idempotencyKey !== idempotencyKey) return current;
              const { [targetConversationId]: _failed, ...remaining } = current;
              return remaining;
            });
            setSubmissionErrors((current) => ({
              ...current,
              [targetConversationId]: {
                code: event.data.code,
                content: input,
                scopeDocumentId: nextDraft.scopeDocumentId,
              },
            }));
            refreshConversation(targetConversationId);
          }
        },
        requestController.signal,
      );
      if (sessionRef.current !== currentSession) return;
      if (reply.kind === "json") {
        handleDone(reply.data, targetConversationId);
      } else if (!reply.terminal) {
        pendingPreviews.current.delete(targetConversationId);
        setDrafts((current) => current[targetConversationId]?.idempotencyKey === idempotencyKey ? {
          ...current,
          [targetConversationId]: { ...current[targetConversationId], preview: "", interrupted: true, interruptedAt: Date.now() },
        } : current);
        refreshConversation(targetConversationId);
      }
    } catch (cause) {
      if (requestController?.signal.aborted) return;
      if (id) {
        const failedConversationId = id;
        pendingPreviews.current.delete(failedConversationId);
        if (cause instanceof ApiError) {
          setDrafts((current) => {
            const { [failedConversationId]: _failed, ...remaining } = current;
            return remaining;
          });
          setInputs((current) => ({ ...current, [failedConversationId]: input }));
          setSubmissionErrors((current) => ({
            ...current,
            [failedConversationId]: {
              code: cause.code,
              content: input,
              scopeDocumentId: override?.scopeDocumentId ?? retry?.scopeDocumentId ?? scopeDocumentId,
            },
          }));
        } else {
          setDrafts((current) => current[failedConversationId] ? {
            ...current,
            [failedConversationId]: {
              ...current[failedConversationId],
              preview: "",
              interrupted: true,
              interruptedAt: Date.now(),
            },
          } : current);
        }
      }
      if (id) refreshConversation(id);
      toast({
        title: cause instanceof ApiError ? "服务器未接受请求" : "连接已中断",
        detail: cause instanceof ApiError
          ? submissionErrorMessage(cause.code)
          : "请先读取服务器当前状态，再使用原请求恢复。",
      });
    } finally {
      if (requestController) controllers.current.delete(requestController);
      submitting.current = false;
      setIsSubmitting(false);
    }
  };

  const onSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    void submit();
  };
  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey && !isComposing(event)) {
      event.preventDefault();
      void submit();
    }
  };

  const clarification = activeTurn?.status === "waiting_for_user";
  const blocked = Boolean(draft || (activeTurn && !clarification));
  const knownMessageIds = new Set(messages.data?.items.map((item) => item.message_id));

  return (
    <section className="chat-page">
      <header className="chat-header">
        <div><p className="eyebrow">授权知识库</p><h1>{conversation.data?.title || "开始新的对话"}</h1></div>
        {activeTurn?.task_id && <TaskCard taskId={activeTurn.task_id} onOpen={() => setDrawerTaskId(activeTurn.task_id)} />}
      </header>
      <div
        className="chat-layout"
        ref={scrollArea}
        onScroll={(event) => {
          const element = event.currentTarget;
          const nearBottom = element.scrollHeight - element.scrollTop - element.clientHeight < 96;
          followLatest.current = nearBottom;
          setShowLatest(!nearBottom);
        }}
      >
        <div className="message-column">
          {!conversationId && !draft && (
            <div className="chat-welcome"><p className="eyebrow">从问题开始</p><h2>基于你有权读取的资料回答</h2><p>回答会附上可核对的资料来源。</p><div className="prompt-examples">{["总结知识库中的核心制度", "查找这份资料里的关键数据", "比较相关文档中的不同规定"].map((example) => <button type="button" key={example} onClick={() => setContent(example)}>{example}</button>)}</div></div>
          )}
          {conversation.isLoading && <><div className="skeleton h-24 w-3/4" /><div className="skeleton h-32 w-2/3" /></>}
          {conversation.isError && <div className="stream-notice" role="alert"><strong>无法读取此会话</strong><p>会话可能已删除、当前身份无权访问，或网络暂时不可用。</p><button className="text-button" type="button" onClick={() => void conversation.refetch()}>重新加载</button></div>}
          {messages.isError && !conversation.isError && <div className="stream-notice" role="alert"><strong>消息加载失败</strong><p>会话仍然保留，可以重新读取消息。</p><button className="text-button" type="button" onClick={() => void messages.refetch()}>重新加载</button></div>}
          {messages.data?.items.map((message) => <MessageBubble key={message.message_id} message={message} onOpenTask={setDrawerTaskId} />)}
          {draft && <DraftBubble draft={draft} knownMessageIds={knownMessageIds} />}
          {routeNotice && <div className="controlled-route-notice" role="status">{routeNotice}</div>}
          {submissionError && (
            <div className="stream-notice" role="alert">
              <strong>本次请求未执行</strong>
              <p>{submissionErrorMessage(submissionError.code)}</p>
              {submissionError.code === "controlled_action_explicit_mode_required" && (
                <button
                  className="text-button"
                  type="button"
                  onClick={() => {
                    setMode("controlled_action");
                    void submit(undefined, {
                      content: submissionError.content,
                      mode: "controlled_action",
                      scopeDocumentId: submissionError.scopeDocumentId,
                    });
                  }}
                >
                  切换到受控操作并重新提交
                </button>
              )}
            </div>
          )}
          {!draft && latestTask.data?.status === "failed" && latestTaskId && (
            <div className="stream-notice" role="alert">
              <strong>本次研究未完成</strong>
              <p>执行过程遇到问题，可以查看任务详情后调整问题并重试。</p>
              <button className="text-button" type="button" onClick={() => setDrawerTaskId(latestTaskId)}>查看失败详情</button>
            </div>
          )}
          {clarification && <div className="clarification-callout"><strong>需要补充信息</strong><p>请通过下方输入框提交服务器请求的补充信息。</p></div>}
        </div>
        {showLatest && <button className="back-to-latest" type="button" onClick={() => { followLatest.current = true; setShowLatest(false); scrollArea.current?.scrollTo({ top: scrollArea.current.scrollHeight, behavior: "smooth" }); }}>回到最新</button>}
        {drawerTaskId && <><button className="drawer-scrim" type="button" aria-label="关闭任务详情" onClick={() => setDrawerTaskId(null)} /><TaskDrawer taskId={drawerTaskId} onClose={() => setDrawerTaskId(null)} /></>}
      </div>
      <form className="composer" onSubmit={onSubmit}>
        {!clarification && scopeDocumentId && (
          <div className="scope-chip" role="status">
            <span>仅此文档：{scopeTitle}</span>
            <button
              type="button"
              onClick={() => {
                const next = new URLSearchParams(searchParams);
                next.delete("scopeDocument");
                next.delete("scopeTitle");
                setSearchParams(next, { replace: true });
              }}
              disabled={blocked}
              aria-label="恢复全库检索"
            >
              恢复全库
            </button>
          </div>
        )}
        <div className="composer-topline">
          {!clarification && (
            <label className="mode-select"><span>模式</span><select value={mode} onChange={(event) => setMode(event.target.value as MessageMode)} disabled={blocked}>{Object.entries(modeLabels).map(([value, label]) => <option value={value} key={value}>{label}</option>)}</select></label>
          )}
          <span className="composer-hint">{clarification ? "将作为当前任务的补充信息提交" : modeHints[mode]}</span>
        </div>
        <label className="sr-only" htmlFor="chat-input">{clarification ? "补充信息" : "问题"}</label>
        <textarea
          id="chat-input"
          ref={inputArea}
          value={content}
          maxLength={8000}
          onChange={(event) => setContent(event.target.value)}
          onKeyDown={onKeyDown}
          disabled={Boolean(activeTurn && !clarification) || isSubmitting}
          placeholder={clarification ? "补充任务所需的信息" : blocked ? "当前会话正在执行" : "输入问题，Enter 发送，Shift+Enter 换行"}
          rows={3}
        />
        <div className="composer-actions">
          <small>{content.length}/8000</small>
          {draft?.interrupted ? (
            <button className="secondary-button" type="button" onClick={() => void submit(draft)}>使用原请求重试</button>
          ) : (
            <button className="primary-button" type="submit" disabled={blocked || isSubmitting || !content.trim()}>{isSubmitting ? "正在提交" : clarification ? "提交补充" : "发送"}</button>
          )}
        </div>
      </form>
    </section>
  );
}
