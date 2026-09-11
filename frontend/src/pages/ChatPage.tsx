import {
  useEffect,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
} from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import ReactMarkdown from "react-markdown";

import {
  ApiError,
  createConversation,
  getConversation,
  getMessages,
  newIdempotencyKey,
  submitMessage,
} from "../api/client";
import type { ConversationMessage, MessageMode, TurnResponse } from "../api/types";
import { useAuth } from "../app/auth";
import { CitationList, TaskCard, TaskDrawer } from "../components/task-drawer";
import { useToast } from "../components/feedback";

type DraftState = {
  conversationId: string;
  idempotencyKey: string;
  content: string;
  stage: string;
  preview: string;
  userMessageId?: string;
  error?: string;
  interrupted?: boolean;
  scopeDocumentId?: string;
};

const modeLabels: Record<MessageMode, string> = {
  auto: "自动",
  answer: "知识问答",
  research: "深入研究",
  controlled_action: "受控操作",
};

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
        <div className="markdown-content"><ReactMarkdown skipHtml>{message.content ?? ""}</ReactMarkdown></div>
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
      {!draft.error && !draft.interrupted && (
        <article className="message-bubble from-assistant preview-message">
          <div className="message-meta"><span>知识工作台</span><small>{draft.stage === "validating" ? "正在校验来源" : "生成中，尚未校验"}</small></div>
          {draft.preview ? <div className="markdown-content"><ReactMarkdown skipHtml>{draft.preview}</ReactMarkdown></div> : <div className="skeleton h-5 w-32" />}
        </article>
      )}
      {(draft.error || draft.interrupted) && (
        <article className="stream-notice" role="status">
          <strong>{draft.interrupted ? "连接已中断" : "本次回答未完成"}</strong>
          <p>{draft.interrupted ? "正在从服务器恢复会话状态；不会以新请求重新生成。" : `服务器返回：${draft.error}`}</p>
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
  const [content, setContent] = useState("");
  const [mode, setMode] = useState<MessageMode>("auto");
  const [drafts, setDrafts] = useState<Record<string, DraftState>>({});
  const [drawerTaskId, setDrawerTaskId] = useState<string | null>(null);
  const controllers = useRef(new Set<AbortController>());
  const pendingPreviews = useRef(new Map<string, { idempotencyKey: string; preview: string }>());
  const previewFrame = useRef<number | null>(null);
  const sessionRef = useRef(session);
  sessionRef.current = session;
  const scopeDocumentId = searchParams.get("scopeDocument") || undefined;
  const scopeTitle = searchParams.get("scopeTitle") || scopeDocumentId;

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
  });

  const draft = conversationId ? drafts[conversationId] : undefined;

  useEffect(() => () => {
    controllers.current.forEach((item) => item.abort());
    controllers.current.clear();
    if (previewFrame.current !== null) cancelAnimationFrame(previewFrame.current);
  }, [session]);

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
    refreshConversation(id);
    if (response.task?.task_id) setDrawerTaskId(response.task.task_id);
  };

  const submit = async (retry?: DraftState) => {
    const input = (retry?.content ?? content).trim();
    if (!input || !token || (draft && !retry)) return;
    const activeTurn = conversation.data?.active_turn;
    let requestController: AbortController | null = null;
    const clarification = activeTurn?.status === "waiting_for_user";
    if (activeTurn && !clarification && !retry) {
      toast({ title: "当前轮次仍在执行", detail: "请等待、查看任务详情或在另一会话继续。" });
      return;
    }
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
      const currentSession = session;
      const nextDraft: DraftState = {
        conversationId: targetConversationId,
        idempotencyKey,
        content: input,
        stage: "retrieving",
        preview: "",
        scopeDocumentId: retry?.scopeDocumentId ?? scopeDocumentId,
      };
      setDrafts((current) => ({ ...current, [targetConversationId]: nextDraft }));
      if (!retry) setContent("");
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
              mode,
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
            setDrafts((current) => current[targetConversationId]?.idempotencyKey === idempotencyKey ? {
              ...current,
              [targetConversationId]: { ...current[targetConversationId], preview: "", error: event.data.code },
            } : current);
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
          [targetConversationId]: { ...current[targetConversationId], preview: "", interrupted: true },
        } : current);
        refreshConversation(targetConversationId);
      }
    } catch (cause) {
      if (requestController?.signal.aborted) return;
      const detail = cause instanceof ApiError ? cause.code : "network_unavailable";
      if (id) {
        const failedConversationId = id;
        pendingPreviews.current.delete(failedConversationId);
        setDrafts((current) => current[failedConversationId] ? { ...current, [failedConversationId]: { ...current[failedConversationId], preview: "", error: detail } } : current);
      }
      if (id) refreshConversation(id);
      toast({ title: "消息未发送完成", detail: "请先读取服务器当前状态，再使用原请求重试。" });
    } finally {
      if (requestController) controllers.current.delete(requestController);
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

  const activeTurn = conversation.data?.active_turn;
  const clarification = activeTurn?.status === "waiting_for_user";
  const blocked = Boolean(draft || (activeTurn && !clarification));
  const knownMessageIds = new Set(messages.data?.items.map((item) => item.message_id));

  return (
    <section className="chat-page">
      <header className="chat-header">
        <div><p className="eyebrow">授权知识库</p><h1>{conversation.data?.title || "开始新的对话"}</h1></div>
        {activeTurn?.task_id && <TaskCard taskId={activeTurn.task_id} onOpen={() => setDrawerTaskId(activeTurn.task_id)} />}
      </header>
      <div className="chat-layout">
        <div className="message-column">
          {!conversationId && !draft && (
            <div className="chat-welcome"><p className="eyebrow">从问题开始</p><h2>基于你有权读取的资料回答</h2><p>普通问题将流式展示草稿，最终内容和来源须经服务器校验后才会保存。</p></div>
          )}
          {conversation.isLoading && <><div className="skeleton h-24 w-3/4" /><div className="skeleton h-32 w-2/3" /></>}
          {messages.data?.items.map((message) => <MessageBubble key={message.message_id} message={message} onOpenTask={setDrawerTaskId} />)}
          {draft && <DraftBubble draft={draft} knownMessageIds={knownMessageIds} />}
          {clarification && <div className="clarification-callout"><strong>需要补充信息</strong><p>请通过下方输入框提交服务器请求的补充信息。</p></div>}
        </div>
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
          <span className="composer-hint">{clarification ? "将作为当前任务的补充信息提交" : mode === "answer" ? "知识问答优先流式返回" : "服务端将决定最适合的执行路径"}</span>
        </div>
        <label className="sr-only" htmlFor="chat-input">{clarification ? "补充信息" : "问题"}</label>
        <textarea
          id="chat-input"
          value={content}
          maxLength={8000}
          onChange={(event) => setContent(event.target.value)}
          onKeyDown={onKeyDown}
          disabled={blocked}
          placeholder={clarification ? "补充任务所需的信息" : blocked ? "当前会话正在执行" : "输入问题，Enter 发送，Shift+Enter 换行"}
          rows={3}
        />
        <div className="composer-actions">
          <small>{content.length}/8000</small>
          {draft?.error || draft?.interrupted ? (
            <button className="secondary-button" type="button" onClick={() => void submit(draft)}>使用原请求重试</button>
          ) : (
            <button className="primary-button" type="submit" disabled={blocked || !content.trim()}>{clarification ? "提交补充" : "发送"}</button>
          )}
        </div>
      </form>
    </section>
  );
}
