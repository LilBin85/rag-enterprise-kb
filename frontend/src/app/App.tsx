import { useState } from "react";
import { Navigate, NavLink, Outlet, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { useInfiniteQuery, useMutation, useQueryClient } from "@tanstack/react-query";

import { ApiError, deleteConversation, listConversations } from "../api/client";
import { useAuth } from "./auth";
import { Hint, useToast } from "../components/feedback";
import { ApprovalsPage } from "../pages/ApprovalsPage";
import { ChatPage } from "../pages/ChatPage";
import { DocumentsPage } from "../pages/DocumentsPage";

function ConnectPage() {
  const { connect, checking, error } = useAuth();
  const [token, setToken] = useState("");
  const [message, setMessage] = useState<string | null>(null);

  const submit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setMessage(null);
    try {
      await connect(token);
    } catch (cause) {
      setMessage(cause instanceof ApiError ? "令牌无效或当前服务不可用。" : "无法验证演示身份。")
    }
  };

  return (
    <main className="connect-page">
      <section className="connect-panel">
        <p className="eyebrow">本地演示环境</p>
        <h1>连接知识工作台</h1>
        <p>输入已配置的演示令牌。令牌只保留在当前浏览器标签页，用于调用真实后端。</p>
        <form className="stack-form" onSubmit={submit}>
          <label>
            <span>演示令牌</span>
            <input
              autoComplete="off"
              value={token}
              onChange={(event) => setToken(event.target.value)}
              placeholder="粘贴 .env 中对应角色的令牌"
              required
            />
            <small>管理员可上传和提问；审批请使用独立 approver 令牌。</small>
          </label>
          {(message || error) && <p className="form-error" role="alert">{message ?? error}</p>}
          <button className="primary-button" disabled={checking} type="submit">
            {checking ? "正在验证" : "连接身份"}
          </button>
        </form>
      </section>
      <aside className="connect-note">
        <span className="connect-mark">KB</span>
        <p>文档摄取、授权检索与任务审批均由服务端裁决。界面只展示已验证身份可访问的内容。</p>
      </aside>
    </main>
  );
}

function ConversationRail({ onNavigate }: { onNavigate: () => void }) {
  const { token, session } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const toast = useToast();
  const client = useQueryClient();
  const conversations = useInfiniteQuery({
    queryKey: ["conversations", session],
    initialPageParam: null as string | null,
    queryFn: ({ pageParam, signal }) => listConversations(token!, pageParam, signal),
    getNextPageParam: (page) => page.next_cursor,
    enabled: Boolean(token),
  });
  const remove = useMutation({
    mutationFn: (conversationId: string) => deleteConversation(token!, conversationId),
    onSuccess: (_, conversationId) => {
      void client.invalidateQueries({ queryKey: ["conversations", session] });
      if (location.pathname.endsWith(`/${conversationId}`)) navigate("/chat");
      toast({ title: "会话已请求删除", detail: "删除将在服务器完成状态校验后生效。" });
    },
    onError: (cause) => {
      toast({
        title: "无法删除会话",
        detail: cause instanceof ApiError && cause.status === 409 ? "会话仍有活动任务，请先等待或取消。" : "请稍后重试。",
      });
    },
  });
  const rows = conversations.data?.pages.flatMap((page) => page.items) ?? [];

  return (
    <section className="conversation-rail">
      <div className="rail-heading">
        <span>会话</span>
        <Hint label="新建对话">
          <button className="new-chat-button" type="button" onClick={() => { navigate("/chat"); onNavigate(); }}>新建</button>
        </Hint>
      </div>
      <div className="conversation-list">
        {conversations.isLoading && <><div className="skeleton h-10 w-full" /><div className="skeleton h-10 w-full" /></>}
        {rows.map((conversation) => {
          const selected = location.pathname.endsWith(`/${conversation.conversation_id}`);
          return (
            <div className={`conversation-row ${selected ? "selected" : ""}`} key={conversation.conversation_id}>
              <button
                className="conversation-link"
                type="button"
                onClick={() => { navigate(`/chat/${conversation.conversation_id}`); onNavigate(); }}
              >
                <span>{conversation.title || "未命名对话"}</span>
                {conversation.active_turn && <small>{conversation.active_turn.status}</small>}
              </button>
              <button
                className="delete-conversation"
                type="button"
                aria-label="删除会话"
                disabled={remove.isPending}
                onClick={() => window.confirm("确认删除此会话？活动任务需要先由服务器结束或取消。") && remove.mutate(conversation.conversation_id)}
              >
                删除
              </button>
            </div>
          );
        })}
        {!conversations.isLoading && !rows.length && <p className="empty-copy">还没有会话。发送第一条问题后将自动创建。</p>}
      </div>
      {conversations.hasNextPage && (
        <button className="text-button" type="button" disabled={conversations.isFetchingNextPage} onClick={() => conversations.fetchNextPage()}>
          {conversations.isFetchingNextPage ? "正在加载" : "加载更多"}
        </button>
      )}
    </section>
  );
}

function Workspace() {
  const { identity, disconnect, hasPermission } = useAuth();
  const [menuOpen, setMenuOpen] = useState(false);
  const location = useLocation();
  const nav = [
    { to: "/chat", label: "对话", visible: true },
    { to: "/documents", label: "文档", visible: hasPermission("document:read") },
    { to: "/approvals", label: "审批", visible: hasPermission("task:approve") },
  ];
  return (
    <div className="workspace-shell">
      <button className="mobile-menu-button" type="button" onClick={() => setMenuOpen(true)}>菜单</button>
      {menuOpen && <button className="sidebar-scrim" type="button" aria-label="关闭菜单" onClick={() => setMenuOpen(false)} />}
      <aside className={`sidebar ${menuOpen ? "open" : ""}`}>
        <div className="brand"><span>KB</span><strong>知识工作台</strong></div>
        <nav className="primary-nav" aria-label="主要导航">
          {nav.filter((item) => item.visible).map((item) => (
            <NavLink key={item.to} to={item.to} className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`} onClick={() => setMenuOpen(false)}>
              {item.label}
            </NavLink>
          ))}
        </nav>
        <ConversationRail onNavigate={() => setMenuOpen(false)} />
        <div className="identity-strip">
          <span className="identity-dot" />
          <div><strong>{identity?.user_id}</strong><small>{identity?.tenant_id}</small></div>
          <button className="text-button" type="button" onClick={disconnect}>切换</button>
        </div>
      </aside>
      <main className={`workspace-main ${location.pathname === "/chat" ? "chat-empty" : ""}`}>
        <Outlet />
      </main>
    </div>
  );
}

function AuthGate() {
  const { identity, checking, session } = useAuth();
  if (checking) return <main className="connect-page"><div className="connect-panel"><div className="skeleton h-8 w-40" /><div className="skeleton h-20 w-full" /></div></main>;
  if (!identity) return <ConnectPage />;
  return <Workspace key={session} />;
}

export function App() {
  return (
    <Routes>
      <Route element={<AuthGate />}>
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/chat/:conversationId" element={<ChatPage />} />
        <Route path="/documents" element={<DocumentsPage />} />
        <Route path="/approvals" element={<ApprovalsPage />} />
        <Route path="*" element={<Navigate to="/chat" replace />} />
      </Route>
    </Routes>
  );
}
