import type {
  Approval,
  ApprovalDecision,
  ApprovalStatus,
  ConversationList,
  ConversationDeleteResponse,
  ConversationMessages,
  ConversationStreamEvent,
  ConversationSummary,
  DocumentDetail,
  DocumentList,
  DocumentUpload,
  DocumentUploadRecovery,
  Identity,
  IngestionJob,
  MessageMode,
  Task,
  TaskEvent,
  TaskResult,
  TurnResponse,
} from "./types";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message?: string,
    readonly recovery?: DocumentUploadRecovery,
  ) {
    super(message ?? code);
    this.name = "ApiError";
  }
}

function headers(token: string, extra?: HeadersInit): Headers {
  const value = new Headers(extra);
  value.set("Authorization", `Bearer ${token}`);
  return value;
}

async function errorFor(response: Response): Promise<ApiError> {
  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    return new ApiError(response.status, "request_failed");
  }
  if (typeof payload === "object" && payload !== null && "detail" in payload) {
    const detail = payload.detail;
    if (typeof detail === "object" && detail !== null && "code" in detail) {
      const objectDetail = detail as Record<string, unknown>;
      const code = objectDetail.code;
      if (typeof code === "string") {
        return new ApiError(
          response.status,
          code,
          typeof objectDetail.message === "string" ? objectDetail.message : undefined,
          documentRecovery(objectDetail.recovery),
        );
      }
    }
    if (typeof detail === "string") {
      return new ApiError(response.status, "request_failed", detail);
    }
  }
  return new ApiError(response.status, "request_failed");
}

function documentRecovery(value: unknown): DocumentUploadRecovery | undefined {
  if (typeof value !== "object" || value === null) return undefined;
  const recovery = value as Record<string, unknown>;
  const fields = [
    "document_id",
    "version_id",
    "ingestion_job_id",
    "lifecycle_status",
    "ingestion_status",
  ];
  if (!fields.every((field) => typeof recovery[field] === "string")) return undefined;
  return recovery as unknown as DocumentUploadRecovery;
}

async function request<T>(
  path: string,
  token: string,
  init: RequestInit = {},
): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: headers(token, init.headers),
  });
  if (!response.ok) {
    throw await errorFor(response);
  }
  return response.json() as Promise<T>;
}

export function newIdempotencyKey(): string {
  return crypto.randomUUID();
}

export function getMe(token: string, signal?: AbortSignal): Promise<Identity> {
  return request<Identity>("/v1/me", token, { signal });
}

export function listConversations(
  token: string,
  cursor?: string | null,
  signal?: AbortSignal,
): Promise<ConversationList> {
  const query = new URLSearchParams({ limit: "20" });
  if (cursor) query.set("cursor", cursor);
  return request<ConversationList>(`/v1/conversations?${query}`, token, { signal });
}

export function createConversation(
  token: string,
  title: string,
): Promise<ConversationSummary> {
  return request<ConversationSummary>("/v1/conversations", token, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
}

export function getConversation(
  token: string,
  conversationId: string,
  signal?: AbortSignal,
): Promise<ConversationSummary> {
  return request<ConversationSummary>(`/v1/conversations/${conversationId}`, token, {
    signal,
  });
}

export async function getMessages(
  token: string,
  conversationId: string,
  signal?: AbortSignal,
): Promise<ConversationMessages> {
  const items: ConversationMessages["items"] = [];
  let afterSequence = 0;
  while (true) {
    const page = await request<ConversationMessages>(
      `/v1/conversations/${conversationId}/messages?limit=100&after_sequence=${afterSequence}`,
      token,
      { signal },
    );
    items.push(...page.items);
    if (page.items.length < 100) return { items };
    const nextSequence = Math.max(...page.items.map((item) => item.sequence));
    if (nextSequence <= afterSequence) return { items };
    afterSequence = nextSequence;
  }
}

export function deleteConversation(token: string, conversationId: string): Promise<ConversationDeleteResponse> {
  return request<ConversationDeleteResponse>(`/v1/conversations/${conversationId}`, token, {
    method: "DELETE",
  });
}

type SubmitMessage = {
  kind: "new";
  content: string;
  mode: MessageMode;
  scope_document_id?: string;
} | {
  kind: "clarification";
  content: string;
  turn_id: string;
  task_id: string;
  expected_task_state_version: number;
};

export async function submitMessage(
  token: string,
  conversationId: string,
  message: SubmitMessage,
  idempotencyKey: string,
  onEvent: (event: ConversationStreamEvent) => void,
  signal?: AbortSignal,
): Promise<{ kind: "stream"; terminal: boolean } | { kind: "json"; data: TurnResponse }> {
  const response = await fetch(`/v1/conversations/${conversationId}/messages`, {
    method: "POST",
    signal,
    headers: headers(token, {
      Accept: "text/event-stream, application/json",
      "Content-Type": "application/json",
      "Idempotency-Key": idempotencyKey,
    }),
    body: JSON.stringify(message),
  });
  if (!response.ok) {
    throw await errorFor(response);
  }
  const contentType = response.headers.get("content-type") ?? "";
  if (!contentType.includes("text/event-stream")) {
    return { kind: "json", data: (await response.json()) as TurnResponse };
  }
  return { kind: "stream", terminal: await decodeSse(response, onEvent) };
}

async function decodeSse(
  response: Response,
  onEvent: (event: ConversationStreamEvent) => void,
): Promise<boolean> {
  if (!response.body) throw new ApiError(response.status, "stream_unavailable");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let terminal = false;

  const consume = (block: string) => {
    let event = "message";
    const data: string[] = [];
    for (const line of block.split(/\r?\n/)) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
    }
    if (!data.length) return;
    try {
      const parsed = JSON.parse(data.join("\n")) as ConversationStreamEvent["data"];
      const item = { type: event, data: parsed } as ConversationStreamEvent;
      onEvent(item);
      terminal ||= item.type === "done" || item.type === "error";
    } catch {
      throw new ApiError(response.status, "stream_event_invalid");
    }
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      const blocks = buffer.split(/\r?\n\r?\n/);
      buffer = blocks.pop() ?? "";
      for (const block of blocks) consume(block);
      if (done) break;
    }
    if (buffer.trim()) consume(buffer);
  } finally {
    reader.releaseLock();
  }
  return terminal;
}

export function listDocuments(
  token: string,
  cursor?: string | null,
  signal?: AbortSignal,
): Promise<DocumentList> {
  const query = new URLSearchParams({ limit: "20" });
  if (cursor) query.set("cursor", cursor);
  return request<DocumentList>(`/v1/documents?${query}`, token, { signal });
}

export function getDocument(
  token: string,
  documentId: string,
  signal?: AbortSignal,
): Promise<DocumentDetail> {
  return request<DocumentDetail>(`/v1/documents/${documentId}`, token, { signal });
}

export function getIngestionJob(
  token: string,
  jobId: string,
  signal?: AbortSignal,
): Promise<IngestionJob> {
  return request<IngestionJob>(`/v1/ingestion-jobs/${jobId}`, token, { signal });
}

export function getDocumentIngestionJobs(
  token: string,
  documentId: string,
  signal?: AbortSignal,
): Promise<IngestionJob[]> {
  return request<IngestionJob[]>(`/v1/documents/${documentId}/ingestion-jobs`, token, {
    signal,
  });
}

export async function uploadDocument(
  token: string,
  values: { documentId: string; title: string; department: string; file: File; versionOf?: string },
  idempotencyKey: string,
): Promise<DocumentUpload> {
  const body = new FormData();
  body.set("file", values.file);
  body.set(
    "metadata",
    JSON.stringify({
      document_id: values.documentId,
      title: values.title,
      department: values.department,
    }),
  );
  const path = values.versionOf
    ? `/v1/documents/${values.versionOf}/versions`
    : "/v1/documents";
  const response = await fetch(path, {
    method: "POST",
    headers: headers(token, { "Idempotency-Key": idempotencyKey }),
    body,
  });
  if (!response.ok) throw await errorFor(response);
  return response.json() as Promise<DocumentUpload>;
}

export function getTask(token: string, taskId: string, signal?: AbortSignal): Promise<Task> {
  return request<Task>(`/v1/tasks/${taskId}`, token, { signal });
}

export function getTaskEvents(
  token: string,
  taskId: string,
  signal?: AbortSignal,
): Promise<TaskEvent[]> {
  return request<TaskEvent[]>(`/v1/tasks/${taskId}/events`, token, { signal });
}

export function getTaskResult(
  token: string,
  taskId: string,
  signal?: AbortSignal,
): Promise<TaskResult> {
  return request<TaskResult>(`/v1/tasks/${taskId}/result`, token, { signal });
}

export function cancelTask(token: string, taskId: string, stateVersion: number): Promise<Task> {
  return request<Task>(`/v1/tasks/${taskId}/cancel`, token, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ expected_state_version: stateVersion }),
  });
}

export async function getArtifactContent(
  token: string,
  taskId: string,
  artifactId: string,
): Promise<Blob> {
  const response = await fetch(`/v1/tasks/${taskId}/artifacts/${artifactId}/content`, {
    headers: headers(token),
  });
  if (!response.ok) throw await errorFor(response);
  return response.blob();
}

export function listApprovals(
  token: string,
  status: ApprovalStatus,
  signal?: AbortSignal,
): Promise<Approval[]> {
  return request<Approval[]>(`/v1/approvals?status=${status}`, token, { signal });
}

export function getApproval(
  token: string,
  approvalId: string,
  signal?: AbortSignal,
): Promise<Approval> {
  return request<Approval>(`/v1/approvals/${approvalId}`, token, { signal });
}

export function decideApproval(
  token: string,
  approvalId: string,
  values: { expected_task_state_version: number; decision: "approve" | "reject"; reason?: string },
): Promise<ApprovalDecision> {
  return request<ApprovalDecision>(`/v1/approvals/${approvalId}/decision`, token, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(values),
  });
}
