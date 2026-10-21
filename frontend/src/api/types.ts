export type Permission =
  | "knowledge:read"
  | "document:read"
  | "document:write"
  | "task:create"
  | "task:read"
  | "task:approve"
  | "artifact:write"
  | "admin";

export interface Identity {
  tenant_id: string;
  user_id: string;
  roles: string[];
  permissions: Permission[];
}

export interface ConversationSummary {
  conversation_id: string;
  title: string | null;
  state_version: number;
  status: string;
  created_at: string;
  updated_at: string;
  active_turn: ActiveTurn | null;
}

export interface ActiveTurn {
  turn_id: string;
  task_id: string | null;
  status: string;
  requested_mode: MessageMode;
  selected_mode: "answer" | "research" | "controlled_action";
  state_version: number;
}

export interface ConversationList {
  items: ConversationSummary[];
  next_cursor: string | null;
}

export interface ConversationDeleteResponse {
  conversation_id: string;
  status: "deleting";
}

export interface Citation {
    document_id: string;
    document_version: number;
    source_file_name: string;
    source_locator: string;
    chunk_id: string;
    document_title?: string | null;
}

export interface ArtifactSummary {
  artifact_id: string;
  artifact_type: string;
  display_name?: string;
  file_extension?: "md" | "json";
  content_hash: string;
  status: "ready";
}

export interface ConversationMessage {
  message_id: string;
  turn_id: string | null;
  task_id: string | null;
  sequence: number;
  role: "user" | "assistant";
  kind: string;
  content: string | null;
  visibility: "visible" | "redacted";
  redaction_reason: string | null;
  citations: Citation[];
  artifacts: ArtifactSummary[];
  created_at: string;
}

export interface ConversationMessages {
  items: ConversationMessage[];
}

export type MessageMode = "auto" | "answer" | "research" | "controlled_action";

export interface TaskRef {
  task_id: string;
  status: string;
  state_version: number;
  result_available: boolean;
}

export interface TurnResponse {
  conversation_id: string;
  conversation_state_version: number;
  turn_id: string;
  user_message: { message_id: string; sequence: number };
  requested_mode: MessageMode;
  selected_mode: "answer" | "research" | "controlled_action";
  route_reason_code: string;
  turn_status: string;
  task: TaskRef | null;
  assistant_message: ConversationMessage | null;
  idempotent_replay: boolean;
}

export interface DocumentListItem {
  document_id: string;
  title: string;
  department: string;
  status: string;
}

export interface DocumentList {
  items: DocumentListItem[];
  next_cursor: string | null;
}

export interface DocumentVersion {
  version_id: string;
  version: number;
  lifecycle_status: string;
}

export interface DocumentDetail extends DocumentListItem {
  versions: DocumentVersion[];
}

export interface IngestionJob {
  job_id: string;
  version_id: string;
  status: string;
  attempts: number;
  error_code: string | null;
}

export interface DocumentUpload {
  document_id: string;
  version_id: string;
  version: number;
  ingestion_job_id: string;
  lifecycle_status: string;
  ingestion_status: string;
  outcome: string;
}

export interface DocumentUploadRecovery {
  document_id: string;
  version_id: string;
  ingestion_job_id: string;
  lifecycle_status: string;
  ingestion_status: string;
}

export interface Task {
  task_id: string;
  conversation_id: string | null;
  turn_id: string | null;
  goal: string;
  task_mode: string;
  status: string;
  state_version: number;
  interrupt_kind: string | null;
  interrupt_summary: string | null;
  result_available: boolean;
  cancelled: boolean;
}

export interface TaskEvent {
  sequence: number;
  event_type: string;
  stage: string;
  status: string;
  counts: Record<string, unknown>;
  token_usage: number | null;
  error_code: string | null;
  created_at: string;
}

export interface TaskResult {
  task_id: string;
  result_kind: "research" | "controlled_action";
  status: "succeeded" | "refused";
  answer: string;
  refusal_code: string | null;
  citations: Citation[];
  artifacts: ArtifactSummary[];
  stop_reason: string;
  created_at: string;
}

export type ApprovalStatus = "pending" | "approved" | "rejected" | "expired" | "executed";

export interface Approval {
  approval_id: string;
  task_id: string;
  task_state_version: number;
  summary: string | null;
  plan_revision: number;
  tool_name: string;
  tool_version: string;
  expires_at: string;
  status: ApprovalStatus;
  approved_by: string | null;
  approved_at: string | null;
  rejected_by: string | null;
  rejected_at: string | null;
}

export interface ResumeCommand {
  command_id: string;
  status: "accepted";
}

export interface ApprovalDecision extends Approval {
  resume_command: ResumeCommand | null;
}

export interface StreamAccepted {
  conversation_id: string;
  turn_id: string;
  task_id: string | null;
  user_message: { message_id: string; sequence: number };
}

export type ConversationStreamEvent =
  | { type: "turn.accepted"; data: StreamAccepted }
  | { type: "status"; data: { stage: string } }
  | { type: "answer.preview"; data: { answer: string } }
  | { type: "done"; data: TurnResponse }
  | { type: "error"; data: { code: string; refresh_required?: boolean } };
