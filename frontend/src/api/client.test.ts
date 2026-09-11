import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, submitMessage, uploadDocument } from "./client";

afterEach(() => vi.unstubAllGlobals());

describe("submitMessage", () => {
  it("decodes fragmented Chinese SSE snapshots and combined terminal events", async () => {
    const text = [
      "event: turn.accepted\ndata: {\"conversation_id\":\"c-1\",\"turn_id\":\"t-1\",\"task_id\":\"task-1\",\"user_message\":{\"message_id\":\"m-1\",\"sequence\":1}}\n\n",
      "event: answer.preview\ndata: {\"answer\":\"第一段草稿\"}\n\n",
      "event: done\ndata: {\"conversation_id\":\"c-1\",\"conversation_state_version\":2,\"turn_id\":\"t-1\",\"user_message\":{\"message_id\":\"m-1\",\"sequence\":1},\"requested_mode\":\"answer\",\"selected_mode\":\"answer\",\"route_reason_code\":\"test\",\"turn_status\":\"succeeded\",\"task\":null,\"assistant_message\":null,\"idempotent_replay\":false}\n\n",
    ].join("");
    const bytes = new TextEncoder().encode(text);
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(bytes.slice(0, 151));
        controller.enqueue(bytes.slice(151, 159));
        controller.enqueue(bytes.slice(159));
        controller.close();
      },
    });
    const fetchMock = vi.fn().mockResolvedValue(new Response(stream, {
      status: 200,
      headers: { "content-type": "text/event-stream" },
    }));
    vi.stubGlobal("fetch", fetchMock);
    const events: string[] = [];

    const result = await submitMessage(
      "demo-token",
      "c-1",
      { kind: "new", content: "问题", mode: "answer" },
      "stable-key",
      (event) => events.push(event.type === "answer.preview" ? event.data.answer : event.type),
    );

    expect(result).toEqual({ kind: "stream", terminal: true });
    expect(events).toEqual(["turn.accepted", "第一段草稿", "done"]);
    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(new Headers(init.headers).get("Idempotency-Key")).toBe("stable-key");
    expect(new Headers(init.headers).get("Accept")).toContain("text/event-stream");
  });

  it("keeps document upload recovery details without exposing server internals", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({
      detail: {
        code: "document_already_exists",
        recovery: {
          document_id: "policy",
          version_id: "v-2",
          ingestion_job_id: "job-2",
          lifecycle_status: "pending",
          ingestion_status: "running",
        },
      },
    }), { status: 409, headers: { "content-type": "application/json" } }));
    vi.stubGlobal("fetch", fetchMock);

    let error: unknown;
    try {
      await uploadDocument(
        "demo-token",
        {
          documentId: "policy",
          title: "制度",
          department: "hr",
          file: new File(["正文"], "policy.md", { type: "text/markdown" }),
          versionOf: "policy",
        },
        "upload-key",
      );
    } catch (cause) {
      error = cause;
    }

    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({
      status: 409,
      code: "document_already_exists",
      recovery: { document_id: "policy", ingestion_job_id: "job-2" },
    });
    expect(fetchMock.mock.calls[0][0]).toBe("/v1/documents/policy/versions");
    expect(new Headers(fetchMock.mock.calls[0][1].headers).get("Idempotency-Key")).toBe("upload-key");
    expect(fetchMock.mock.calls[0][1].body).toBeInstanceOf(FormData);
  });
});
