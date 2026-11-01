import { ReadableStream as NodeReadableStream } from "node:stream/web";

jest.mock("@/modules/configuration/pipelineSettings", () => ({
  getPipelineSettingsFromStorage: () => ({ topK: 15, includeReferences: true }),
}));

const mockNotifyInsufficientCredits = jest.fn();
jest.mock("@/services/credits/insufficientCreditsBridge", () => ({
  notifyInsufficientCredits: (...args: unknown[]) => mockNotifyInsufficientCredits(...args),
}));

import recallKnowledge, { type RecallStreamHandlers } from "../recallKnowledge";
import { HttpError } from "@/services/http/errors";
import type { CogneeInstance } from "@/modules/instances/types";

if (typeof globalThis.ReadableStream === "undefined") {
  globalThis.ReadableStream = NodeReadableStream as unknown as typeof globalThis.ReadableStream;
}

const TENANT_ID = "tenant-1";

function sseBody(frames: Array<[string, unknown]>): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  const text = frames
    .map(([event, data]) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)
    .join(": keepalive\n\n");
  return new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(encoder.encode(text));
      controller.close();
    },
  });
}

function response(contentType: string, init: { json?: unknown; body?: ReadableStream<Uint8Array> }): Response {
  return {
    ok: true,
    status: 200,
    headers: { get: (name: string) => (name.toLowerCase() === "content-type" ? contentType : null) },
    json: () => Promise.resolve(init.json),
    body: init.body ?? null,
  } as unknown as Response;
}

function instanceAnswering(res: Response): CogneeInstance & { fetch: jest.Mock } {
  return { name: "test", instanceId: TENANT_ID, fetch: jest.fn().mockResolvedValue(res) };
}

function sentBody(instance: { fetch: jest.Mock }): Record<string, unknown> {
  const init = instance.fetch.mock.calls[0][1] as RequestInit;
  return JSON.parse(init.body as string) as Record<string, unknown>;
}

function sentHeaders(instance: { fetch: jest.Mock }): Record<string, string> {
  const init = instance.fetch.mock.calls[0][1] as RequestInit;
  return init.headers as Record<string, string>;
}

function handlers(): Required<RecallStreamHandlers> & { onStage: jest.Mock; onDelta: jest.Mock; onReset: jest.Mock } {
  return { onStage: jest.fn(), onDelta: jest.fn(), onReset: jest.fn() };
}

afterEach(() => jest.clearAllMocks());

describe("recallKnowledge — JSON transport", () => {
  it("posts the query with the session and returns the payload", async () => {
    const instance = instanceAnswering(response("application/json", { json: [{ text: "answer" }] }));

    const result = await recallKnowledge(instance, { query: "q", sessionId: "search-ui-1", datasetIds: ["ds"] });

    expect(result).toEqual([{ text: "answer" }]);
    expect(instance.fetch).toHaveBeenCalledWith("/v1/recall", expect.objectContaining({ method: "POST" }));
    expect(sentBody(instance)).toEqual({
      query: "q",
      session_id: "search-ui-1",
      dataset_ids: ["ds"],
      top_k: 15,
      include_references: true,
      search_type: "HYBRID_COMPLETION",
    });
    expect(sentHeaders(instance).Accept).toBeUndefined();
  });
});

describe("recallKnowledge — streamed transport", () => {
  it("asks for a stream and relays stages and tokens before resolving the payload", async () => {
    const instance = instanceAnswering(
      response("text/event-stream", {
        body: sseBody([
          ["stage", { stage: "retrieving" }],
          ["stage", { stage: "generating" }],
          ["delta", { text: "Three " }],
          ["delta", { text: "entities." }],
          ["answer_done", {}],
          ["final", { results: [{ text: "Three entities. [1]" }] }],
        ]),
      }),
    );
    const on = handlers();

    const result = await recallKnowledge(instance, { query: "q", sessionId: "search-ui-1" }, on);

    expect(sentBody(instance).stream).toBe(true);
    expect(sentHeaders(instance).Accept).toMatch(/^text\/event-stream/);
    expect(on.onStage.mock.calls.map(([s]) => s)).toEqual(["retrieving", "generating"]);
    expect(on.onDelta.mock.calls.map(([t]) => t)).toEqual(["Three ", "entities."]);
    expect(result).toEqual([{ text: "Three entities. [1]" }]);
  });

  it("tells the caller to discard the preview when the answer restarts", async () => {
    const instance = instanceAnswering(
      response("text/event-stream", {
        body: sseBody([["delta", { text: "first try" }], ["reset", {}], ["delta", { text: "second" }], ["final", { results: [] }]]),
      }),
    );
    const on = handlers();

    await recallKnowledge(instance, { query: "q" }, on);

    expect(on.onReset).toHaveBeenCalledTimes(1);
    expect(on.onDelta.mock.calls.map(([t]) => t)).toEqual(["first try", "second"]);
  });

  it("falls back to the JSON payload when the pod answers without streaming", async () => {
    const instance = instanceAnswering(response("application/json", { json: [{ text: "plain" }] }));
    const on = handlers();

    const result = await recallKnowledge(instance, { query: "q" }, on);

    expect(result).toEqual([{ text: "plain" }]);
    expect(on.onDelta).not.toHaveBeenCalled();
  });

  it("raises an in-stream failure with the status the JSON path would have used", async () => {
    const instance = instanceAnswering(
      response("text/event-stream", {
        body: sseBody([["stage", { stage: "generating" }], ["error", { message: "boom", status: 409 }]]),
      }),
    );

    await expect(recallKnowledge(instance, { query: "q" }, handlers())).rejects.toMatchObject({
      name: "HttpError",
      status: 409,
      message: "boom",
    });
    expect(mockNotifyInsufficientCredits).not.toHaveBeenCalled();
  });

  it("opens the credits modal for a 402 that arrived mid-stream", async () => {
    const instance = instanceAnswering(
      response("text/event-stream", {
        body: sseBody([["error", { message: "Insufficient credits to run recall", status: 402 }]]),
      }),
    );

    const failure = await recallKnowledge(instance, { query: "q" }, handlers()).catch((e: unknown) => e);

    expect(failure).toBeInstanceOf(HttpError);
    expect((failure as HttpError).status).toBe(402);
    expect(mockNotifyInsufficientCredits).toHaveBeenCalledWith(
      expect.objectContaining({ operation: "recall", tenantId: TENANT_ID }),
    );
  });

  it("fails rather than returning a preview when the stream ends without the payload", async () => {
    const instance = instanceAnswering(
      response("text/event-stream", { body: sseBody([["delta", { text: "partial" }]]) }),
    );

    await expect(recallKnowledge(instance, { query: "q" }, handlers())).rejects.toThrow(
      "The connection dropped before the answer was delivered.",
    );
  });
});
