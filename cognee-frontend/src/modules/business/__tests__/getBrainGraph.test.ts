import { ReadableStream as NodeReadableStream } from "node:stream/web";
import getBrainGraph from "../getBrainGraph";
import { SLOW_READ_TIMEOUT_MS } from "@/services/http";
import { HttpError } from "@/services/http/errors";
import type { CogneeInstance } from "@/modules/instances/types";

// jsdom ships no ReadableStream; the browser and Node both do.
if (typeof globalThis.ReadableStream === "undefined") {
  globalThis.ReadableStream = NodeReadableStream as unknown as typeof globalThis.ReadableStream;
}

const JSON_GRAPH = { nodes: [{ id: "json-node" }], links: [], color_maps: {} };

// A plain object rather than jsdom's Response, which cannot carry a stream body.
function response(contentType: string, init: { json?: unknown; body?: ReadableStream<Uint8Array> }): Response {
  return {
    ok: true,
    status: 200,
    headers: { get: (name: string) => (name.toLowerCase() === "content-type" ? contentType : null) },
    json: () => Promise.resolve(init.json),
    body: init.body ?? null,
  } as unknown as Response;
}

function jsonResponse(): Response {
  return response("application/json", { json: JSON_GRAPH });
}

function sseBody(text: string, signal?: AbortSignal): ReadableStream<Uint8Array> {
  return new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(text));
      // Left open when a signal is given, the way a stream still in flight is.
      if (!signal) controller.close();
      signal?.addEventListener("abort", () => controller.error(new DOMException("Aborted", "AbortError")));
    },
  });
}

function sseFrame(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

function streamResponse(node: { id: string; type?: string } = { id: "streamed-node" }, ...more: { id: string; type?: string }[]): Response {
  const nodes = [node, ...more];
  const text = [
    sseFrame("chunk", { index: 0, nodes, links: [] }),
    sseFrame("summary", { nodes: Object.fromEntries(nodes.map((n) => [n.id, { importance: 1 }])) }),
    sseFrame("done", { nodes: nodes.length, links: 0, chunks: 1 }),
  ].join("");
  return response("text/event-stream; charset=utf-8", { body: sseBody(text) });
}

function instanceSpy(makeResponse: () => Response = jsonResponse) {
  const fetch = jest.fn().mockImplementation(async () => makeResponse());
  return { instance: { fetch } as unknown as CogneeInstance, fetch };
}

afterEach(() => jest.restoreAllMocks());

describe("getBrainGraph", () => {
  it("asks for the focused dataset's graph", async () => {
    const { instance, fetch } = instanceSpy();

    await getBrainGraph(instance, "ds-1");

    expect(fetch.mock.calls[0][0]).toContain("/v1/visualize/json?dataset_id=ds-1");
  });

  it("overrides the default GET budget, which the graph_grew refetch cannot meet", async () => {
    // The push arrives the moment a cognify run finishes, so this read runs
    // against a pod still busy with that run and measured well past the 10s
    // default. Dropping back to the default silently reintroduces a graph
    // that only catches up after two aborted attempts.
    const { instance, fetch } = instanceSpy();

    await getBrainGraph(instance, "ds-1");

    expect(fetch.mock.calls[0][1]).toMatchObject({
      timeoutMs: SLOW_READ_TIMEOUT_MS,
      headers: { Accept: "text/event-stream, application/json;q=0.9" },
    });
    expect(SLOW_READ_TIMEOUT_MS).toBeGreaterThan(10_000);
  });

  it("passes the query's abort signal on, so a cancelled read frees its connection", async () => {
    const { instance, fetch } = instanceSpy();
    const controller = new AbortController();

    await getBrainGraph(instance, "ds-1", controller.signal);

    expect(fetch.mock.calls[0][1].signal).toBe(controller.signal);
  });

  it("asks for a stream while still accepting JSON from a pod that cannot stream", async () => {
    const { instance, fetch } = instanceSpy();

    await getBrainGraph(instance, "ds-1");

    expect(fetch.mock.calls[0][1].headers.Accept).toBe("text/event-stream, application/json;q=0.9");
  });

  it("assembles the graph from the stream when the pod streams it", async () => {
    const { instance, fetch } = instanceSpy(streamResponse);

    const graph = await getBrainGraph(instance, "ds-1");

    expect(graph.nodes).toEqual([{ id: "streamed-node", importance: 1 }]);
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it("loads full row values after discovering a compact CSV graph with the same bounds and signal", async () => {
    const row = { id: "row", type: "DltRow", name: "Table: people Columns: - person (TEXT)" };
    const detailed = { nodes: [{ ...row, text: "Row Data:\n  person: Ana Horvat" }], links: [], color_maps: {} };
    const { instance, fetch } = instanceSpy();
    fetch.mockResolvedValueOnce(streamResponse(row));
    fetch.mockResolvedValueOnce(response("application/json", { json: detailed }));
    const controller = new AbortController();

    expect((await getBrainGraph(instance, "csv-dataset", controller.signal)).nodes).toEqual([
      { ...row, importance: 1, text: "Row Data:\n  person: Ana Horvat" },
    ]);
    expect(fetch).toHaveBeenCalledTimes(2);
    expect(fetch.mock.calls[1]).toEqual([
      fetch.mock.calls[0][0],
      { timeoutMs: SLOW_READ_TIMEOUT_MS, signal: controller.signal, headers: { Accept: "application/json" } },
    ]);
  });

  it("reuses row values from the previous payload instead of reading JSON again", async () => {
    const row = { id: "row", type: "DltRow", name: "Table: people Columns: - person (TEXT)" };
    const previous = { nodes: [{ ...row, text: "Row Data:\n  person: Ana Horvat",
      document_name: "people.csv", document_id: "doc", chunk_index: 3,
    }], links: [], color_maps: {} };
    const { instance, fetch } = instanceSpy();
    fetch.mockResolvedValueOnce(streamResponse(row));

    const graph = await getBrainGraph(instance, "csv-dataset", undefined, previous);

    expect(fetch).toHaveBeenCalledTimes(1);
    expect(graph.nodes[0]).toMatchObject({ id: "row", text: "Row Data:\n  person: Ana Horvat", importance: 1,
      document_name: "people.csv", document_id: "doc", chunk_index: 3,
    });
  });

  it("reads JSON again when a row appears that the previous payload never described", async () => {
    const row = { id: "new-row", type: "DltRow" };
    const previous = { nodes: [{ id: "old-row", type: "DltRow", text: "Row Data:\n  person: Ana" }], links: [], color_maps: {} };
    const detailed = { nodes: [{ ...row, text: "Row Data:\n  person: Luka" }], links: [], color_maps: {} };
    const { instance, fetch } = instanceSpy();
    fetch.mockResolvedValueOnce(streamResponse(row));
    fetch.mockResolvedValueOnce(response("application/json", { json: detailed }));

    const graph = await getBrainGraph(instance, "csv-dataset", undefined, previous);

    expect(graph.nodes).toEqual([{ ...row, importance: 1, text: "Row Data:\n  person: Luka" }]);
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it("keeps the streamed node set when the JSON read sampled a different one", async () => {
    const streamed = { id: "row", type: "DltRow" };
    const detailed = { nodes: [{ id: "other", type: "DltRow", text: "Row Data:\n  person: Luka" }], links: [], color_maps: {} };
    const { instance, fetch } = instanceSpy();
    fetch.mockResolvedValueOnce(streamResponse(streamed));
    fetch.mockResolvedValueOnce(response("application/json", { json: detailed }));

    const graph = await getBrainGraph(instance, "csv-dataset");

    expect(graph.nodes.map((node) => node.id)).toEqual(["row"]);
    expect(graph.nodes[0].text).toBe("");
  });

  it("keeps the streamed graph, with previously read values, when the detailed read fails", async () => {
    const row = { id: "row", type: "DltRow" };
    const previous = { nodes: [{ id: "other", type: "DltRow", text: "Row Data:\n  person: Ana" }], links: [], color_maps: {} };
    const { instance, fetch } = instanceSpy();
    fetch.mockResolvedValueOnce(streamResponse(row));
    fetch.mockRejectedValueOnce(new HttpError(500, "Internal Server Error", "Row read failed"));

    const graph = await getBrainGraph(instance, "csv-dataset", undefined, previous);

    expect(graph.nodes).toEqual([{ id: "row", type: "DltRow", importance: 1, text: "" }]);
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it("keeps the fields the stream sent when the previous payload lacks them", async () => {
    const row = { id: "row", type: "DltRow", document_name: "people.csv" };
    const previous = { nodes: [{ id: "row", type: "DltRow", text: "Row Data:\n  person: Ana" }], links: [], color_maps: {} };
    const { instance, fetch } = instanceSpy();
    fetch.mockResolvedValueOnce(streamResponse(row));

    const graph = await getBrainGraph(instance, "csv-dataset", undefined, previous);

    expect(graph.nodes[0]).toMatchObject({ document_name: "people.csv", text: "Row Data:\n  person: Ana" });
  });

  it("lets a text the stream sent win over the carried one", async () => {
    const row = { id: "row", type: "DltRow", text: "Row Data:\n  person: Luka" };
    const previous = { nodes: [{ id: "row", type: "DltRow", text: "" }], links: [], color_maps: {} };
    const { instance, fetch } = instanceSpy();
    fetch.mockResolvedValueOnce(streamResponse(row));

    const graph = await getBrainGraph(instance, "csv-dataset", undefined, previous);

    expect(graph.nodes[0].text).toBe("Row Data:\n  person: Luka");
  });

  it("does not repeat the JSON read on later polls for a row the JSON read had no values for", async () => {
    const row = { id: "row", type: "DltRow" };
    const bare = { nodes: [row], links: [], color_maps: {} };
    const { instance, fetch } = instanceSpy();
    fetch.mockResolvedValueOnce(streamResponse(row));
    fetch.mockResolvedValueOnce(response("application/json", { json: bare }));
    const first = await getBrainGraph(instance, "csv-dataset");
    fetch.mockResolvedValueOnce(streamResponse(row));

    await getBrainGraph(instance, "csv-dataset", undefined, first);

    expect(fetch).toHaveBeenCalledTimes(3);
  });

  it("does not repeat a JSON read that failed on later polls", async () => {
    const row = { id: "row", type: "DltRow" };
    const { instance, fetch } = instanceSpy();
    fetch.mockResolvedValueOnce(streamResponse(row));
    fetch.mockRejectedValueOnce(new HttpError(500, "Internal Server Error", "Row read failed"));
    const first = await getBrainGraph(instance, "csv-dataset");
    fetch.mockResolvedValueOnce(streamResponse(row));

    await getBrainGraph(instance, "csv-dataset", undefined, first);

    expect(fetch).toHaveBeenCalledTimes(3);
  });

  it("fills a row marked as asked once a later JSON read returns its values", async () => {
    const r0 = { id: "r0", type: "DltRow" }, r1 = { id: "r1", type: "DltRow" };
    const { instance, fetch } = instanceSpy();
    fetch.mockResolvedValueOnce(streamResponse(r0));
    fetch.mockRejectedValueOnce(new HttpError(500, "Internal Server Error", "Row read failed"));
    const first = await getBrainGraph(instance, "csv-dataset");
    const detailed = {
      nodes: [{ ...r0, text: "Row Data:\n  name: A" }, { ...r1, text: "Row Data:\n  name: B" }], links: [], color_maps: {},
    };
    fetch.mockResolvedValueOnce(streamResponse(r0, r1));
    fetch.mockResolvedValueOnce(response("application/json", { json: detailed }));

    const graph = await getBrainGraph(instance, "csv-dataset", undefined, first);

    expect(graph.nodes.map((node) => node.text)).toEqual(["Row Data:\n  name: A", "Row Data:\n  name: B"]);
  });

  it("carries values recorded as properties as well as text", async () => {
    const row = { id: "row", type: "TableRow" };
    const previous = { nodes: [{ id: "row", type: "TableRow", properties: '{"name":"Ana"}' }], links: [], color_maps: {} };
    const { instance, fetch } = instanceSpy();
    fetch.mockResolvedValueOnce(streamResponse(row));

    const graph = await getBrainGraph(instance, "csv-dataset", undefined, previous);

    expect(graph.nodes[0].properties).toBe('{"name":"Ana"}');
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it("does not swallow a cancelled detailed read", async () => {
    const controller = new AbortController();
    const { instance, fetch } = instanceSpy();
    fetch.mockResolvedValueOnce(streamResponse({ id: "row", type: "DltRow" }));
    fetch.mockImplementationOnce(async () => { controller.abort(); throw new Error("aborted"); });

    await expect(getBrainGraph(instance, "csv-dataset", controller.signal)).rejects.toThrow("aborted");
  });

  it("reads the JSON payload when the pod answers without streaming", async () => {
    const { instance } = instanceSpy(jsonResponse);

    const graph = await getBrainGraph(instance, "ds-1");

    expect(graph).toEqual(JSON_GRAPH);
  });

  it("rethrows an error frame from mid-stream as the HttpError the JSON path would throw", async () => {
    const { instance } = instanceSpy(() =>
      response("text/event-stream", {
        body: sseBody(
          sseFrame("chunk", { index: 0, nodes: [{ id: "a" }], links: [] }) +
            sseFrame("error", { message: "Graph store unavailable", status: 500 }),
        ),
      }),
    );

    await expect(getBrainGraph(instance, "ds-1")).rejects.toMatchObject({ status: 500, message: "Graph store unavailable" });
  });

  it("rejects with the abort, not a partial graph, when the read is cancelled mid-stream", async () => {
    const controller = new AbortController();
    const { instance } = instanceSpy(() =>
      response("text/event-stream", {
        body: sseBody(sseFrame("chunk", { index: 0, nodes: [{ id: "a" }], links: [] }), controller.signal),
      }),
    );

    const read = getBrainGraph(instance, "ds-1", controller.signal);
    controller.abort();

    await expect(read).rejects.toMatchObject({ name: "AbortError" });
  });

  it("falls back to the JSON read when the pod refuses the stream for capacity", async () => {
    const { instance, fetch } = instanceSpy();
    fetch.mockRejectedValueOnce(new HttpError(503, "Service Unavailable", "Too many graph streams in flight"));

    const graph = await getBrainGraph(instance, "ds-1");

    expect(fetch.mock.calls.map((call) => call[1].headers.Accept)).toEqual([
      "text/event-stream, application/json;q=0.9",
      "application/json",
    ]);
    expect(graph).toEqual(JSON_GRAPH);
  });

  it("does not retry a failure that is not the stream cap", async () => {
    const { instance, fetch } = instanceSpy();
    fetch.mockRejectedValueOnce(new HttpError(500, "Internal Server Error", "boom"));

    await expect(getBrainGraph(instance, "ds-1")).rejects.toMatchObject({ status: 500 });
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it("names the transport when the pod announces a stream without a body", async () => {
    const { instance } = instanceSpy(() => response("text/event-stream", {}));

    await expect(getBrainGraph(instance, "ds-1")).rejects.toThrow("The pod announced a graph stream but sent no body.");
  });
});
