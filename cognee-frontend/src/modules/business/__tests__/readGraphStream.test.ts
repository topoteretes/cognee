import { ReadableStream as NodeReadableStream } from "node:stream/web";
import { HttpError } from "@/services/http/errors";
import readGraphStream, { STREAM_IDLE_TIMEOUT_MS } from "../readGraphStream";

// jsdom ships no ReadableStream; the browser and Node both do.
if (typeof globalThis.ReadableStream === "undefined") {
  globalThis.ReadableStream = NodeReadableStream as unknown as typeof globalThis.ReadableStream;
}

type Frame = [event: string, data: unknown];

function frame([event, data]: Frame): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

function bodyOf(frames: Frame[], extra: string[] = [], onCancel?: () => void): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const text of [...frames.map(frame), ...extra]) controller.enqueue(encoder.encode(text));
      controller.close();
    },
    cancel() {
      onCancel?.();
    },
  });
}

const META: Frame = ["meta", { seeds: ["a"], seed_source: "degree", depth: 2, max_nodes: 1000, chunk_size: 1000 }];
const CHUNK_0: Frame = [
  "chunk",
  {
    index: 0,
    nodes: [
      { id: "a", name: "Alpha", type: "Entity", entity_type: "Entity", stage: "entity", is_unnamed: false },
      { id: "t", name: "Person", type: "EntityType", entity_type: "EntityType", stage: "type", is_unnamed: false },
    ],
    links: [{ source: "a", target: "t", relation: "is_a", edge_class: "type" }],
  },
];
const CHUNK_1: Frame = [
  "chunk",
  {
    index: 1,
    nodes: [{ id: "b", name: "Beta", type: "Entity", entity_type: "Entity", stage: "entity", is_unnamed: false }],
    links: [{ source: "b", target: "a", relation: "knows", edge_class: "semantic" }],
  },
];
// A summary entry for every node, as the server always sends before `done`.
function summaryOf(ids: string[]): Frame {
  return ["summary", { nodes: Object.fromEntries(ids.map((id) => [id, { importance: 0, label_priority: false }])) }];
}
const SUMMARY_ALL = summaryOf(["a", "t", "b"]);
const SUMMARY_CHUNK_0 = summaryOf(["a", "t"]);
const DONE: Frame = ["done", { nodes: 3, links: 2, chunks: 2 }];
const DONE_CHUNK_0: Frame = ["done", { nodes: 2, links: 1, chunks: 1 }];

describe("readGraphStream", () => {
  it("joins every chunk's nodes and links in the order they arrived", async () => {
    const graph = await readGraphStream(bodyOf([META, CHUNK_0, CHUNK_1, SUMMARY_ALL, DONE]));

    expect(graph.nodes.map((n) => n.id)).toEqual(["a", "t", "b"]);
    expect(graph.links.map((l) => `${l.source}->${l.target}`)).toEqual(["a->t", "b->a"]);
  });

  it("applies importance, label priority and the color map from a summary split across events", async () => {
    const graph = await readGraphStream(
      bodyOf([
        META,
        CHUNK_0,
        CHUNK_1,
        [
          "summary",
          {
            nodes: { a: { importance: 1, label_priority: true }, t: { importance: 0.5, label_priority: false } },
            color_maps: { node_set: { github: "#123456" } },
          },
        ],
        ["summary", { nodes: { b: { importance: 0.25, label_priority: false } } }],
        DONE,
      ]),
    );

    expect(graph.nodes.map((n) => [n.id, n.importance, n.label_priority])).toEqual([
      ["a", 1, true],
      ["t", 0.5, false],
      ["b", 0.25, false],
    ]);
    expect(graph.color_maps).toEqual({ node_set: { github: "#123456" } });
  });

  it("takes the summary's entity_type over the one a chunk sent before the is_a link arrived", async () => {
    const graph = await readGraphStream(
      bodyOf([
        META,
        CHUNK_0,
        CHUNK_1,
        summaryOf(["a", "t"]),
        ["summary", { nodes: { b: { importance: 0, entity_type: "Person" } } }],
        DONE,
      ]),
    );

    expect(graph.nodes.find((n) => n.id === "b")?.entity_type).toBe("Person");
    expect(graph.nodes.find((n) => n.id === "a")?.entity_type).toBe("Entity");
  });

  it("rethrows an error event as the HttpError the JSON path would have thrown", async () => {
    const read = readGraphStream(bodyOf([META, CHUNK_0, ["error", { message: "Graph store unavailable", status: 503 }]]));

    await expect(read).rejects.toBeInstanceOf(HttpError);
    await expect(read).rejects.toMatchObject({ status: 503, message: "Graph store unavailable" });
  });

  it("refuses a stream that ends before done instead of returning part of the graph", async () => {
    const read = readGraphStream(bodyOf([META, CHUNK_0, ["summary", { nodes: {} }]]));

    await expect(read).rejects.toThrow("The graph stream ended before the whole graph arrived.");
  });

  it("skips keepalive comments and events it does not know", async () => {
    const graph = await readGraphStream(
      bodyOf([META, ["progress", { pct: 50 }], CHUNK_0, SUMMARY_CHUNK_0, DONE_CHUNK_0]),
    );
    const withKeepalive = await readGraphStream(
      bodyOf([], [": keepalive\n\n", frame(META), frame(CHUNK_0), ": keepalive\n\n", frame(SUMMARY_CHUNK_0), frame(DONE_CHUNK_0)]),
    );

    expect(graph.nodes).toHaveLength(2);
    expect(withKeepalive.nodes).toHaveLength(2);
  });

  it("drops a node without an id and a link without both endpoints", async () => {
    const graph = await readGraphStream(
      bodyOf([
        ["chunk", { nodes: [{ id: "a" }, { name: "no id" }], links: [{ source: "a", target: "a" }, { source: "a" }] }],
        summaryOf(["a"]),
        ["done", { nodes: 1, links: 1, chunks: 1 }],
      ]),
    );

    expect(graph.nodes).toEqual([{ id: "a", importance: 0, label_priority: false }]);
    expect(graph.links).toEqual([{ source: "a", target: "a" }]);
  });

  it("refuses a chunk that did not parse as soon as it arrives, instead of counting it", async () => {
    const read = readGraphStream(bodyOf([META, CHUNK_0], ["event: chunk\ndata: {not json\n\n", frame(DONE)]));

    await expect(read).rejects.toThrow("The graph stream sent a chunk that could not be read");
  });

  it("refuses a chunk that lacks its links array, though it parsed", async () => {
    const read = readGraphStream(bodyOf([META, ["chunk", { index: 0, nodes: [{ id: "a" }] }]]));

    await expect(read).rejects.toThrow("The graph stream sent a chunk that could not be read");
  });

  it("accepts a chunk that legitimately carries nothing", async () => {
    const graph = await readGraphStream(
      bodyOf([["chunk", { index: 0, nodes: [], links: [] }], summaryOf([]), ["done", { nodes: 0, links: 0, chunks: 1 }]]),
    );

    expect(graph.nodes).toEqual([]);
  });

  it("refuses a stream that delivered fewer chunks than done reports", async () => {
    const read = readGraphStream(bodyOf([META, CHUNK_0, ["done", { nodes: 2, links: 1, chunks: 2 }]]));

    await expect(read).rejects.toThrow(
      "The graph stream lost part of the graph: 2 of 2 nodes, 1 of 1 links, 1 of 2 chunks.",
    );
  });

  it("refuses a node sent twice, since the canvas would draw it twice", async () => {
    const repeat: Frame = ["chunk", { nodes: [{ id: "a" }], links: [] }];
    const read = readGraphStream(
      bodyOf([CHUNK_0, repeat, SUMMARY_CHUNK_0, ["done", { nodes: 2, links: 1, chunks: 2 }]]),
    );

    await expect(read).rejects.toThrow("The graph stream lost part of the graph: 3 of 2 nodes");
  });

  it("refuses a done frame that was cut before its totals, since nothing then vouches for the graph", async () => {
    const read = readGraphStream(bodyOf([META, CHUNK_0], ['event: done\ndata: {"nodes": 2, "li']));

    await expect(read).rejects.toThrow("The graph stream ended with unreadable totals");
  });

  it("refuses a stream that lost a link even when every node arrived", async () => {
    const read = readGraphStream(bodyOf([META, CHUNK_0, ["done", { nodes: 2, links: 2, chunks: 1 }]]));

    await expect(read).rejects.toThrow("1 of 2 links");
  });

  it("refuses a graph whose summary did not cover every node, rather than drawing it unsized", async () => {
    const read = readGraphStream(bodyOf([META, CHUNK_0, CHUNK_1, SUMMARY_CHUNK_0, DONE]));

    await expect(read).rejects.toThrow("The graph stream's summary covered 2 of 3 nodes.");
  });

  it("refuses a graph whose only summary did not parse", async () => {
    const read = readGraphStream(bodyOf([META, CHUNK_0], ["event: summary\ndata: {not json\n\n", frame(DONE_CHUNK_0)]));

    await expect(read).rejects.toThrow("The graph stream's summary covered 0 of 2 nodes.");
  });

  it("merges a summary entry for a node it already holds instead of replacing it", async () => {
    const graph = await readGraphStream(
      bodyOf([
        CHUNK_0,
        ["summary", { nodes: { a: { importance: 3 }, t: { importance: 1, label_priority: true } } }],
        ["summary", { nodes: { a: { label_priority: true } } }],
        DONE_CHUNK_0,
      ]),
    );

    expect(graph.nodes.find((n) => n.id === "a")).toMatchObject({ importance: 3, label_priority: true });
  });

  describe("when the pod goes quiet", () => {
    beforeEach(() => {
      jest.useFakeTimers();
    });

    afterEach(() => {
      jest.useRealTimers();
    });

    function quietAfter(frames: Frame[], onCancel: () => void): ReadableStream<Uint8Array> {
      const encoder = new TextEncoder();
      return new ReadableStream<Uint8Array>({
        start(controller) {
          for (const f of frames) controller.enqueue(encoder.encode(frame(f)));
        },
        cancel: onCancel,
      });
    }

    it("gives up after the idle timeout and releases the connection, rather than waiting for the pod's own drop", async () => {
      const onCancel = jest.fn();
      const read = readGraphStream(quietAfter([META, CHUNK_0], onCancel));
      const outcome = expect(read).rejects.toThrow("The stream sent nothing for 30 s.");

      await jest.advanceTimersByTimeAsync(STREAM_IDLE_TIMEOUT_MS);

      await outcome;
      expect(onCancel).toHaveBeenCalledTimes(1);
    });

    it("is not cut by a pod that keeps sending keepalives past the timeout in total", async () => {
      const encoder = new TextEncoder();
      let controller: ReadableStreamDefaultController<Uint8Array> | undefined;
      const body = new ReadableStream<Uint8Array>({
        start(c) {
          controller = c;
        },
      });
      const read = readGraphStream(body);

      for (let i = 0; i < 4; i += 1) {
        await jest.advanceTimersByTimeAsync(STREAM_IDLE_TIMEOUT_MS - 1);
        controller?.enqueue(encoder.encode(": keepalive\n\n"));
      }
      for (const f of [CHUNK_0, SUMMARY_CHUNK_0, DONE_CHUNK_0]) controller?.enqueue(encoder.encode(frame(f)));

      expect((await read).nodes.map((n) => n.id)).toEqual(["a", "t"]);
    });

    it.each([
      ["an error frame", [["error", { message: "boom", status: 500 }]] as Frame[]],
      ["an unreadable chunk", [["chunk", {}]] as Frame[]],
      ["totals that do not match", [CHUNK_0, ["done", { nodes: 9, links: 1, chunks: 1 }]] as Frame[]],
    ])("leaves no timer behind after %s", async (_name, frames) => {
      await expect(readGraphStream(bodyOf(frames))).rejects.toThrow();

      expect(jest.getTimerCount()).toBe(0);
    });

    it("leaves no timer behind once done has arrived", async () => {
      await readGraphStream(bodyOf([CHUNK_0, SUMMARY_CHUNK_0, DONE_CHUNK_0]));

      expect(jest.getTimerCount()).toBe(0);
    });
  });

  it("releases the connection as soon as done arrives, without waiting for the body to close", async () => {
    const onCancel = jest.fn();
    await readGraphStream(bodyOf([CHUNK_0, SUMMARY_CHUNK_0, DONE_CHUNK_0], [": trailing keepalive\n\n"], onCancel));

    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("returns an empty graph for a read that resolved no seeds", async () => {
    const graph = await readGraphStream(
      bodyOf([
        ["meta", { seeds: [] }],
        ["summary", { nodes: {}, color_maps: { node_set: {} } }],
        ["done", { nodes: 0, links: 0, chunks: 0 }],
      ]),
    );

    expect(graph).toEqual({ nodes: [], links: [], color_maps: { node_set: {} } });
  });
});
