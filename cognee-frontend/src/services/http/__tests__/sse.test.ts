import { ReadableStream as NodeReadableStream } from "node:stream/web";
import { HttpError } from "../errors";
import { errorFrameToHttpError, isEventStream, parseFrameData, readSseFrames, type SseFrame } from "../sse";

// jsdom ships no ReadableStream; the browser and Node both do.
if (typeof globalThis.ReadableStream === "undefined") {
  globalThis.ReadableStream = NodeReadableStream as unknown as typeof globalThis.ReadableStream;
}

function bodyOf(chunks: string[], onCancel?: () => void): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
    cancel() {
      onCancel?.();
    },
  });
}

async function collect(body: ReadableStream<Uint8Array>): Promise<SseFrame[]> {
  const frames: SseFrame[] = [];
  for await (const frame of readSseFrames(body)) frames.push(frame);
  return frames;
}

describe("readSseFrames", () => {
  it("yields one frame per blank-line-terminated block", async () => {
    const frames = await collect(
      bodyOf(['event: stage\ndata: {"stage":"retrieving"}\n\nevent: delta\ndata: {"text":"Hi"}\n\n']),
    );

    expect(frames).toEqual([
      { event: "stage", data: '{"stage":"retrieving"}' },
      { event: "delta", data: '{"text":"Hi"}' },
    ]);
  });

  it("reassembles a frame split across chunks, mid-line included", async () => {
    const frames = await collect(
      bodyOf(["event: del", 'ta\ndata: {"te', 'xt":"He', 'llo"}\n', "\n"]),
    );

    expect(frames).toEqual([{ event: "delta", data: '{"text":"Hello"}' }]);
  });

  it("does not dispatch a block before its terminating blank line", async () => {
    const frames = await collect(bodyOf(['event: final\ndata: {"results":[]}\n']));

    // The stream closed right after the data line: that is the last frame.
    expect(frames).toEqual([{ event: "final", data: '{"results":[]}' }]);
  });

  it("joins multi-line data with newlines and defaults the event name", async () => {
    const frames = await collect(bodyOf(["data: first\ndata: second\n\n"]));

    expect(frames).toEqual([{ event: "message", data: "first\nsecond" }]);
  });

  it("drops keepalive comments and tolerates CRLF line endings", async () => {
    const frames = await collect(
      bodyOf([": keepalive\r\n\r\nevent: delta\r\ndata: {\"text\":\"x\"}\r\n\r\n: keepalive\n\n"]),
    );

    expect(frames).toEqual([{ event: "delta", data: '{"text":"x"}' }]);
  });

  it("keeps a value's leading spaces beyond the first", async () => {
    const frames = await collect(bodyOf(["data:  two spaces\n\n"]));

    expect(frames).toEqual([{ event: "message", data: " two spaces" }]);
  });

  it("cancels the body when the consumer stops early", async () => {
    const onCancel = jest.fn();
    // Left open on purpose: a stream the server has already closed has
    // nothing to cancel, and this is about a consumer that leaves mid-answer.
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode("event: a\ndata: 1\n\nevent: b\ndata: 2\n\n"));
      },
      cancel: onCancel,
    });

    for await (const frame of readSseFrames(body)) {
      expect(frame.event).toBe("a");
      break;
    }

    expect(onCancel).toHaveBeenCalledTimes(1);
  });
});

function withContentType(contentType: string | null): Response {
  const headers = { get: (name: string) => (name.toLowerCase() === "content-type" ? contentType : null) };
  return { headers } as unknown as Response;
}

describe("readSseFrames with an idle timeout", () => {
  const IDLE_MS = 1_000;
  const encoder = new TextEncoder();

  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  // A body the test feeds by hand, so it controls when bytes arrive.
  function liveBody(onCancel: () => void = () => undefined): {
    body: ReadableStream<Uint8Array>;
    send: (text: string) => void;
    close: () => void;
  } {
    let controller: ReadableStreamDefaultController<Uint8Array> | undefined;
    const body = new ReadableStream<Uint8Array>({
      start(c) {
        controller = c;
      },
      cancel: onCancel,
    });
    return {
      body,
      send: (text) => controller?.enqueue(encoder.encode(text)),
      close: () => controller?.close(),
    };
  }

  async function drain(body: ReadableStream<Uint8Array>): Promise<SseFrame[]> {
    const frames: SseFrame[] = [];
    for await (const frame of readSseFrames(body, IDLE_MS)) frames.push(frame);
    return frames;
  }

  it("throws, and releases the body once, when nothing arrives for the whole timeout", async () => {
    const onCancel = jest.fn();
    const { body, send } = liveBody(onCancel);
    send("data: x\n\n");
    const reading = drain(body);
    const outcome = expect(reading).rejects.toThrow("The stream sent nothing for 1 s.");

    await jest.advanceTimersByTimeAsync(IDLE_MS);

    await outcome;
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("throws instead of ending cleanly, so a caller waiting for a final frame cannot mistake it for a close", async () => {
    const { body } = liveBody();
    const reading = drain(body);
    const outcome = expect(reading).rejects.toThrow("sent nothing");

    await jest.advanceTimersByTimeAsync(IDLE_MS);

    await outcome;
  });

  it("is not cut by a stream that keeps sending, however long it takes in total", async () => {
    const { body, send, close } = liveBody();
    const reading = drain(body);

    for (let i = 0; i < 5; i += 1) {
      await jest.advanceTimersByTimeAsync(IDLE_MS - 1);
      send(`data: ${i}\n\n`);
    }
    close();

    expect((await reading).map((f) => f.data)).toEqual(["0", "1", "2", "3", "4"]);
  });

  it("counts a keepalive comment as activity", async () => {
    const { body, send, close } = liveBody();
    const reading = drain(body);

    await jest.advanceTimersByTimeAsync(IDLE_MS - 1);
    send(": keepalive\n\n");
    await jest.advanceTimersByTimeAsync(IDLE_MS - 1);
    send("data: late\n\n");
    close();

    expect(await reading).toEqual([{ event: "message", data: "late" }]);
  });

  it("holds no timer once the body has ended or the consumer left early", async () => {
    const closed = liveBody();
    closed.send("data: x\n\n");
    closed.close();
    await drain(closed.body);

    const early = liveBody();
    early.send("data: x\n\ndata: y\n\n");
    for await (const frame of readSseFrames(early.body, IDLE_MS)) {
      expect(frame.data).toBe("x");
      break;
    }

    expect(jest.getTimerCount()).toBe(0);
  });

  it("holds no timer when the body errors while a read is waiting", async () => {
    let controller: ReadableStreamDefaultController<Uint8Array> | undefined;
    const body = new ReadableStream<Uint8Array>({
      start(c) {
        controller = c;
      },
    });
    const reading = drain(body);
    const outcome = expect(reading).rejects.toThrow("connection reset");

    controller?.error(new Error("connection reset"));

    await outcome;
    expect(jest.getTimerCount()).toBe(0);
  });

  it("reads exactly as before when no timeout is given", async () => {
    const frames: SseFrame[] = [];
    for await (const frame of readSseFrames(bodyOf(["data: x\n\n"]))) frames.push(frame);

    expect(frames).toEqual([{ event: "message", data: "x" }]);
    expect(jest.getTimerCount()).toBe(0);
  });
});

describe("isEventStream", () => {
  it("recognises an event stream whatever its parameters or case", () => {
    expect(isEventStream(withContentType("Text/Event-Stream; charset=utf-8"))).toBe(true);
  });

  it("treats a JSON answer as the non-streamed fallback", () => {
    expect(isEventStream(withContentType("application/json"))).toBe(false);
  });

  it("treats a response with no content type as not streamed", () => {
    expect(isEventStream(withContentType(null))).toBe(false);
  });
});

describe("parseFrameData", () => {
  it("returns the frame's JSON object", () => {
    expect(parseFrameData('{"a":1}')).toEqual({ a: 1 });
  });

  it("returns an empty object for data that is not JSON, or JSON null, a scalar or an array", () => {
    expect(["not json", "null", "42", "[1,2]"].map(parseFrameData)).toEqual([{}, {}, {}, {}]);
  });
});

describe("errorFrameToHttpError", () => {
  it("keeps the status and message the pod put in the frame", () => {
    const error = errorFrameToHttpError({ status: 402, message: "Out of credits" }, "fallback");

    expect(error).toBeInstanceOf(HttpError);
    expect([error.status, error.message]).toEqual([402, "Out of credits"]);
  });

  it("falls back to the pod's catch-all status and the caller's message", () => {
    const error = errorFrameToHttpError({}, "Graph read failed");

    expect([error.status, error.message]).toEqual([409, "Graph read failed"]);
  });
});
