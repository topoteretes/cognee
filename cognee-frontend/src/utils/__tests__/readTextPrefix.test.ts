import readTextPrefix from "../readTextPrefix";

function streamOf(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  let index = 0;
  return {
    getReader: () => ({
      read: async () =>
        index < chunks.length
          ? { done: false, value: encoder.encode(chunks[index++]) }
          : { done: true, value: undefined },
      cancel: async () => undefined,
    }),
  } as unknown as ReadableStream<Uint8Array>;
}

function responseOf(body: ReadableStream<Uint8Array> | null, text = ""): Response {
  return { body, text: async () => text } as unknown as Response;
}

describe("readTextPrefix", () => {
  it("stops reading once it has enough bytes", async () => {
    const result = await readTextPrefix(responseOf(streamOf(["Session ID: abc\n", "x".repeat(5000)])), 20);

    expect(result).toBe("Session ID: abc\nxxxx");
  });

  it("returns the whole body when it is shorter than the cap", async () => {
    const result = await readTextPrefix(responseOf(streamOf(["# Q3 roadmap\n\nbody"])), 8192);

    expect(result).toBe("# Q3 roadmap\n\nbody");
  });

  it("cancels the rest of the stream rather than draining it", async () => {
    const cancel = jest.fn().mockResolvedValue(undefined);
    const encoder = new TextEncoder();
    const read = jest.fn().mockResolvedValue({ done: false, value: encoder.encode("y".repeat(100)) });
    const response = {
      body: { getReader: () => ({ read, cancel }) },
      text: async () => "",
    } as unknown as Response;

    await readTextPrefix(response, 50);

    expect(cancel).toHaveBeenCalled();
    // One read covered the cap; an unbounded loop would keep pulling forever.
    expect(read).toHaveBeenCalledTimes(1);
  });

  it("falls back to text() when the body is not a readable stream", async () => {
    const result = await readTextPrefix(responseOf(null, "plain body"), 8192);

    expect(result).toBe("plain body");
  });

  it("still returns what it read when cancelling throws", async () => {
    const encoder = new TextEncoder();
    const response = {
      body: {
        getReader: () => ({
          read: jest.fn().mockResolvedValue({ done: false, value: encoder.encode("hello") }),
          cancel: jest.fn().mockRejectedValue(new Error("already closed")),
        }),
      },
      text: async () => "",
    } as unknown as Response;

    await expect(readTextPrefix(response, 5)).resolves.toBe("hello");
  });
});
