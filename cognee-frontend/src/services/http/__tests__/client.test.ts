import { createHttpClient, makeRegistry, HttpError } from "../client";

// ─── fetch mock setup ─────────────────────────────────────────────────────────

function mockFetch(responses: Array<{ status: number; body?: unknown; headers?: Record<string, string> }>) {
  let call = 0;
  global.fetch = jest.fn().mockImplementation(() => {
    const r = responses[Math.min(call++, responses.length - 1)];
    const bodyText = r.body !== undefined ? JSON.stringify(r.body) : "";
    return Promise.resolve(
      new Response(bodyText, {
        status: r.status,
        headers: { "Content-Type": "application/json", ...r.headers },
      })
    );
  });
}

function mockFetchError(error: unknown) {
  global.fetch = jest.fn().mockRejectedValue(error);
}

afterEach(() => {
  jest.restoreAllMocks();
});

// ─── makeRegistry ─────────────────────────────────────────────────────────────

describe("makeRegistry", () => {
  it("registers and iterates handlers in insertion order", () => {
    const reg = makeRegistry<(x: number) => number>();
    reg.use((x) => x + 1);
    reg.use((x) => x * 2);
    const fns = Array.from(reg.values());
    expect(fns[0](3)).toBe(4);
    expect(fns[1](3)).toBe(6);
  });

  it("ejects by id", () => {
    const reg = makeRegistry<() => void>();
    const id = reg.use(() => {});
    reg.eject(id);
    expect(Array.from(reg.values())).toHaveLength(0);
  });

  it("clears all", () => {
    const reg = makeRegistry<() => void>();
    reg.use(() => {});
    reg.use(() => {});
    reg.clear();
    expect(Array.from(reg.values())).toHaveLength(0);
  });

  it("returns incremental ids", () => {
    const reg = makeRegistry<() => void>();
    const a = reg.use(() => {});
    const b = reg.use(() => {});
    expect(b).toBeGreaterThan(a);
  });
});

// ─── basic requests ───────────────────────────────────────────────────────────

describe("http verbs", () => {
  it("GET resolves on 200", async () => {
    mockFetch([{ status: 200, body: { ok: true } }]);
    const client = createHttpClient();
    const res = await client.get("/api/test");
    expect(res.status).toBe(200);
  });

  it("POST sends JSON body with Content-Type header", async () => {
    mockFetch([{ status: 201, body: {} }]);
    const client = createHttpClient();
    await client.post("/api/test", { name: "foo" });
    const init = (global.fetch as jest.Mock).mock.calls[0][1] as RequestInit;
    expect((init.headers as Record<string, string>)["Content-Type"]).toBe("application/json");
    expect(init.body).toBe(JSON.stringify({ name: "foo" }));
  });

  it("DELETE sends no body", async () => {
    mockFetch([{ status: 204 }]);
    const client = createHttpClient();
    await client.delete("/api/test");
    const init = (global.fetch as jest.Mock).mock.calls[0][1] as RequestInit;
    expect(init.body).toBeUndefined();
  });

  it("method is normalized to uppercase", async () => {
    mockFetch([{ status: 200, body: {} }]);
    const client = createHttpClient();
    await client.request("/api/test", { method: "get" });
    const init = (global.fetch as jest.Mock).mock.calls[0][1] as RequestInit;
    expect(init.method).toBe("GET");
  });
});

// ─── error handling ───────────────────────────────────────────────────────────

describe("error handling", () => {
  it("throws HttpError on non-ok response", async () => {
    mockFetch([{ status: 404, body: { detail: "Not found" } }]);
    const client = createHttpClient();
    await expect(client.get("/api/missing")).rejects.toBeInstanceOf(HttpError);
  });

  it("HttpError.status matches response status", async () => {
    mockFetch([{ status: 422, body: { detail: "Validation error" } }]);
    const client = createHttpClient();
    try {
      await client.get("/api/bad");
    } catch (e) {
      expect(e).toBeInstanceOf(HttpError);
      expect((e as HttpError).status).toBe(422);
    }
  });

  it("TypeError becomes 'No connection to the server'", async () => {
    mockFetchError(new TypeError("Failed to fetch"));
    const client = createHttpClient();
    await expect(client.get("/api/test")).rejects.toThrow("No connection to the server.");
  });

  it("TimeoutError becomes 'Request timed out'", async () => {
    const err = new DOMException("Timed out", "TimeoutError");
    mockFetchError(err);
    const client = createHttpClient();
    await expect(client.get("/api/test")).rejects.toThrow("Request timed out.");
  });

  it("AbortError (caller cancel) propagates as-is for React Query suppression", async () => {
    const err = new DOMException("Aborted", "AbortError");
    mockFetchError(err);
    const client = createHttpClient();
    const caught = await client.get("/api/test").catch((e) => e);
    expect(caught.name).toBe("AbortError");
  });
});

// ─── JSON helpers ─────────────────────────────────────────────────────────────

describe("getJson / postJson", () => {
  it("getJson returns parsed body", async () => {
    mockFetch([{ status: 200, body: { id: 1, name: "Alice" } }]);
    const client = createHttpClient();
    const data = await client.getJson<{ id: number; name: string }>("/api/users/1");
    expect(data).toEqual({ id: 1, name: "Alice" });
  });

  it("getJson runs validate when provided", async () => {
    mockFetch([{ status: 200, body: { id: "bad" } }]);
    const client = createHttpClient();
    const validate = jest.fn((d: unknown) => {
      if (typeof (d as any).id !== "number") throw new Error("Invalid");
      return d as { id: number };
    });
    await expect(client.getJson("/api/users/1", { validate })).rejects.toThrow("Invalid");
    expect(validate).toHaveBeenCalledTimes(1);
  });

  it("validate does not leak into fetch headers", async () => {
    mockFetch([{ status: 200, body: {} }]);
    const client = createHttpClient();
    await client.getJson("/api/test", { validate: (d) => d });
    const init = (global.fetch as jest.Mock).mock.calls[0][1] as RequestInit;
    expect(JSON.stringify(init)).not.toContain("validate");
  });

  it("postJson parses response body", async () => {
    mockFetch([{ status: 200, body: { created: true } }]);
    const client = createHttpClient();
    const data = await client.postJson<{ created: boolean }>("/api/items", { name: "x" });
    expect(data.created).toBe(true);
  });
});

// ─── request() method ────────────────────────────────────────────────────────

describe("request()", () => {
  it("null body is treated as no body", async () => {
    mockFetch([{ status: 200, body: {} }]);
    const client = createHttpClient();
    await client.request("/api/test", { body: null, method: "POST" });
    const init = (global.fetch as jest.Mock).mock.calls[0][1] as RequestInit;
    expect(init.body).toBeUndefined();
  });
});

// ─── FormData — no Content-Type injection ────────────────────────────────────

describe("FormData body", () => {
  it("does not set Content-Type for FormData", async () => {
    mockFetch([{ status: 200, body: {} }]);
    const client = createHttpClient();
    const fd = new FormData();
    fd.append("key", "value");
    await client.post("/api/upload", fd);
    const init = (global.fetch as jest.Mock).mock.calls[0][1] as RequestInit;
    expect((init.headers as Record<string, string>)?.["Content-Type"]).toBeUndefined();
  });
});

// ─── interceptors ────────────────────────────────────────────────────────────

describe("request interceptors", () => {
  it("interceptor can add headers", async () => {
    mockFetch([{ status: 200, body: {} }]);
    const client = createHttpClient();
    client.interceptors.request.use((ctx) => ({
      ...ctx,
      headers: { ...ctx.headers, "X-Test": "yes" },
    }));
    await client.get("/api/test");
    const init = (global.fetch as jest.Mock).mock.calls[0][1] as RequestInit;
    expect((init.headers as Record<string, string>)["X-Test"]).toBe("yes");
  });

  it("interceptor modifications do not affect other client instances", async () => {
    mockFetch([{ status: 200, body: {} }, { status: 200, body: {} }]);
    const a = createHttpClient();
    const b = createHttpClient();
    a.interceptors.request.use((ctx) => ({ ...ctx, headers: { ...ctx.headers, "X-From": "a" } }));
    await a.get("/api/test");
    await b.get("/api/test");
    const callA = (global.fetch as jest.Mock).mock.calls[0][1] as RequestInit;
    const callB = (global.fetch as jest.Mock).mock.calls[1][1] as RequestInit;
    expect((callA.headers as Record<string, string>)["X-From"]).toBe("a");
    expect((callB.headers as Record<string, string>)?.["X-From"]).toBeUndefined();
  });

  it("eject removes interceptor", async () => {
    mockFetch([{ status: 200, body: {} }]);
    const client = createHttpClient();
    const id = client.interceptors.request.use((ctx) => ({
      ...ctx,
      headers: { ...ctx.headers, "X-Ejected": "true" },
    }));
    client.interceptors.request.eject(id);
    await client.get("/api/test");
    const init = (global.fetch as jest.Mock).mock.calls[0][1] as RequestInit;
    expect((init.headers as Record<string, string>)?.["X-Ejected"]).toBeUndefined();
  });
});

describe("response interceptors", () => {
  it("can transform response", async () => {
    mockFetch([{ status: 200, body: {} }]);
    const client = createHttpClient();
    client.interceptors.response.use((res) => new Response("{}", { status: 202 }));
    const res = await client.get("/api/test");
    expect(res.status).toBe(202);
  });
});

describe("error interceptors", () => {
  it("can recover by returning a Response", async () => {
    mockFetch([{ status: 401, body: {} }]);
    const client = createHttpClient();
    client.interceptors.error.use(async (err) => {
      if (err instanceof HttpError && err.status === 401) {
        return new Response("{}", { status: 200 });
      }
    });
    const res = await client.get("/api/test");
    expect(res.status).toBe(200);
  });

  it("can re-throw a transformed error", async () => {
    mockFetch([{ status: 500, body: {} }]);
    const client = createHttpClient();
    client.interceptors.error.use(() => {
      throw new Error("custom error");
    });
    await expect(client.get("/api/test")).rejects.toThrow("custom error");
  });
});

// ─── retry ────────────────────────────────────────────────────────────────────

describe("retry", () => {
  beforeEach(() => jest.useFakeTimers());
  afterEach(() => jest.useRealTimers());

  it("retries on 503 and succeeds", async () => {
    mockFetch([{ status: 503 }, { status: 200, body: { ok: true } }]);
    const client = createHttpClient();
    const promise = client.get("/api/test", { retries: 1, backoffMs: 10 });
    await jest.runAllTimersAsync();
    const res = await promise;
    expect(res.status).toBe(200);
    expect((global.fetch as jest.Mock)).toHaveBeenCalledTimes(2);
  });

  it("retries on 502", async () => {
    mockFetch([{ status: 502 }, { status: 200, body: {} }]);
    const client = createHttpClient();
    const promise = client.get("/api/test", { retries: 1, backoffMs: 10 });
    await jest.runAllTimersAsync();
    await promise;
    expect((global.fetch as jest.Mock)).toHaveBeenCalledTimes(2);
  });

  it("does not retry on 404", async () => {
    mockFetch([{ status: 404, body: {} }]);
    const client = createHttpClient();
    await expect(client.get("/api/test", { retries: 2 })).rejects.toBeInstanceOf(HttpError);
    expect((global.fetch as jest.Mock)).toHaveBeenCalledTimes(1);
  });

  it("throws after exhausting all retries", async () => {
    jest.useRealTimers();
    mockFetch([{ status: 503 }, { status: 503 }]);
    const client = createHttpClient();
    // backoffMs: 0 — no actual wait, resolves synchronously via microtask
    await expect(client.get("/api/test", { retries: 1, backoffMs: 0 })).rejects.toBeInstanceOf(HttpError);
    expect((global.fetch as jest.Mock)).toHaveBeenCalledTimes(2);
  });

  it("respects Retry-After header in seconds", async () => {
    mockFetch([
      { status: 429, headers: { "Retry-After": "1" } },
      { status: 200, body: {} },
    ]);
    const client = createHttpClient();
    const promise = client.get("/api/test", { retries: 1 });
    await jest.runAllTimersAsync();
    const res = await promise;
    expect(res.status).toBe(200);
  });

  it("aborts retry when caller signal is already aborted", async () => {
    jest.useRealTimers();
    const controller = new AbortController();
    controller.abort();
    mockFetch([{ status: 503 }, { status: 200, body: {} }]);
    const client = createHttpClient();
    // Aborted signal — first fetch fires, gets 503, then next iteration checks abort → throws
    await expect(
      client.get("/api/test", { retries: 1, backoffMs: 0, signal: controller.signal })
    ).rejects.toThrow();
  });
});

// ─── logger ───────────────────────────────────────────────────────────────────

describe("setLogger", () => {
  it("calls logger on success with status and durationMs", async () => {
    mockFetch([{ status: 200, body: {} }]);
    const client = createHttpClient();
    const logger = jest.fn();
    client.setLogger(logger);
    await client.get("/api/test");
    expect(logger).toHaveBeenCalledWith(
      expect.objectContaining({ status: 200, method: "GET", url: "/api/test" })
    );
    expect(logger.mock.calls[0][0].durationMs).toBeGreaterThanOrEqual(0);
  });

  it("calls logger with error on failure", async () => {
    mockFetch([{ status: 500, body: {} }]);
    const client = createHttpClient();
    const logger = jest.fn();
    client.setLogger(logger);
    await expect(client.get("/api/test")).rejects.toBeInstanceOf(HttpError);
    expect(logger).toHaveBeenCalledWith(expect.objectContaining({ error: expect.any(HttpError) }));
  });

  it("logger crash does not fail the request", async () => {
    mockFetch([{ status: 200, body: {} }]);
    const client = createHttpClient();
    client.setLogger(() => { throw new Error("logger bug"); });
    await expect(client.get("/api/test")).resolves.toBeDefined();
  });
});
