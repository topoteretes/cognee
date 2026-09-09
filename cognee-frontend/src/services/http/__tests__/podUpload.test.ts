import podUpload, { podUploadWithRetry } from "../podUpload";
import { HttpError } from "../errors";

// The XHR transport had no coverage at all: the hook tests build their instance
// as `{ name, fetch }` with no `upload`, so every one of them takes rememberData's
// fetch fallback and this file never executes. These drive it directly.

interface FakeXhrState {
  method?: string;
  url?: string;
  headers: Record<string, string>;
  body?: unknown;
  aborted: boolean;
  timeout: number;
}

const xhrs: FakeXhrState[] = [];
// How each successive send() should resolve. Shifted per request so a retry
// test can queue "429 then 200".
let outcomes: Array<() => void> = [];

class FakeXhr {
  static instances: FakeXhr[] = [];

  state: FakeXhrState = { headers: {}, aborted: false, timeout: 0 };
  status = 0;
  statusText = "";
  responseText = "";
  responseType = "";
  upload: { onprogress?: (e: ProgressEvent) => void } = {};
  onload?: () => void;
  onerror?: () => void;
  ontimeout?: () => void;
  onabort?: () => void;
  private responseHeaders: Record<string, string> = {};

  set timeout(ms: number) {
    this.state.timeout = ms;
  }
  get timeout(): number {
    return this.state.timeout;
  }

  open(method: string, url: string): void {
    this.state.method = method;
    this.state.url = url;
  }

  setRequestHeader(key: string, value: string): void {
    this.state.headers[key] = value;
  }

  getResponseHeader(name: string): string | null {
    return this.responseHeaders[name] ?? null;
  }

  send(body: unknown): void {
    this.state.body = body;
    xhrs.push(this.state);
    FakeXhr.instances.push(this);
    const outcome = outcomes.shift();
    // Async, like a real request: the promise must be pending first.
    setTimeout(() => outcome?.call(this), 0);
  }

  abort(): void {
    this.state.aborted = true;
    this.onabort?.();
  }

  // ── helpers used by the queued outcomes ────────────────────────────────────
  succeed(text = '{"status":"ok"}'): void {
    this.status = 200;
    this.statusText = "OK";
    this.responseText = text;
    this.onload?.();
  }

  fail(status: number, text: string, headers: Record<string, string> = {}): void {
    this.status = status;
    this.statusText = "Error";
    this.responseText = text;
    this.responseHeaders = headers;
    this.onload?.();
  }

  emitProgress(loaded: number, total: number, lengthComputable = true): void {
    this.upload.onprogress?.({ loaded, total, lengthComputable } as ProgressEvent);
  }
}

const original = global.XMLHttpRequest;

beforeEach(() => {
  xhrs.length = 0;
  FakeXhr.instances.length = 0;
  outcomes = [];
  jest.useFakeTimers();
  (global as unknown as { XMLHttpRequest: unknown }).XMLHttpRequest = FakeXhr;
});

afterEach(() => {
  jest.useRealTimers();
  (global as unknown as { XMLHttpRequest: unknown }).XMLHttpRequest = original;
});

/** Run pending timers until `promise` settles, so fake timers don't deadlock. */
async function settle<T>(promise: Promise<T>): Promise<T> {
  const raced = promise.catch((e: unknown) => Promise.reject(e));
  await Promise.resolve();
  jest.runOnlyPendingTimers();
  await Promise.resolve();
  return raced;
}

describe("podUpload", () => {
  it("POSTs with the api key and resolves a Response on 2xx", async () => {
    outcomes = [
      function (this: FakeXhr) {
        this.succeed('{"status":"ok","pipeline_run_id":"run-1"}');
      },
    ];

    const res = await settle(podUpload("https://pod/api/v1/remember", "key-123", new FormData()));

    expect(res.status).toBe(200);
    await expect(res.json()).resolves.toEqual({ status: "ok", pipeline_run_id: "run-1" });
    expect(xhrs[0].method).toBe("POST");
    expect(xhrs[0].url).toBe("https://pod/api/v1/remember");
    expect(xhrs[0].headers["X-Api-Key"]).toBe("key-123");
  });

  it("reports upload progress from lengthComputable events only", async () => {
    const onProgress = jest.fn();
    outcomes = [
      function (this: FakeXhr) {
        this.emitProgress(50, 100);
        this.emitProgress(999, 0, false); // unmeasurable — must be ignored
        this.emitProgress(100, 100);
        this.succeed();
      },
    ];

    await settle(podUpload("https://pod/x", "k", new FormData(), { onProgress }));

    expect(onProgress.mock.calls).toEqual([
      [50, 100],
      [100, 100],
    ]);
  });

  it("rejects with an HttpError carrying status, parsed body and Retry-After", async () => {
    outcomes = [
      function (this: FakeXhr) {
        this.fail(429, '{"detail":"busy"}', { "Retry-After": "7" });
      },
    ];

    const error = await settle(podUpload("https://pod/x", "k", new FormData())).catch((e) => e);

    expect(error).toBeInstanceOf(HttpError);
    expect(error.status).toBe(429);
    expect(error.message).toBe("busy");
    expect(error.retryAfter).toBe("7");
  });

  it("surfaces a non-JSON error body as the message", async () => {
    outcomes = [
      function (this: FakeXhr) {
        this.fail(502, "upstream exploded");
      },
    ];

    const error = await settle(podUpload("https://pod/x", "k", new FormData())).catch((e) => e);

    expect(error).toBeInstanceOf(HttpError);
    expect(error.message).toBe("upstream exploded");
    expect(error.body).toBe("upstream exploded");
  });

  it("maps timeout to the same message the shared fetch client uses", async () => {
    outcomes = [
      function (this: FakeXhr) {
        this.ontimeout?.();
      },
    ];

    const error = await settle(
      podUpload("https://pod/x", "k", new FormData(), { timeoutMs: 1000 }),
    ).catch((e) => e);

    expect(error.message).toBe("Request timed out.");
    expect(xhrs[0].timeout).toBe(1000);
  });

  it("rejects immediately with AbortError when the signal is already aborted", async () => {
    const controller = new AbortController();
    controller.abort();

    const error = await podUpload("https://pod/x", "k", new FormData(), {
      signal: controller.signal,
    }).catch((e) => e);

    expect(error.name).toBe("AbortError");
    // Nothing was ever sent.
    expect(xhrs).toHaveLength(0);
  });

  it("aborts the request when the signal fires mid-flight", async () => {
    const controller = new AbortController();
    outcomes = [function (this: FakeXhr) { /* never settles on its own */ }];

    const pending = podUpload("https://pod/x", "k", new FormData(), {
      signal: controller.signal,
    });
    await Promise.resolve();
    jest.runOnlyPendingTimers();
    controller.abort();

    const error = await pending.catch((e) => e);
    expect(error.name).toBe("AbortError");
    expect(xhrs[0].aborted).toBe(true);
  });
});

describe("podUploadWithRetry", () => {
  it("honours the Retry-After header over the body field", async () => {
    outcomes = [
      function (this: FakeXhr) {
        // Header says 2s, body says 30s. The header is the standard and is what
        // the shared client reads — before the fix only the body was consulted,
        // so the header was silently ignored.
        this.fail(429, '{"retry_after_seconds":30}', { "Retry-After": "2" });
      },
      function (this: FakeXhr) {
        this.succeed();
      },
    ];

    const setTimeoutSpy = jest.spyOn(global, "setTimeout");
    const promise = podUploadWithRetry("https://pod/x", "k", () => new FormData());

    await Promise.resolve();
    jest.runOnlyPendingTimers();
    await Promise.resolve();
    const waits = setTimeoutSpy.mock.calls.map((c) => c[1]).filter((ms) => ms === 2000);
    jest.runOnlyPendingTimers();
    await Promise.resolve();
    jest.runOnlyPendingTimers();
    await promise;

    expect(waits).toContain(2000);
    expect(xhrs).toHaveLength(2);
    setTimeoutSpy.mockRestore();
  });

  it("falls back to the body field when no header is present", async () => {
    outcomes = [
      function (this: FakeXhr) {
        this.fail(429, '{"retry_after_seconds":3}');
      },
      function (this: FakeXhr) {
        this.succeed();
      },
    ];

    const setTimeoutSpy = jest.spyOn(global, "setTimeout");
    const promise = podUploadWithRetry("https://pod/x", "k", () => new FormData());

    await Promise.resolve();
    jest.runOnlyPendingTimers();
    await Promise.resolve();
    const waits = setTimeoutSpy.mock.calls.map((c) => c[1]).filter((ms) => ms === 3000);
    jest.runOnlyPendingTimers();
    await Promise.resolve();
    jest.runOnlyPendingTimers();
    await promise;

    expect(waits).toContain(3000);
    setTimeoutSpy.mockRestore();
  });

  it("rebuilds the FormData for every attempt — a sent body is consumed", async () => {
    outcomes = [
      function (this: FakeXhr) {
        this.fail(429, "{}", { "Retry-After": "0" });
      },
      function (this: FakeXhr) {
        this.succeed();
      },
    ];

    const makeBody = jest.fn(() => new FormData());
    const promise = podUploadWithRetry("https://pod/x", "k", makeBody);

    await Promise.resolve();
    jest.runOnlyPendingTimers();
    await Promise.resolve();
    jest.runOnlyPendingTimers();
    await Promise.resolve();
    jest.runOnlyPendingTimers();
    await promise;

    expect(makeBody).toHaveBeenCalledTimes(2);
    expect(xhrs[0].body).not.toBe(xhrs[1].body);
  });

  it("does not retry a non-429 failure", async () => {
    outcomes = [
      function (this: FakeXhr) {
        this.fail(500, '{"detail":"boom"}');
      },
    ];

    const error = await settle(
      podUploadWithRetry("https://pod/x", "k", () => new FormData()),
    ).catch((e) => e);

    expect(error.status).toBe(500);
    expect(xhrs).toHaveLength(1);
  });
});
