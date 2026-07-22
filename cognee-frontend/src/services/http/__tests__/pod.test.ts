import { createPodClient } from "../pod";

function mockFetch(status = 200, body: unknown = {}) {
  global.fetch = jest.fn().mockResolvedValue(
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    }),
  );
}

function calledUrl(): string {
  return (global.fetch as jest.Mock).mock.calls[0][0] as string;
}

function calledInit(): RequestInit {
  return (global.fetch as jest.Mock).mock.calls[0][1] as RequestInit;
}

afterEach(() => jest.restoreAllMocks());

// ─── URL building ─────────────────────────────────────────────────────────────

describe("createPodClient — URL building", () => {
  it("builds correct URL from string path", async () => {
    mockFetch();
    const client = createPodClient("https://pod.example.com", "key");
    await client.fetch("/v1/datasets");
    expect(calledUrl()).toBe("https://pod.example.com/api/v1/datasets");
  });

  it("strips trailing slash from serviceUrl", async () => {
    mockFetch();
    const client = createPodClient("https://pod.example.com/", "key");
    await client.fetch("/datasets");
    expect(calledUrl()).toBe("https://pod.example.com/api/datasets");
  });

  it("preserves query string from string path", async () => {
    mockFetch();
    const client = createPodClient("https://pod.example.com", "key");
    await client.fetch("/datasets?page=2&limit=10");
    expect(calledUrl()).toBe("https://pod.example.com/api/datasets?page=2&limit=10");
  });

  it("extracts pathname + search from URL object", async () => {
    mockFetch();
    const client = createPodClient("https://pod.example.com", "key");
    await client.fetch(new URL("https://other.host/v1/datasets?page=2"));
    expect(calledUrl()).toBe("https://pod.example.com/api/v1/datasets?page=2");
  });

  it("extracts pathname + search from Request object", async () => {
    mockFetch();
    const client = createPodClient("https://pod.example.com", "key");
    await client.fetch(new Request("https://other.host/v1/datasets?filter=x"));
    expect(calledUrl()).toBe("https://pod.example.com/api/v1/datasets?filter=x");
  });
});

// ─── headers ──────────────────────────────────────────────────────────────────

describe("createPodClient — headers", () => {
  it("injects X-Api-Key header", async () => {
    mockFetch();
    const client = createPodClient("https://pod.example.com", "secret-key");
    await client.fetch("/path");
    const headers = calledInit().headers as Record<string, string>;
    expect(headers["X-Api-Key"]).toBe("secret-key");
  });

  it("X-Api-Key cannot be overridden by caller", async () => {
    mockFetch();
    const client = createPodClient("https://pod.example.com", "factory-key");
    await client.fetch("/path", { headers: { "X-Api-Key": "caller-key" } });
    const headers = calledInit().headers as Record<string, string>;
    expect(headers["X-Api-Key"]).toBe("factory-key");
  });

  it("merges caller headers alongside X-Api-Key", async () => {
    mockFetch();
    const client = createPodClient("https://pod.example.com", "key");
    await client.fetch("/path", { headers: { "X-Custom": "yes" } });
    const headers = calledInit().headers as Record<string, string>;
    expect(headers["X-Custom"]).toBe("yes");
    expect(headers["X-Api-Key"]).toBe("key");
  });
});

// ─── credentials ─────────────────────────────────────────────────────────────

describe("createPodClient — credentials", () => {
  it("defaults to omit", async () => {
    mockFetch();
    const client = createPodClient("https://pod.example.com", "key");
    await client.fetch("/path");
    expect(calledInit().credentials).toBe("omit");
  });

  it("caller can override credentials", async () => {
    mockFetch();
    const client = createPodClient("https://pod.example.com", "key");
    await client.fetch("/path", { credentials: "include" });
    expect(calledInit().credentials).toBe("include");
  });
});

// ─── failure reporting ────────────────────────────────────────────────────────

describe("createPodClient — failure reporting", () => {
  it("forwards a failed pod request to /api/log with the POD-API tag", async () => {
    // Pod requests run entirely in the browser, so a failure here is invisible
    // in Vercel's server logs unless explicitly forwarded — this is what
    // reportPodFailure (wired via http.setLogger in pod.ts) is for.
    global.fetch = jest.fn()
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce(new Response("{}", { status: 200 }));

    const client = createPodClient("https://tenant-abc.example.com", "key");
    await expect(client.fetch("/v1/datasets/")).rejects.toThrow();

    // Give the fire-and-forget /api/log POST a tick to fire.
    await new Promise((resolve) => setTimeout(resolve, 0));

    const calls = (global.fetch as jest.Mock).mock.calls;
    const logCall = calls.find(([url]) => String(url).includes("/api/log"));
    expect(logCall).toBeDefined();
    const logBody = JSON.parse((logCall?.[1] as RequestInit).body as string);
    expect(logBody).toMatchObject({
      level: "error",
      tag: "POD-API",
      url: "https://tenant-abc.example.com/api/v1/datasets/",
      method: "GET",
    });
  });
});
