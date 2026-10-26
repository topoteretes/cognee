import getDatasetDataCount from "../getDatasetDataCount";
import { createHttpClient } from "@/services/http/client";
import type { CogneeInstance } from "@/modules/instances/types";

function row(id: string) {
  return { id };
}

// Every CogneeInstance.fetch implementation is backed by http.request, which
// rejects with HttpError on any non-ok response instead of resolving with
// `ok: false` (see toHttpError in services/http/client.ts). Stubbing
// global.fetch and driving requests through a real client, instead of a fake
// `{ ok: false }` object, is what makes these tests actually exercise the
// 404-catch path getDatasetDataCount relies on.
function mockFetchResponses(responses: Array<{ status: number; body: unknown }>): void {
  let call = 0;
  global.fetch = jest.fn().mockImplementation(() => {
    const r = responses[Math.min(call++, responses.length - 1)];
    return Promise.resolve(
      new Response(JSON.stringify(r.body), {
        status: r.status,
        headers: { "Content-Type": "application/json" },
      }),
    );
  });
}

function instanceOverRealClient(): CogneeInstance {
  const http = createHttpClient();
  return {
    name: "test",
    instanceId: "tenant-1",
    fetch: (input, init) => http.request(input as string, init),
  };
}

afterEach(() => {
  jest.restoreAllMocks();
});

describe("getDatasetDataCount", () => {
  it("returns the count from /data/count when the pod has it", async () => {
    mockFetchResponses([{ status: 200, body: { count: 42 } }]);
    const instance = instanceOverRealClient();

    await expect(getDatasetDataCount("ds-1", instance)).resolves.toBe(42);
    expect(global.fetch).toHaveBeenCalledTimes(1);
  });

  it("hands the caller's abort signal to the count request", async () => {
    const { signal } = new AbortController();
    const instance: CogneeInstance = {
      name: "test",
      instanceId: "tenant-1",
      fetch: jest.fn().mockResolvedValue({ json: () => Promise.resolve({ count: 7 }) }),
    };

    await expect(getDatasetDataCount("ds-1", instance, signal)).resolves.toBe(7);
    expect(instance.fetch).toHaveBeenCalledWith("/v1/datasets/ds-1/data/count", { signal });
  });

  it("falls back to paging through /data when /data/count 404s (pre-1.6.1 pod)", async () => {
    mockFetchResponses([
      { status: 404, body: { message: "Not Found" } }, // /data/count
      { status: 200, body: [row("a"), row("b"), row("c")] }, // /data page 0
    ]);
    const instance = instanceOverRealClient();

    await expect(getDatasetDataCount("ds-1", instance)).resolves.toBe(3);
  });

  it("falls back to paging through /data when /data/count 405s (1.6.0 pod matching its DELETE /data/{data_id} route)", async () => {
    mockFetchResponses([
      { status: 405, body: { detail: "Method Not Allowed" } }, // /data/count
      { status: 200, body: [row("a"), row("b")] }, // /data page 0
    ]);
    const instance = instanceOverRealClient();

    await expect(getDatasetDataCount("ds-1", instance)).resolves.toBe(2);
  });

  it("falls back the same way when /data/count returns a body without a numeric count", async () => {
    mockFetchResponses([
      { status: 200, body: {} },
      { status: 200, body: [row("a")] },
    ]);
    const instance = instanceOverRealClient();

    await expect(getDatasetDataCount("ds-1", instance)).resolves.toBe(1);
  });

  it("rethrows any other HttpError instead of silently falling back", async () => {
    mockFetchResponses([{ status: 500, body: { message: "boom" } }]);
    const instance = instanceOverRealClient();

    await expect(getDatasetDataCount("ds-1", instance)).rejects.toMatchObject({ name: "HttpError", status: 500 });
    expect(global.fetch).toHaveBeenCalledTimes(1); // never falls through to /data
  });

  it("returns null instead of a capped, non-exact number", async () => {
    let dataCall = 0;
    global.fetch = jest.fn().mockImplementation((url: string) => {
      if (typeof url === "string" && url.endsWith("/data/count")) {
        return Promise.resolve(new Response(JSON.stringify({ message: "Not Found" }), { status: 404 }));
      }
      const page = dataCall++;
      const batch = Array.from({ length: 1000 }, (_, i) => row(`pg${page}-${i}`));
      return Promise.resolve(new Response(JSON.stringify(batch), { status: 200 }));
    });
    jest.spyOn(console, "warn").mockImplementation(() => {});
    const instance = instanceOverRealClient();

    await expect(getDatasetDataCount("ds-1", instance)).resolves.toBeNull();
  });
});
