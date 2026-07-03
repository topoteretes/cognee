import { describe, expect, it } from "vitest";

import { CogneeApiError } from "./errors";
import { HttpCogneeClient } from "./httpClient";

interface Recorded {
  url: string;
  init: RequestInit;
}

/** Build a fetch stub that records calls and returns a canned response. */
function stubFetch(response: () => Response): { fetch: typeof fetch; calls: Recorded[] } {
  const calls: Recorded[] = [];
  const fetchImpl = (async (url: string | URL | Request, init?: RequestInit) => {
    calls.push({ url: String(url), init: init ?? {} });
    return response();
  }) as unknown as typeof fetch;
  return { fetch: fetchImpl, calls };
}

function headerValue(init: RequestInit, name: string): string | undefined {
  const headers = init.headers as Record<string, string> | undefined;
  return headers?.[name];
}

describe("HttpCogneeClient.recall", () => {
  it("posts a snake_case body to /api/v1/recall with the API key header", async () => {
    const { fetch, calls } = stubFetch(
      () =>
        new Response(
          JSON.stringify([
            { source: "graph", kind: "graph_completion", search_type: "GRAPH_COMPLETION", text: "hi" },
          ]),
          { status: 200 },
        ),
    );
    const client = new HttpCogneeClient({ endpoint: "http://localhost:8011/", apiKey: "secret", fetch });

    const results = await client.recall("q", {
      datasets: ["vscode_abc"],
      includeReferences: true,
      topK: 5,
      searchType: null,
    });

    expect(calls).toHaveLength(1);
    const call = calls[0];
    expect(call.url).toBe("http://localhost:8011/api/v1/recall");
    expect(call.init.method).toBe("POST");
    expect(headerValue(call.init, "X-Api-Key")).toBe("secret");
    expect(headerValue(call.init, "Content-Type")).toBe("application/json");

    const body = JSON.parse(call.init.body as string);
    expect(body).toMatchObject({
      query: "q",
      datasets: ["vscode_abc"],
      include_references: true,
      top_k: 5,
      search_type: null,
    });
    expect(results).toHaveLength(1);
  });

  it("omits the API key header when no key is configured", async () => {
    const { fetch, calls } = stubFetch(() => new Response("[]", { status: 200 }));
    const client = new HttpCogneeClient({ endpoint: "http://localhost:8011", fetch });

    await client.recall("q");

    expect(headerValue(calls[0].init, "X-Api-Key")).toBeUndefined();
  });

  it("raises CogneeApiError on a non-2xx response", async () => {
    const { fetch } = stubFetch(() => new Response(JSON.stringify({ error: "boom" }), { status: 409 }));
    const client = new HttpCogneeClient({ endpoint: "http://localhost:8011", fetch });

    await expect(client.recall("q")).rejects.toBeInstanceOf(CogneeApiError);
  });
});

describe("HttpCogneeClient.remember", () => {
  it("posts multipart form data with the dataset name to /api/v1/remember", async () => {
    const { fetch, calls } = stubFetch(() => new Response(JSON.stringify({ status: "completed" }), { status: 200 }));
    const client = new HttpCogneeClient({ endpoint: "http://localhost:8011", fetch });

    const result = await client.remember("hello world", {
      datasetName: "vscode_abc",
      filename: "note.txt",
    });

    const call = calls[0];
    expect(call.url).toBe("http://localhost:8011/api/v1/remember");
    expect(call.init.method).toBe("POST");
    const form = call.init.body as FormData;
    expect(form).toBeInstanceOf(FormData);
    expect(form.get("datasetName")).toBe("vscode_abc");
    expect(result.status).toBe("completed");
  });
});

describe("HttpCogneeClient.forget", () => {
  it("posts the dataset and memory_only flag to /api/v1/forget", async () => {
    const { fetch, calls } = stubFetch(() => new Response(JSON.stringify({ ok: true }), { status: 200 }));
    const client = new HttpCogneeClient({ endpoint: "http://localhost:8011", fetch });

    await client.forget({ dataset: "vscode_abc", memoryOnly: true });

    const body = JSON.parse(calls[0].init.body as string);
    expect(body).toEqual({ dataset: "vscode_abc", memory_only: true });
  });
});
