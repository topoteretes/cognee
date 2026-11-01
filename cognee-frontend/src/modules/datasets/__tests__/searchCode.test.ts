import searchCode from "../searchCode";
import type { CogneeInstance } from "@/modules/instances/types";

const FACTS = {
  operation: "query_facts",
  facts: [{ id: "n1", kind: "symbol", name: "retry_with_backoff", file: "src/net.py", line: 42 }],
  total: 1,
  has_more: false,
};

function instanceAnswering(json: unknown): CogneeInstance & { fetch: jest.Mock } {
  const res = { ok: true, status: 200, json: () => Promise.resolve(json) } as unknown as Response;
  return { name: "test", instanceId: "tenant-1", fetch: jest.fn().mockResolvedValue(res) };
}

function sent(instance: { fetch: jest.Mock }): { path: string; method?: string; body: Record<string, unknown> } {
  const [path, init] = instance.fetch.mock.calls[0] as [string, RequestInit];
  return { path, method: init.method, body: JSON.parse(init.body as string) as Record<string, unknown> };
}

const REQUEST = {
  query: "where is retry handled",
  codeQuery: { operation: "query_facts", name: "retry", limit: 50 } as const,
  datasetIds: ["ds-1"],
};

describe("searchCode", () => {
  // /v1/recall only honours code_query inside a "code" scope that is newer
  // than code search itself; this route has carried it from the start.
  it("posts to /v1/search, the route every pod that can search code accepts", async () => {
    const instance = instanceAnswering([]);
    await searchCode(instance, REQUEST);

    expect(sent(instance).path).toBe("/v1/search");
    expect(sent(instance).method).toBe("POST");
  });

  // Without it the pod defaults the operation to "explore" and resolves the
  // query text as a symbol name — the failure this module exists to prevent.
  it("always names the operation in codeQuery", async () => {
    const instance = instanceAnswering([]);
    await searchCode(instance, REQUEST);

    expect(sent(instance).body).toEqual({
      searchType: "CODE",
      query: "where is retry handled",
      codeQuery: { operation: "query_facts", name: "retry", limit: 50 },
      datasetIds: ["ds-1"],
    });
  });

  it("reads the payload out of the per-dataset entry the pod wraps it in", async () => {
    const instance = instanceAnswering([{ dataset_id: "ds-1", dataset_name: "repo", search_result: FACTS }]);

    await expect(searchCode(instance, REQUEST)).resolves.toEqual([FACTS]);
  });

  it("keeps one payload per dataset when several were searched", async () => {
    const other = { ...FACTS, total: 0, facts: [] };
    const instance = instanceAnswering([
      { dataset_id: "ds-1", dataset_name: "a", search_result: FACTS },
      { dataset_id: "ds-2", dataset_name: "b", search_result: other },
    ]);

    await expect(searchCode(instance, REQUEST)).resolves.toEqual([FACTS, other]);
  });

  // The route is typed `list[SearchResult] | list`, so these are in contract.
  it.each([
    ["a list of payloads under search_result", [{ search_result: [FACTS] }]],
    ["a bare list of payloads", [FACTS]],
    ["a bare payload", FACTS],
  ])("also accepts %s", async (_shape, json) => {
    await expect(searchCode(instanceAnswering(json), REQUEST)).resolves.toEqual([FACTS]);
  });

  it.each([
    ["an empty list", []],
    ["entries with no payload", [{ dataset_id: "ds-1", search_result: null }]],
    ["a completion string, which is some other search type's answer", [{ search_result: ["prose"] }]],
    ["null", null],
  ])("returns nothing for %s rather than something unrenderable", async (_shape, json) => {
    await expect(searchCode(instanceAnswering(json), REQUEST)).resolves.toEqual([]);
  });

  // The pod client rejects on a non-2xx; the message is what tells a missing
  // operation from a real failure one layer up.
  it("lets the pod's error through untouched", async () => {
    const failure = new Error("Unsupported CODE operation 'architecture'; expected one of explore, query_facts.");
    const instance: CogneeInstance = { name: "test", instanceId: "t", fetch: jest.fn().mockRejectedValue(failure) };

    await expect(searchCode(instance, REQUEST)).rejects.toBe(failure);
  });
});
