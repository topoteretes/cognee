import { getCodeNeighbors } from "../searchCode";
import type { CogneeInstance } from "@/modules/instances/types";

const anchor = { id: "module", type: "CodeModule", kind: "module", name: "src/users", doc: "Manages user data." };
const page = { operation: "neighbors", anchor, neighbors: [], total: 0, offset: 0, limit: 50, has_more: false };
function answering(data: unknown): CogneeInstance & { fetch: jest.Mock } {
  return { name: "test", instanceId: "t1", fetch: jest.fn().mockResolvedValue({ json: async () => data }) };
}

it("reads exact node details and forwards dataset scope and cancellation without changing server page size", async () => {
  const instance = answering([{ dataset_id: "d1", search_result: page }]);
  const signal = new AbortController().signal;
  await expect(getCodeNeighbors(instance, "d1", "module", 0, signal)).resolves.toEqual(page);
  const [path, init] = instance.fetch.mock.calls[0];
  expect(path).toBe("/v1/search");
  expect(init.signal).toBe(signal);
  expect(JSON.parse(init.body)).toEqual({
    searchType: "CODE", query: "", datasetIds: ["d1"],
    codeQuery: { operation: "neighbors", id: "module", direction: "both", offset: 0 },
  });
});

it.each([
  ["unsupported response", { operation: "explore", nodes: [] }],
  ["missing anchor", { ...page, anchor: null }],
  ["wrong anchor", { ...page, anchor: { ...anchor, id: "other" } }],
  ["unreadable description", { ...page, anchor: { ...anchor, description: {} } }],
  ["missing relationships", { ...page, neighbors: null }],
  ["invalid direction", { ...page, neighbors: [{ relation: "calls", direction: "reverse", node: anchor }], total: 1 }],
  ["page with no progress", { ...page, has_more: true, total: 10 }],
  ["silently omitted relationships", { ...page, has_more: false, total: 10 }],
])("rejects %s instead of presenting empty details", async (_label, data) => {
  await expect(getCodeNeighbors(answering(data), "d1", "module", 0)).rejects.toThrow();
});

it("keeps missing metadata as missing on a valid page", async () => {
  const data = { ...page, anchor: { id: "module", type: "CodeModule", kind: "module" } };
  await expect(getCodeNeighbors(answering(data), "d1", "module", 0)).resolves.toEqual(data);
});

it("preserves server failures so denied, removed and unsupported items can be distinguished", async () => {
  const failure = new Error("CODE could not resolve neighbors anchor id 'module'.");
  const instance = answering(null);
  instance.fetch.mockRejectedValue(failure);
  await expect(getCodeNeighbors(instance, "d1", "module", 0)).rejects.toBe(failure);
});
