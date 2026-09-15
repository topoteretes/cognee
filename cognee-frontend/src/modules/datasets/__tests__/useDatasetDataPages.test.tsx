import { act, renderHook } from "@testing-library/react";
import type { CogneeInstance } from "../../instances/types";
import useDatasetDataPages from "../useDatasetDataPages";

const page = (start: number, count: number) => Array.from({ length: count }, (_, i) => ({ id: String(start + i) }));
const response = (body: unknown) => ({ ok: true, json: async () => body }) as Response;
function setup(maxRows = Infinity) {
  const fetch = jest.fn();
  const instance = { name: "test", instanceId: "test", fetch } as CogneeInstance;
  return { fetch, ...renderHook(() => useDatasetDataPages<{ id: string }>(instance, maxRows)) };
}

test("loads one bounded page, then reaches documents beyond 100 on demand", async () => {
  const { fetch, result } = setup();
  fetch.mockResolvedValueOnce(response(page(0, 100))).mockResolvedValueOnce(response(page(100, 25)));
  await act(async () => { await result.current.load("a"); });
  expect(result.current.data).toHaveLength(100);
  expect(fetch).toHaveBeenCalledTimes(1);
  expect(result.current.hasMore).toBe(true);
  await act(async () => { result.current.loadMore(); });
  expect(fetch.mock.calls[1][0]).toContain("limit=100&offset=100");
  expect(result.current.data).toHaveLength(125);
  expect(result.current.hasMore).toBe(false);
});

test("failed next page preserves documents and retries the same offset", async () => {
  const { fetch, result } = setup();
  fetch.mockResolvedValueOnce(response(page(0, 100)))
    .mockRejectedValueOnce(new Error("offline")).mockResolvedValueOnce(response(page(100, 2)));
  await act(async () => { await result.current.load("a"); });
  await act(async () => { result.current.loadMore(); });
  expect(result.current.data).toHaveLength(100);
  expect(result.current.error).toBe(true);
  await act(async () => { result.current.loadMore(); });
  expect(fetch.mock.calls[2][0]).toContain("offset=100");
  expect(result.current.data).toHaveLength(102);
  expect(result.current.error).toBe(false);
});

test("a late response cannot replace the newly selected dataset", async () => {
  const { fetch, result } = setup();
  let resolve!: (value: Response) => void;
  fetch.mockReturnValueOnce(new Promise<Response>(r => { resolve = r; }))
    .mockResolvedValueOnce(response([{ id: "b" }]));
  let first!: Promise<void>;
  act(() => { first = result.current.load("a"); });
  await act(async () => { await result.current.load("b"); });
  await act(async () => { resolve(response(page(0, 100))); await first; });
  expect(result.current.data).toEqual([{ id: "b" }]);
  expect(result.current.hasMore).toBe(false);
});

test("refresh resets paging and deletion adjusts the next offset", async () => {
  const { fetch, result } = setup();
  fetch.mockResolvedValue(response(page(0, 100)));
  await act(async () => { await result.current.load("a"); });
  act(() => { result.current.setData(rows => rows.slice(1)); });
  await act(async () => { result.current.loadMore(); });
  expect(fetch.mock.calls[1][0]).toContain("offset=99");
  await act(async () => { await result.current.load("a"); });
  expect(fetch.mock.calls[2][0]).toContain("offset=0");
  expect(result.current.data).toHaveLength(100);
});

test("reset cancels pending results and removes the previous dataset's paging controls", async () => {
  const { fetch, result } = setup();
  let resolve!: (value: Response) => void;
  fetch.mockReturnValueOnce(new Promise<Response>(r => { resolve = r; }));
  let pending!: Promise<void>;
  act(() => { pending = result.current.load("a"); });
  act(() => { result.current.reset(); });
  await act(async () => { resolve(response(page(0, 100))); await pending; });
  expect(result.current.data).toEqual([]);
  expect(result.current.loading).toBe(false);
  expect(result.current.hasMore).toBe(false);
});


test("advances by consumed rows, deduplicating both within and across pages", async () => {
  const { fetch, result } = setup();
  fetch.mockResolvedValueOnce(response(page(0, 100)))
    .mockResolvedValueOnce(response([...page(50, 99), { id: "148" }]))
    .mockResolvedValueOnce(response([{ id: "last" }]));
  await act(async () => { await result.current.load("a"); });
  await act(async () => { result.current.loadMore(); });
  expect(result.current.data).toHaveLength(149);
  await act(async () => { result.current.loadMore(); });
  expect(fetch.mock.calls[2][0]).toContain("offset=200");
  expect(result.current.data.at(-1)).toEqual({ id: "last" });
});

test("stops when the server returns only duplicates or an empty page", async () => {
  for (const next of [page(0, 100), []]) {
    const { fetch, result, unmount } = setup();
    fetch.mockResolvedValueOnce(response(page(0, 100))).mockResolvedValueOnce(response(next));
    await act(async () => { await result.current.load("a"); });
    await act(async () => { result.current.loadMore(); });
    expect(result.current.hasMore).toBe(false);
    await act(async () => { result.current.loadMore(); });
    expect(fetch).toHaveBeenCalledTimes(2);
    unmount();
  }
});

test("a stale append cannot mix documents into another dataset", async () => {
  const { fetch, result } = setup();
  let resolve!: (value: Response) => void;
  fetch.mockResolvedValueOnce(response(page(0, 100)))
    .mockReturnValueOnce(new Promise<Response>(r => { resolve = r; }))
    .mockResolvedValueOnce(response([{ id: "b" }]));
  await act(async () => { await result.current.load("a"); });
  act(() => { result.current.loadMore(); });
  await act(async () => { await result.current.load("b"); });
  await act(async () => { resolve(response(page(100, 100))); });
  expect(result.current.data).toEqual([{ id: "b" }]);
});

test("ignores a duplicate request before loading state has rendered", async () => {
  const { fetch, result } = setup();
  fetch.mockResolvedValueOnce(response(page(0, 100))).mockResolvedValueOnce(response(page(100, 100)));
  await act(async () => { await result.current.load("a"); });
  await act(async () => { result.current.loadMore(); result.current.loadMore(); });
  expect(fetch).toHaveBeenCalledTimes(2);
});

test("bounds requested and stored rows independently of a stale total", async () => {
  const { fetch, result } = setup(150);
  fetch.mockImplementation((url: string) => Promise.resolve(response(
    url.endsWith("/count") ? { count: 1 } : url.includes("offset=100") ? page(100, 100) : page(0, 100)
  )));
  await act(async () => { await result.current.load("a"); });
  await act(async () => { result.current.loadMore(); });
  expect(fetch).toHaveBeenCalledWith(expect.stringContaining("limit=50&offset=100"), expect.anything());
  expect(result.current.data).toHaveLength(150);
  await act(async () => { result.current.loadMore(); });
  expect(fetch).toHaveBeenCalledTimes(3);
});

test("a count failure does not discard readable documents", async () => {
  const { fetch, result } = setup(2000);
  fetch.mockImplementation((url: string) => url.endsWith("/count")
    ? Promise.reject(new Error("count unavailable")) : Promise.resolve(response(page(0, 100))));
  await act(async () => { await result.current.load("a"); });
  expect(result.current.total).toBeNull();
  expect(result.current.data).toHaveLength(100);
  expect(result.current.error).toBe(false);
});

test("caps an oversized first response from an older server", async () => {
  const { fetch, result } = setup(2000);
  fetch.mockImplementation((url: string) => Promise.resolve(response(
    url.endsWith("/count") ? { count: 171828 } : page(0, 171828)
  )));
  await act(async () => { await result.current.load("a"); });
  expect(result.current.total).toBe(171828);
  expect(result.current.data).toHaveLength(2000);
  await act(async () => { result.current.loadMore(); });
  expect(fetch).toHaveBeenCalledTimes(2);
});


test("a failed refresh retries from the first page while preserving the visible rows", async () => {
  const { fetch, result } = setup();
  fetch.mockResolvedValueOnce(response(page(0, 2)))
    .mockRejectedValueOnce(new Error("offline"))
    .mockResolvedValueOnce(response([{ id: "fresh" }]));
  await act(async () => { await result.current.load("a"); });
  await act(async () => { await result.current.load("a"); });
  expect(result.current.data).toHaveLength(2);
  expect(result.current.error).toBe(true);
  await act(async () => { result.current.loadMore(); });
  expect(fetch.mock.calls[2][0]).toContain("offset=0");
  expect(result.current.data).toEqual([{ id: "fresh" }]);
});


test("loadMore keeps its identity while pages arrive", async () => {
  const { fetch, result } = setup();
  fetch.mockResolvedValue(response(page(0, 100)));
  const loadMore = result.current.loadMore;
  await act(async () => { await result.current.load("a"); });
  expect(result.current.loadMore).toBe(loadMore);
});
