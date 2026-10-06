import type { CogneeInstance } from "../../instances/types";
import waitForDatasetDocs, { WaitForDocsTimeoutError } from "../waitForDatasetDocs";

jest.mock("@/utils/monitoring", () => ({ captureException: jest.fn() }));
const rows = Array.from({ length: 1001 }, (_, id) => ({ id: String(id) }));
const response = (body: unknown) => ({ ok: true, json: async () => body }) as Response;

function setup(background = false) {
  let counts = 0;
  const fetch = jest.fn(async (url: string) => response(url.endsWith("/count")
    ? { count: background && counts++ === 0 ? 0 : rows.length }
    : url.includes("offset=1000") ? rows.slice(1000) : rows.slice(0, 1000)));
  return { name: "test", instanceId: "test", fetch } as CogneeInstance;
}

test("foreground completion returns every document beyond the page limit", async () => {
  await expect(waitForDatasetDocs("a", setup(), 1001)).resolves.toEqual(rows);
});

test("background completion retains the full collection callback contract", async () => {
  const done = jest.fn();
  await expect(waitForDatasetDocs("a", setup(true), 1001, {
    timeoutMs: 0, intervalMs: 1, backgroundTimeoutMs: 1000, onTimeout: done,
  })).rejects.toBeInstanceOf(WaitForDocsTimeoutError);
  await new Promise(resolve => setTimeout(resolve, 30));
  expect(done).toHaveBeenCalledWith(rows);
});
