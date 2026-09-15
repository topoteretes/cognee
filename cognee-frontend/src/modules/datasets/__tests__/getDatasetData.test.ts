import type { CogneeInstance } from "../../instances/types";
import getDatasetData, { getAllDatasetData, getDatasetDataCount } from "../getDatasetData";

const fetch = jest.fn();
const instance = { name: "test", instanceId: "test", fetch } as CogneeInstance;
const response = (body: unknown, ok = true) => ({ ok, status: ok ? 200 : 404, json: async () => body }) as Response;
beforeEach(() => fetch.mockReset());

test.each([{ count: -1 }, { count: 1.5 }, {}, { count: "100" }])("rejects invalid counts: %j", async body => {
  fetch.mockResolvedValue(response(body));
  await expect(getDatasetDataCount("a", instance)).rejects.toThrow();
});

test("count errors cannot masquerade as an empty dataset", async () => {
  fetch.mockResolvedValue(response({ count: 0 }, false));
  await expect(getDatasetDataCount("a", instance)).rejects.toThrow();
  fetch.mockResolvedValue(response({ count: 250 }));
  await expect(getDatasetDataCount("a", instance)).resolves.toBe(250);
});

test("rejects non-array document responses", async () => {
  fetch.mockResolvedValue(response({ message: "missing" }));
  await expect(getDatasetData("a", instance)).rejects.toThrow();
});

test("explicit full traversal follows offsets past the server maximum", async () => {
  fetch.mockResolvedValueOnce(response(Array.from({ length: 1000 }, (_, i) => ({ id: i }))))
    .mockResolvedValueOnce(response([{ id: 1000 }]));
  expect(await getAllDatasetData("a", instance)).toHaveLength(1001);
  expect(fetch.mock.calls[1][0]).toContain("limit=1000&offset=1000");
});
