import getDatasetData from "../getDatasetData";
import type { CogneeInstance } from "@/modules/instances/types";

function instanceReturning(body: unknown): CogneeInstance & { fetch: jest.Mock } {
  return {
    name: "test",
    instanceId: "tenant-1",
    fetch: jest.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve(body) }),
  };
}

describe("getDatasetData", () => {
  it("sends the default limit and offset when none are given", async () => {
    const instance = instanceReturning([]);
    await getDatasetData("ds-1", instance);
    expect(instance.fetch).toHaveBeenCalledWith("/v1/datasets/ds-1/data?limit=100&offset=0", { signal: undefined });
  });

  it("sends the caller's limit and offset", async () => {
    const instance = instanceReturning([]);
    await getDatasetData("ds-1", instance, { limit: 1000, offset: 2000 });
    expect(instance.fetch).toHaveBeenCalledWith("/v1/datasets/ds-1/data?limit=1000&offset=2000", { signal: undefined });
  });

  it("hands the caller's abort signal to the request", async () => {
    const instance = instanceReturning([]);
    const { signal } = new AbortController();
    await getDatasetData("ds-1", instance, { signal });
    expect(instance.fetch).toHaveBeenCalledWith(expect.any(String), { signal });
  });

  it("resolves with the parsed JSON body", async () => {
    const rows = [{ id: "a" }, { id: "b" }];
    const instance = instanceReturning(rows);
    await expect(getDatasetData("ds-1", instance)).resolves.toEqual(rows);
  });

  it("propagates a rejected fetch instead of swallowing it", async () => {
    const instance: CogneeInstance = {
      name: "test",
      instanceId: "tenant-1",
      fetch: jest.fn().mockRejectedValue(new Error("network down")),
    };
    await expect(getDatasetData("ds-1", instance)).rejects.toThrow("network down");
  });
});
