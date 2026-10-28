import getAllDatasetData, { fetchAllDatasetDataPages } from "../getAllDatasetData";
import type { CogneeInstance } from "@/modules/instances/types";

function row(id: string) {
  return { id, name: `file-${id}` };
}

function instanceWithPages(pages: Array<Array<Record<string, unknown>>>): CogneeInstance & { fetch: jest.Mock } {
  const fetch = jest.fn().mockImplementation((url: string) => {
    const offset = Number(new URL(url, "http://pod").searchParams.get("offset"));
    const limit = Number(new URL(url, "http://pod").searchParams.get("limit"));
    const page = pages[Math.floor(offset / limit)] ?? [];
    return Promise.resolve({ ok: true, json: () => Promise.resolve(page) });
  });
  return { name: "test", instanceId: "tenant-1", fetch };
}

describe("getAllDatasetData", () => {
  it("returns everything from a single short page without a second request", async () => {
    const instance = instanceWithPages([[row("a"), row("b")]]);
    const rows = await getAllDatasetData("ds-1", instance);
    expect(rows).toEqual([row("a"), row("b")]);
    expect(instance.fetch).toHaveBeenCalledTimes(1);
  });

  it("hands the caller's abort signal to every page request", async () => {
    const firstPage = Array.from({ length: 1000 }, (_, i) => row(`p1-${i}`));
    const instance = instanceWithPages([firstPage, [row("p2-0")]]);
    const { signal } = new AbortController();

    await fetchAllDatasetDataPages("ds-1", instance, signal);

    expect(instance.fetch).toHaveBeenCalledTimes(2);
    for (const call of instance.fetch.mock.calls) expect(call[1]).toEqual({ signal });
  });

  it("returns an empty list for a dataset with no data", async () => {
    const instance = instanceWithPages([[]]);
    const rows = await getAllDatasetData("ds-1", instance);
    expect(rows).toEqual([]);
    expect(instance.fetch).toHaveBeenCalledTimes(1);
  });

  it("pages through a dataset larger than one page", async () => {
    const firstPage = Array.from({ length: 1000 }, (_, i) => row(`p1-${i}`));
    const secondPage = [row("p2-0"), row("p2-1")];
    const instance = instanceWithPages([firstPage, secondPage]);
    const rows = await getAllDatasetData("ds-1", instance);
    expect(rows).toHaveLength(1002);
    expect(rows[1000]).toEqual(row("p2-0"));
    expect(instance.fetch).toHaveBeenCalledTimes(2);
  });

  it("stops after one request when a pre-1.6.1 pod ignores offset and echoes the same full list back", async () => {
    // A pod that predates limit/offset support returns the identical full
    // list no matter what offset is requested — simulate that here instead
    // of the page-keyed stub, which would hide the bug this guards against.
    const fullList = Array.from({ length: 1200 }, (_, i) => row(`x-${i}`));
    const fetch = jest.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve(fullList) });
    const instance: CogneeInstance = { name: "test", instanceId: "tenant-1", fetch };

    const rows = await getAllDatasetData("ds-1", instance);

    expect(rows).toEqual(fullList);
    expect(instance.fetch).toHaveBeenCalledTimes(2); // page 0, then the duplicate-detection page
  });

  it("propagates a mid-page fetch failure instead of returning a truncated list", async () => {
    const fetch = jest
      .fn()
      .mockResolvedValueOnce({ ok: true, json: () => Promise.resolve(Array.from({ length: 1000 }, (_, i) => row(`p1-${i}`))) })
      .mockRejectedValueOnce(new Error("pod unreachable"));
    const instance: CogneeInstance = { name: "test", instanceId: "tenant-1", fetch };

    await expect(getAllDatasetData("ds-1", instance)).rejects.toThrow("pod unreachable");
  });

  it("keeps paging when one row on a full page is malformed, instead of mistaking a filtered-short page for the end", async () => {
    // 1007 rows total: page 0 is a full 1000-row page with one row missing an
    // id (dropped by isDatasetDataRow), page 1 has the remaining 7. Pagination
    // must key off the raw page length (1000), not the post-filter length
    // (999) — otherwise page 0 alone looks short and the last 7 rows are
    // silently dropped.
    const malformedPage = Array.from({ length: 1000 }, (_, i) =>
      i === 500 ? { name: "no-id-here" } : row(`p1-${i}`),
    );
    const secondPage = Array.from({ length: 7 }, (_, i) => row(`p2-${i}`));
    const instance = instanceWithPages([malformedPage, secondPage]);

    const rows = await getAllDatasetData("ds-1", instance);

    expect(rows).toHaveLength(1006); // 999 valid on page 0 + 7 on page 1
    expect(instance.fetch).toHaveBeenCalledTimes(2);
  });

  it("dedupes a row repeated across a page boundary by a created_at-DESC reorder", async () => {
    // A row inserted mid-walk shifts everything after it down one slot in the
    // DESC order, so the offset-1000 request can hand back the same row that
    // was already the last item on page 0.
    const firstPage = Array.from({ length: 1000 }, (_, i) => row(`p1-${i}`));
    const secondPage = [row("p1-999"), row("p2-0"), row("p2-1")]; // p1-999 repeats
    const instance = instanceWithPages([firstPage, secondPage]);

    const rows = await getAllDatasetData("ds-1", instance);

    expect(rows).toHaveLength(1002); // 1000 + 2 new, the repeat isn't counted twice
    expect(rows.filter((r) => r.id === "p1-999")).toHaveLength(1);
  });

  it("reports capped=true and warns once the page cap is hit", async () => {
    const warnSpy = jest.spyOn(console, "warn").mockImplementation(() => {});
    // Every page is a full 1000-row batch with a distinct first id, so the
    // loop never sees a short page or a repeated first id and must run to
    // the MAX_PAGES cap.
    let call = 0;
    const fetch = jest.fn().mockImplementation(() => {
      const page = call++;
      const batch = Array.from({ length: 1000 }, (_, i) => row(`pg${page}-${i}`));
      return Promise.resolve({ ok: true, json: () => Promise.resolve(batch) });
    });
    const instance: CogneeInstance = { name: "test", instanceId: "tenant-1", fetch };

    const result = await fetchAllDatasetDataPages("ds-1", instance);

    expect(result.capped).toBe(true);
    expect(result.rows).toHaveLength(200 * 1000);
    expect(warnSpy).toHaveBeenCalledTimes(1);
    warnSpy.mockRestore();
  });
});
