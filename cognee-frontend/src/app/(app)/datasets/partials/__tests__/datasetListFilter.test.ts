import { filterDatasetList } from "@/app/(app)/datasets/partials/datasetListFilter";
import type { DisplayStatus } from "@/app/(app)/datasets/brainsTypes";

function dataset(id: string, status: DisplayStatus): { id: string; name: string; status: DisplayStatus } {
  return { id, name: id, status };
}

const READY = dataset("company-brain", "completed");
const RUNNING = dataset("product-docs", "running");
const FAILED = dataset("crm", "failed_insufficient_credits");
const OUTDATED = dataset("wiki", "completed");
const ALL = [READY, RUNNING, FAILED, OUTDATED];
const OUTDATED_IDS = new Set(["wiki"]);

describe("filterDatasetList", () => {
  it("keeps everything under All", () => {
    expect(filterDatasetList(ALL, "all", "", OUTDATED_IDS)).toEqual(ALL);
  });

  it("counts an outdated dataset as needing attention, not as ready", () => {
    expect(filterDatasetList(ALL, "ready", "", OUTDATED_IDS)).toEqual([READY]);
    expect(filterDatasetList(ALL, "attention", "", OUTDATED_IDS)).toEqual([FAILED, OUTDATED]);
  });

  it("puts queued and running datasets under Processing", () => {
    expect(filterDatasetList([...ALL, dataset("queued", "pending")], "processing", "", OUTDATED_IDS).map((d) => d.id)).toEqual([
      "product-docs",
      "queued",
    ]);
  });

  it("matches the search case-insensitively within the status filter", () => {
    expect(filterDatasetList(ALL, "all", "  BRAIN ", OUTDATED_IDS)).toEqual([READY]);
  });

  it("matches a mixed-case dataset name against a mixed-case query", () => {
    const mixedCase = dataset("CRM-Pipeline", "completed");
    expect(filterDatasetList([mixedCase], "all", "Pipeline", OUTDATED_IDS)).toEqual([mixedCase]);
  });

  it("counts a plain failed build as needing attention, same as an out-of-credits one", () => {
    const plainFailed = dataset("ingest", "failed");
    expect(filterDatasetList([plainFailed], "attention", "", OUTDATED_IDS)).toEqual([plainFailed]);
  });

  it("puts an empty dataset under Ready: nothing is wrong with it and nothing is left to process", () => {
    const empty = dataset("fresh", "empty");
    expect(filterDatasetList([empty], "ready", "", OUTDATED_IDS)).toEqual([empty]);
    expect(filterDatasetList([empty], "processing", "", OUTDATED_IDS)).toEqual([]);
    expect(filterDatasetList([empty], "attention", "", OUTDATED_IDS)).toEqual([]);
  });

  it("never hides a loading dataset behind a status chip: there's nothing yet to judge it against", () => {
    const loading = dataset("refreshing", "loading");
    expect(filterDatasetList([loading], "ready", "", OUTDATED_IDS)).toEqual([loading]);
    expect(filterDatasetList([loading], "processing", "", OUTDATED_IDS)).toEqual([loading]);
    expect(filterDatasetList([loading], "attention", "", OUTDATED_IDS)).toEqual([loading]);
  });

  it("still applies the search to a loading dataset, name matching isn't a status judgment", () => {
    const loading = dataset("refreshing", "loading");
    expect(filterDatasetList([loading], "ready", "nope", OUTDATED_IDS)).toEqual([]);
  });

  it("counts an outdated dataset as needing attention even while it's still loading — outdated is already known, it doesn't need the status to resolve first", () => {
    const loadingButOutdated = dataset("wiki", "loading");
    expect(filterDatasetList([loadingButOutdated], "ready", "", OUTDATED_IDS)).toEqual([]);
    expect(filterDatasetList([loadingButOutdated], "attention", "", OUTDATED_IDS)).toEqual([loadingButOutdated]);
  });
});
