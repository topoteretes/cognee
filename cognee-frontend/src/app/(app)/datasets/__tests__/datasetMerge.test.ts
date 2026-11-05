import { dropResolvedPlaceholders, dedupeDatasetsById, type Dataset } from "@/app/(app)/datasets/brainsTypes";

function dataset(id: string, name: string): Dataset {
  return { id, name, documents: -1, status: "loading" };
}

describe("dropResolvedPlaceholders", () => {
  it("drops a placeholder once a real dataset shares its name", () => {
    const result = dropResolvedPlaceholders([dataset("pending-1", "fresh"), dataset("ds-1", "fresh")]);

    expect(result).toEqual([dataset("ds-1", "fresh")]);
  });

  it("leaves a placeholder alone while nothing resolved for it yet", () => {
    const result = dropResolvedPlaceholders([dataset("pending-1", "fresh"), dataset("ds-2", "other")]);

    expect(result).toEqual([dataset("pending-1", "fresh"), dataset("ds-2", "other")]);
  });

  it("leaves two unrelated real datasets alone", () => {
    const result = dropResolvedPlaceholders([dataset("ds-1", "fresh"), dataset("ds-2", "other")]);

    expect(result).toEqual([dataset("ds-1", "fresh"), dataset("ds-2", "other")]);
  });
});

describe("dedupeDatasetsById", () => {
  it("keeps one row per id", () => {
    const result = dedupeDatasetsById([dataset("ds-1", "fresh"), dataset("ds-1", "fresh")]);

    expect(result).toEqual([dataset("ds-1", "fresh")]);
  });

  it("keeps the first occurrence, since it's at least as informative as a later stub", () => {
    const richer = { ...dataset("ds-1", "fresh"), documents: 3, status: "completed" as const };
    const stub = dataset("ds-1", "fresh");

    expect(dedupeDatasetsById([richer, stub])).toEqual([richer]);
  });

  it("leaves distinct ids untouched", () => {
    const result = dedupeDatasetsById([dataset("ds-1", "fresh"), dataset("ds-2", "other")]);

    expect(result).toEqual([dataset("ds-1", "fresh"), dataset("ds-2", "other")]);
  });
});
