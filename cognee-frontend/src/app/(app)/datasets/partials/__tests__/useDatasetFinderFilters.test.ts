import { act, renderHook } from "@testing-library/react";
import useDatasetFinderFilters from "@/app/(app)/datasets/partials/useDatasetFinderFilters";
import useDatasetFacets from "@/app/(app)/datasets/partials/useDatasetFacets";
import { ALL_FACET_ID, PERSONAL_FACET_ID, type DatasetFacet } from "@/app/(app)/datasets/partials/datasetFacets";
import type { Dataset } from "@/app/(app)/datasets/brainsTypes";
import type { DatasetAccess } from "@/modules/permissions/types";

jest.mock("@/app/(app)/datasets/partials/useDatasetFacets");

const mockUseDatasetFacets = jest.mocked(useDatasetFacets);

function dataset(id: string): Dataset {
  return { id, name: id, documents: 0, status: "completed" };
}

function accessEntry(datasetId: string): DatasetAccess {
  return { datasetId, datasetName: datasetId, ownerId: "me", isOwner: true, permissions: ["read"], sources: [], sharedWith: null };
}

const ALL_FACET: DatasetFacet = { id: ALL_FACET_ID, kind: "all", label: "All datasets", hint: "", datasetIds: new Set() };

afterEach(() => {
  jest.resetAllMocks();
});

describe("a dataset the access list hasn't reached yet", () => {
  const personalFacet: DatasetFacet = { id: PERSONAL_FACET_ID, kind: "personal", label: "Personal", hint: "", datasetIds: new Set(["mine"]) };

  beforeEach(() => {
    mockUseDatasetFacets.mockReturnValue({
      facets: [ALL_FACET, personalFacet],
      accessById: new Map([["mine", accessEntry("mine")]]),
      status: "ready",
      refresh: jest.fn(),
    });
  });

  it("counts a still-being-created placeholder under Personal right away, before the access list even knows it exists", () => {
    const { result } = renderHook(() => useDatasetFinderFilters([dataset("mine"), dataset("pending-fresh")], new Set()));

    const personal = result.current.facets.find((facet) => facet.id === PERSONAL_FACET_ID);
    expect([...(personal?.datasetIds ?? [])].sort()).toEqual(["mine", "pending-fresh"]);
  });

  it("counts a dataset this session just had confirmed under Personal, by the same route", () => {
    const { result } = renderHook(() =>
      useDatasetFinderFilters([dataset("mine"), dataset("ds-1")], new Set(), new Set(["ds-1"])));

    const personal = result.current.facets.find((facet) => facet.id === PERSONAL_FACET_ID);
    expect([...(personal?.datasetIds ?? [])].sort()).toEqual(["ds-1", "mine"]);
  });

  it("stays visible once Personal is selected, so the dataset you just created is never hidden by the rail", () => {
    const { result } = renderHook(() => useDatasetFinderFilters([dataset("mine"), dataset("pending-fresh")], new Set()));

    act(() => { result.current.selectFacet(PERSONAL_FACET_ID); });

    expect(result.current.visible.map((entry) => entry.id).sort()).toEqual(["mine", "pending-fresh"]);
  });

  it("never counts a dataset merely missing from the access list as Personal — that can be a share the list hasn't loaded yet, not something this session created", () => {
    // "shared-with-me" is neither pending nor confirmed by a create this
    // session — it's simply absent from accessById, the same shape as an
    // access refresh in flight, one that failed after a good load, or a row
    // the access response dropped for being malformed.
    const { result } = renderHook(() => useDatasetFinderFilters([dataset("mine"), dataset("shared-with-me")], new Set()));

    const personal = result.current.facets.find((facet) => facet.id === PERSONAL_FACET_ID);
    expect(personal?.datasetIds.has("shared-with-me")).toBe(false);
  });
});

describe("a facet that disappears while selected", () => {
  const groupFacet: DatasetFacet = { id: "g-eng", kind: "group", label: "Engineering", hint: "", datasetIds: new Set(["shared-ds"]) };

  it("falls back to All rather than silently filtering by nothing", () => {
    mockUseDatasetFacets.mockReturnValue({ facets: [ALL_FACET, groupFacet], accessById: new Map(), status: "ready", refresh: jest.fn() });
    const { result, rerender } = renderHook(
      ({ datasets }) => useDatasetFinderFilters(datasets, new Set()),
      { initialProps: { datasets: [dataset("shared-ds")] } },
    );

    act(() => { result.current.selectFacet("g-eng"); });
    expect(result.current.facetId).toBe("g-eng");

    // The rail no longer offers it (left the group, access list caught up, etc).
    mockUseDatasetFacets.mockReturnValue({ facets: [ALL_FACET], accessById: new Map(), status: "ready", refresh: jest.fn() });
    rerender({ datasets: [dataset("shared-ds")] });

    expect(result.current.facetId).toBe(ALL_FACET_ID);
  });

  it("does not resurrect the vanished selection if the same facet id reappears later", () => {
    mockUseDatasetFacets.mockReturnValue({ facets: [ALL_FACET, groupFacet], accessById: new Map(), status: "ready", refresh: jest.fn() });
    const { result, rerender } = renderHook(
      ({ datasets }) => useDatasetFinderFilters(datasets, new Set()),
      { initialProps: { datasets: [dataset("shared-ds")] } },
    );
    act(() => { result.current.selectFacet("g-eng"); });

    mockUseDatasetFacets.mockReturnValue({ facets: [ALL_FACET], accessById: new Map(), status: "ready", refresh: jest.fn() });
    rerender({ datasets: [dataset("shared-ds")] });
    expect(result.current.facetId).toBe(ALL_FACET_ID);

    // Engineering is back (rejoined the group, access list refreshed), so the
    // forgotten selection must not silently reclaim it.
    mockUseDatasetFacets.mockReturnValue({ facets: [ALL_FACET, groupFacet], accessById: new Map(), status: "ready", refresh: jest.fn() });
    rerender({ datasets: [dataset("shared-ds")] });

    expect(result.current.facetId).toBe(ALL_FACET_ID);
  });
});

describe("a facet selection during a refresh that is still in flight", () => {
  const groupFacet: DatasetFacet = { id: "g-eng", kind: "group", label: "Engineering", hint: "", datasetIds: new Set(["shared-ds"]) };

  it("does not fall back to All just because a mid-refresh facets list doesn't carry it yet", () => {
    mockUseDatasetFacets.mockReturnValue({ facets: [ALL_FACET, groupFacet], accessById: new Map(), status: "ready", refresh: jest.fn() });
    const { result, rerender } = renderHook(
      ({ datasets }) => useDatasetFinderFilters(datasets, new Set()),
      { initialProps: { datasets: [dataset("shared-ds")] } },
    );
    act(() => { result.current.selectFacet("g-eng"); });

    // A refresh kicked off: useDatasetFacets reports "loading" and its facets
    // are momentarily stale, missing "g-eng" — not proof the group is gone.
    mockUseDatasetFacets.mockReturnValue({ facets: [ALL_FACET], accessById: new Map(), status: "loading", refresh: jest.fn() });
    rerender({ datasets: [dataset("shared-ds")] });
    expect(result.current.facetId).toBe("g-eng");

    // The refresh lands and "g-eng" is still absent — now it really is gone.
    mockUseDatasetFacets.mockReturnValue({ facets: [ALL_FACET], accessById: new Map(), status: "ready", refresh: jest.fn() });
    rerender({ datasets: [dataset("shared-ds")] });
    expect(result.current.facetId).toBe(ALL_FACET_ID);
  });
});
