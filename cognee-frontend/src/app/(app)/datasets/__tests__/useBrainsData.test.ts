import { act, renderHook, waitFor } from "@testing-library/react";
import getDatasets from "@/modules/datasets/getDatasets";
import getDatasetDataCount from "@/modules/datasets/getDatasetDataCount";
import createDataset from "@/modules/datasets/createDataset";
import { loadGraphModelsConfig } from "@/modules/configuration/userConfiguration";
import { useBrainsData } from "@/app/(app)/datasets/useBrainsData";
import { PENDING_DATASET_PREFIX, isPendingDataset } from "@/app/(app)/datasets/brainsTypes";

jest.mock("@/modules/datasets/getDatasets");
jest.mock("@/modules/datasets/getDatasetDataCount");
jest.mock("@/modules/datasets/createDataset");
jest.mock("@/modules/datasets/deleteDataset");
jest.mock("@/modules/datasets/deleteDatasetData");
jest.mock("@/modules/datasets/cognifyDataset");
jest.mock("@/modules/configuration/userConfiguration");
jest.mock("@/modules/ingestion/useBrainUpload", () => ({
  useBrainUpload: () => ({ isUploading: false, stage: "idle", progress: { stage: "idle" }, upload: jest.fn() }),
}));
jest.mock("@/modules/billing/useLowBalanceUploadWarning", () => ({
  useLowBalanceUploadWarning: () => ({ pendingWarning: null, isEstimating: false, confirmUpload: jest.fn(), cancel: jest.fn() }),
}));
jest.mock("@/modules/datasets/useDatasetStatuses", () => ({
  useDatasetStatuses: () => ({ statusDetails: {} }),
}));jest.mock("@/modules/datasets/useDatasetProcessing", () => ({
  useDatasetProcessing: () => ({ data: undefined, isError: false, refetch: jest.fn() }),
}));

jest.mock("@/modules/analytics", () => ({ trackEvent: jest.fn() }));

const mockCogniInstance = { instanceId: "t-1", fetch: jest.fn().mockResolvedValue({ ok: false }) };
jest.mock("@/modules/tenant/TenantProvider", () => ({
  useCogniInstance: () => ({ cogniInstance: mockCogniInstance, isInitializing: false }),
}));

const mockRefreshFilterDatasets = jest.fn();
jest.mock("@/ui/layout/FilterContext", () => ({
  useFilter: () => ({ datasets: [], refreshDatasets: mockRefreshFilterDatasets }),
}));

beforeEach(() => {
  jest.mocked(loadGraphModelsConfig).mockResolvedValue({ models: [], outdatedDatasets: [] });
  mockCogniInstance.fetch.mockResolvedValue({ ok: false });
});

afterEach(() => {
  jest.resetAllMocks();
});

describe("a dataset still being created", () => {
  it("survives a refresh that lands while the pod hasn't answered for it yet", async () => {
    jest.mocked(getDatasets).mockResolvedValue([]);
    jest.mocked(getDatasetDataCount).mockResolvedValue(0);
    let resolveCreate: (ds: { id: string; name: string }) => void = () => undefined;
    jest.mocked(createDataset).mockReturnValue(new Promise((resolve) => { resolveCreate = resolve; }));

    const { result } = renderHook(() => useBrainsData());
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    act(() => { result.current.setNewName("fresh"); });
    let createDone: Promise<boolean> = Promise.resolve(false);
    act(() => { createDone = result.current.handleCreate(null); });

    expect(result.current.datasets.some((d) => d.id.startsWith(PENDING_DATASET_PREFIX))).toBe(true);

    // A reload lands before the pod has answered the create — same shape as
    // handleRefresh racing handleCreate, or the mount load finishing late.
    await act(async () => { await result.current.handleRefresh(); });

    expect(result.current.datasets.some((d) => d.id.startsWith(PENDING_DATASET_PREFIX))).toBe(true);

    await act(async () => { resolveCreate({ id: "ds-1", name: "fresh" }); await createDone; });

    expect(result.current.datasets.some((d) => d.id.startsWith(PENDING_DATASET_PREFIX))).toBe(false);
    expect(result.current.datasets.some((d) => d.id === "ds-1")).toBe(true);
  });

  it("resolves handleCreate to false on a rejected name, so the caller never treats it as created", async () => {
    jest.mocked(getDatasets).mockResolvedValue([]);
    jest.mocked(getDatasetDataCount).mockResolvedValue(0);

    const { result } = renderHook(() => useBrainsData());
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    act(() => { result.current.setNewName("bad.name"); });
    let created = true;
    await act(async () => { created = await result.current.handleCreate(null); });

    expect(created).toBe(false);
    expect(createDataset).not.toHaveBeenCalled();
  });

  it("resolves handleCreate to true once the pod accepts it", async () => {
    jest.mocked(getDatasets).mockResolvedValue([]);
    jest.mocked(getDatasetDataCount).mockResolvedValue(0);
    jest.mocked(createDataset).mockResolvedValue({ id: "ds-1", name: "fresh" });

    const { result } = renderHook(() => useBrainsData());
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    act(() => { result.current.setNewName("fresh"); });
    let created = false;
    await act(async () => { created = await result.current.handleCreate(null); });

    expect(created).toBe(true);
  });

  it("removes its placeholder when the pod rejects the create, instead of leaving a row for a dataset that was never made", async () => {
    jest.mocked(getDatasets).mockResolvedValue([]);
    jest.mocked(getDatasetDataCount).mockResolvedValue(0);
    jest.mocked(createDataset).mockRejectedValue(new Error("pod refused"));

    const { result } = renderHook(() => useBrainsData());
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    act(() => { result.current.setNewName("fresh"); });
    await act(async () => { await result.current.handleCreate(null); });

    expect(result.current.datasets.some((d) => isPendingDataset(d.id))).toBe(false);
    expect(result.current.createError).toBe("Failed to create dataset. Please try again.");
  });

  it("never renders two rows for the same dataset when a refresh discovers it before create() itself resolves", async () => {
    jest.mocked(getDatasets)
      .mockResolvedValueOnce([]) // the mount load
      .mockResolvedValueOnce([{ id: "ds-1", name: "fresh" }]); // the pod already has it by the next refresh
    jest.mocked(getDatasetDataCount).mockResolvedValue(0);
    let resolveCreate: (ds: { id: string; name: string }) => void = () => undefined;
    jest.mocked(createDataset).mockReturnValue(new Promise((resolve) => { resolveCreate = resolve; }));

    const { result } = renderHook(() => useBrainsData());
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    act(() => { result.current.setNewName("fresh"); });
    let createDone: Promise<boolean> = Promise.resolve(false);
    act(() => { createDone = result.current.handleCreate(null); });

    // The refresh's own GET already reflects the pod having made it, ahead of
    // create()'s own promise resolving.
    await act(async () => { await result.current.handleRefresh(); });
    expect(result.current.datasets.filter((d) => d.name === "fresh")).toHaveLength(1);

    await act(async () => { resolveCreate({ id: "ds-1", name: "fresh" }); await createDone; });

    const ids = result.current.datasets.map((d) => d.id);
    expect(ids.filter((id) => id === "ds-1")).toHaveLength(1);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("keeps a dataset a create confirmed when an earlier-started reload answers only after it", async () => {
    let resolveStaleReload: (list: unknown[]) => void = () => undefined;
    jest.mocked(getDatasets)
      .mockResolvedValueOnce([]) // the mount load
      .mockReturnValueOnce(new Promise((resolve) => { resolveStaleReload = resolve; }));
    jest.mocked(getDatasetDataCount).mockResolvedValue(0);
    jest.mocked(createDataset).mockResolvedValue({ id: "ds-1", name: "fresh" });

    const { result } = renderHook(() => useBrainsData());
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    // A reload starts (nothing selected yet, so it won't also wait on doc refresh)...
    let refreshDone: Promise<void> = Promise.resolve();
    act(() => { refreshDone = result.current.handleRefresh(); });

    // ...and a create finishes before that reload's own request answers.
    act(() => { result.current.setNewName("fresh"); });
    await act(async () => { await result.current.handleCreate(null); });
    expect(result.current.selectedId).toBe("ds-1");

    // The reload's answer was a server snapshot taken before the create —
    // it says nothing about "ds-1" — and must not erase it on arrival.
    await act(async () => { resolveStaleReload([]); await refreshDone; });

    expect(result.current.datasets.some((d) => d.id === "ds-1")).toBe(true);
    expect(result.current.selectedDataset).not.toBeNull();
  });

  it("keeps the pending placeholder when the outer catch fires on an unexpected failure, rather than wiping the whole list", async () => {
    jest.mocked(getDatasets).mockResolvedValue([]);
    jest.mocked(getDatasetDataCount).mockResolvedValue(0);
    let resolveCreate: (ds: { id: string; name: string }) => void = () => undefined;
    jest.mocked(createDataset).mockReturnValue(new Promise((resolve) => { resolveCreate = resolve; }));

    const { result } = renderHook(() => useBrainsData());
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    act(() => { result.current.setNewName("fresh"); });
    act(() => { result.current.handleCreate(null); });
    expect(result.current.datasets.some((d) => isPendingDataset(d.id))).toBe(true);

    // The status poll resolves, but its body isn't valid JSON — unlike a
    // request-level failure (already handled further up), nothing catches
    // this except loadDatasets' own outer catch.
    mockCogniInstance.fetch.mockResolvedValueOnce({ ok: true, json: () => Promise.reject(new Error("bad json")) });
    await act(async () => { await result.current.handleRefresh(); });

    expect(result.current.datasetsError).toBe(true);
    expect(result.current.datasets.some((d) => isPendingDataset(d.id))).toBe(true);

    await act(async () => { resolveCreate({ id: "ds-1", name: "fresh" }); });
  });
});
